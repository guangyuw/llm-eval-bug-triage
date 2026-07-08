"""
Download Mozilla Bugzilla bug reports, including RESOLVED DUPLICATE bugs whose
`dupe_of` gives a HARD LABEL for duplicate/near-duplicate retrieval evaluation.

Source: Bugzilla REST API (public bugs, no auth).
  - search:   https://bugzilla.mozilla.org/rest/bug?...
  - comments: https://bugzilla.mozilla.org/rest/bug/{id}/comment   (first = description)

Outputs:
  bugs.csv         one row per bug: id, summary, description, component, product,
                   severity, resolution, creation_time, dupe_of, role
  dup_pairs.csv    (dup_id, master_id) hard labels for retrieval eval

Usage:  python download_bugzilla.py [--dups 1500] [--extra 1500] [--with-desc]
"""
from __future__ import annotations
import argparse, csv, gzip, json, os, threading, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "job2026-resume-research contact: guangyu.research@example.com"
BASE = "https://bugzilla.mozilla.org/rest"
FIELDS = "id,summary,component,product,severity,resolution,creation_time,dupe_of"
DESC_CACHE = "desc_cache.json"


def fetch_json(url: str, timeout: int = 45) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8", "replace"))


def search_bugs(params: str, want: int, page: int = 250) -> list[dict]:
    out, offset = [], 0
    while len(out) < want:
        url = (f"{BASE}/bug?{params}&include_fields={urllib.parse.quote(FIELDS)}"
               f"&limit={page}&offset={offset}&order=bug_id%20DESC")
        bugs = fetch_json(url).get("bugs", [])
        if not bugs:
            break
        out.extend(bugs); offset += page
        print(f"    fetched {len(out)}..."); time.sleep(0.2)
    return out[:want]


def fetch_by_ids(ids: list[int]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        q = "&".join(f"id={b}" for b in batch)
        url = f"{BASE}/bug?{q}&include_fields={urllib.parse.quote(FIELDS)}"
        out.extend(fetch_json(url).get("bugs", [])); time.sleep(0.2)
    return out


def load_desc_cache() -> dict:
    return json.load(open(DESC_CACHE)) if os.path.exists(DESC_CACHE) else {}


def _fetch_one_desc(bid: int) -> tuple[str, str]:
    """First comment = the reporter's description."""
    try:
        c = fetch_json(f"{BASE}/bug/{bid}/comment")
        comments = c["bugs"][str(bid)]["comments"]
        return str(bid), (comments[0]["text"][:4000] if comments else "")
    except Exception:
        return str(bid), ""


def fetch_descriptions(ids: list[int], cache: dict, workers: int = 12) -> dict:
    """First comment = the reporter's description. Concurrent, cached + resumable."""
    todo = [i for i in ids if str(i) not in cache]
    print(f"  [desc] {len(todo)} to fetch ({len(ids) - len(todo)} cached), {workers} workers")
    done, lock = 0, threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_fetch_one_desc, bid) for bid in todo]
        for fut in as_completed(futures):
            k, text = fut.result()
            with lock:
                cache[k] = text
                done += 1
                if done % 500 == 0:
                    json.dump(cache, open(DESC_CACHE, "w"))
                    print(f"    desc {done}/{len(todo)}")
    json.dump(cache, open(DESC_CACHE, "w"))
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dups", type=int, default=1500)
    ap.add_argument("--extra", type=int, default=1500)
    ap.add_argument("--with-desc", action="store_true",
                    help="also fetch first-comment descriptions (slower, richer text)")
    args = ap.parse_args()

    prod = "product=Firefox&product=Core&product=Thunderbird"

    print("[bugzilla] fetching RESOLVED DUPLICATE bugs (hard labels)...")
    dups = search_bugs(f"{prod}&resolution=DUPLICATE", args.dups)
    dups = [b for b in dups if b.get("dupe_of")]
    print(f"  got {len(dups)} duplicates with dupe_of")

    print("[bugzilla] fetching master bugs they point to...")
    master_ids = sorted({int(b["dupe_of"]) for b in dups})
    masters = fetch_by_ids(master_ids)
    print(f"  got {len(masters)} masters")

    print("[bugzilla] fetching extra RESOLVED FIXED bugs (corpus diversity)...")
    extra = search_bugs(f"{prod}&resolution=FIXED", args.extra)
    print(f"  got {len(extra)} extra")

    # Merge unique by id; tag role for later analysis.
    by_id: dict[int, dict] = {}
    for b in masters:
        b["role"] = "master"; by_id[b["id"]] = b
    for b in extra:
        by_id.setdefault(b["id"], {**b, "role": "corpus"})
    for b in dups:
        by_id[b["id"]] = {**b, "role": "dup"}

    # Always reuse any descriptions we've already cached; only fetch more when asked.
    desc = load_desc_cache()
    if args.with_desc:
        desc = fetch_descriptions(sorted(by_id), desc)

    with open("bugs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "summary", "description", "component",
                                          "product", "severity", "resolution",
                                          "creation_time", "dupe_of", "role"])
        w.writeheader()
        for bid, b in by_id.items():
            w.writerow({"id": bid, "summary": b.get("summary", ""),
                        "description": desc.get(str(bid), ""),
                        "component": b.get("component", ""), "product": b.get("product", ""),
                        "severity": b.get("severity", ""), "resolution": b.get("resolution", ""),
                        "creation_time": b.get("creation_time", ""),
                        "dupe_of": b.get("dupe_of", ""), "role": b.get("role", "")})

    # dup->master pairs where BOTH sides are present in the corpus (usable labels).
    present = set(by_id)
    pairs = [(b["id"], int(b["dupe_of"])) for b in dups if int(b["dupe_of"]) in present]
    with open("dup_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dup_id", "master_id"]); w.writerows(pairs)

    yrs = sorted({b.get("creation_time", "")[:4] for b in by_id.values() if b.get("creation_time")})
    print(f"[bugzilla] wrote {len(by_id):,} bugs -> bugs.csv | {len(pairs):,} usable dup pairs "
          f"-> dup_pairs.csv | years {yrs[0] if yrs else '?'}–{yrs[-1] if yrs else '?'}")


if __name__ == "__main__":
    main()
