#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download full monthly Sea Ice Concentration (SIC) files from NSIDC-0051 (V002)
using NASA Earthdata CMR granule search + authenticated download.

Fix: CMR search should NOT use Earthdata Basic Auth (otherwise may get 401).
Only the actual file downloads need Earthdata auth.

Default: Arctic (north) monthly mean, 25km polar stereographic grid,
1979-01 to 2022-12 (as used in the paper).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import requests
from netrc import netrc

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# NSIDC-0051, Version 2 (Sea Ice Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS Passive Microwave Data)
# Collection concept id from CMR virtual directory.
COLLECTION_CONCEPT_ID = "C3177837840-NSIDC_CPRD"

# Monthly files: NSIDC0051_SEAICE_PS_N25km_YYYYMM_v2.0.nc or ..._S25km_YYYYMM_v2.0.nc
MONTHLY_RE = re.compile(
    r"^NSIDC0051_SEAICE_PS_(?P<hem>[NS])25km_(?P<ym>\d{6})_v2\.0\.nc$"
)


@dataclass
class Granule:
    title: str
    url: str


def parse_ym(s: str) -> datetime:
    """Parse YYYYMM to datetime at first day of month."""
    return datetime.strptime(s, "%Y%m")


def ym_range(start_ym: str, end_ym: str) -> Tuple[str, str]:
    """Validate YYYYMM inputs; return normalized strings."""
    s = parse_ym(start_ym)
    e = parse_ym(end_ym)
    if e < s:
        raise ValueError(f"end_ym ({end_ym}) must be >= start_ym ({start_ym})")
    return start_ym, end_ym


def earthdata_auth() -> Tuple[str, str]:
    """
    Read ~/.netrc for Earthdata URS credentials.
    Need entry for 'urs.earthdata.nasa.gov'.
    """
    nrc = netrc()
    auth = nrc.authenticators("urs.earthdata.nasa.gov")
    if not auth:
        raise RuntimeError(
            "Cannot find Earthdata credentials in ~/.netrc for 'urs.earthdata.nasa.gov'.\n"
            "Expected something like:\n"
            "machine urs.earthdata.nasa.gov\n"
            "  login YOUR_USERNAME\n"
            "  password YOUR_PASSWORD\n"
            "Also ensure: chmod 600 ~/.netrc"
        )
    username, _, password = auth
    if not username or not password:
        raise RuntimeError("Invalid ~/.netrc entry for urs.earthdata.nasa.gov")
    return username, password


def cmr_search_monthly_granules(
    hemisphere: str,
    start_ym: str,
    end_ym: str,
    session: requests.Session,
    page_size: int = 2000,
) -> List[Granule]:
    """
    Query CMR for granules in the temporal range, then filter to monthly files for hemisphere.
    IMPORTANT: Use a session WITHOUT Earthdata auth for CMR search.
    """
    hemisphere = hemisphere.upper()
    if hemisphere not in ("N", "S"):
        raise ValueError("hemisphere must be 'N' or 'S'")

    # Convert to ISO time range.
    start_dt = datetime.strptime(start_ym + "01", "%Y%m%d")
    end_month_dt = datetime.strptime(end_ym + "01", "%Y%m%d")
    if end_month_dt.month == 12:
        next_month = datetime(end_month_dt.year + 1, 1, 1)
    else:
        next_month = datetime(end_month_dt.year, end_month_dt.month + 1, 1)

    # CMR temporal end is exclusive here (we give first day of next month)
    temporal = f"{start_dt.strftime('%Y-%m-%dT00:00:00Z')},{next_month.strftime('%Y-%m-%dT00:00:00Z')}"

    granules: List[Granule] = []
    page_num = 1

    while True:
        params = {
            "collection_concept_id": COLLECTION_CONCEPT_ID,
            "temporal": temporal,
            "page_size": page_size,
            "page_num": page_num,
        }
        r = session.get(CMR_GRANULES_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            break

        for e in entries:
            title = e.get("title", "")
            m = MONTHLY_RE.match(title)
            if not m:
                continue
            if m.group("hem") != hemisphere:
                continue

            url = None
            for link in e.get("links", []):
                href = link.get("href")
                if not href:
                    continue
                rel = link.get("rel", "") or ""
                if href.endswith(".nc") and (
                    "data#" in rel or "download#" in rel or "opendap#" in rel
                ):
                    url = href
                    break

            if url is None:
                for link in e.get("links", []):
                    href = link.get("href")
                    if href and href.endswith(".nc"):
                        url = href
                        break

            if url:
                granules.append(Granule(title=title, url=url))

        page_num += 1

    granules.sort(key=lambda g: g.title)
    return granules


def download_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    max_retries: int = 5,
    chunk_size: int = 1024 * 1024,
) -> None:
    """
    Stream download to temp file then atomically rename.
    Skip if already exists and size>0.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = r.headers.get("Content-Length")
                total_mb = (int(total) / 1024 / 1024) if total and total.isdigit() else None

                bytes_written = 0
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bytes_written += len(chunk)

                if bytes_written == 0:
                    raise RuntimeError("Downloaded 0 bytes (possible auth/redirect issue).")

                tmp_size_mb = bytes_written / 1024 / 1024
                if total_mb is not None and abs(tmp_size_mb - total_mb) > max(5, 0.02 * total_mb):
                    raise RuntimeError(
                        f"Size mismatch: got {tmp_size_mb:.1f} MB, expected {total_mb:.1f} MB"
                    )

            tmp_path.replace(out_path)
            return

        except Exception as ex:
            if attempt == max_retries:
                raise
            sleep_s = min(2 ** attempt, 30)
            print(f"[WARN] download failed ({attempt}/{max_retries}) for {url}: {ex}. Retry in {sleep_s}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download after retries: {url}")


def main():
    parser = argparse.ArgumentParser(
        description="Download NSIDC-0051 V002 monthly SIC NetCDF files via CMR."
    )
    parser.add_argument("--out_dir", type=str, default="data/raw/nsidc_sic",
                        help="Output directory for .nc files.")
    parser.add_argument("--hemisphere", type=str, default="N",
                        help="N for Arctic (north), S for Antarctic (south). Default N.")
    parser.add_argument("--start_ym", type=str, default="197901",
                        help="Start month in YYYYMM (inclusive). Default 197901.")
    parser.add_argument("--end_ym", type=str, default="202212",
                        help="End month in YYYYMM (inclusive). Default 202212.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list files, do not download.")
    args = parser.parse_args()

    start_ym, end_ym = ym_range(args.start_ym, args.end_ym)

    username, password = earthdata_auth()

    # ---- Session 1: CMR search (NO auth) ----
    cmr_sess = requests.Session()
    cmr_sess.headers.update({"User-Agent": "seaice_tgrs-nsidc0051-downloader/1.0"})

    # ---- Session 2: actual file download (WITH Earthdata auth) ----
    dl_sess = requests.Session()
    dl_sess.auth = (username, password)
    dl_sess.headers.update({"User-Agent": "seaice_tgrs-nsidc0051-downloader/1.0"})

    try:
        print(f"[INFO] Query CMR granules: hem={args.hemisphere.upper()} {start_ym}-{end_ym}")
        granules = cmr_search_monthly_granules(
            hemisphere=args.hemisphere,
            start_ym=start_ym,
            end_ym=end_ym,
            session=cmr_sess,
        )
        print(f"[INFO] Found {len(granules)} monthly granules.")

        if args.dry_run:
            for g in granules:
                print(g.title, "->", g.url)
            return

        out_dir = Path(args.out_dir)
        n_ok = 0
        for i, g in enumerate(granules, 1):
            out_path = out_dir / g.title
            if out_path.exists() and out_path.stat().st_size > 0:
                n_ok += 1
                continue

            print(f"[{i}/{len(granules)}] Download {g.title}")
            download_file(dl_sess, g.url, out_path)
            n_ok += 1

        print(f"[DONE] {n_ok}/{len(granules)} files present in {out_dir}")

    finally:
        cmr_sess.close()
        dl_sess.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user.")
        sys.exit(130)
