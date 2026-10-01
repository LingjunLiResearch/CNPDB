# -*- coding: utf-8 -*-
"""Maintain the standing "papers to check" reminder list and its email body.

The monthly literature scan surfaces crustacean-neuropeptide papers a curator
should look at. This module keeps a persistent list of those papers and, each
month:

* **adds** newly shortlisted papers that aren't already listed, and
* **drops** any paper whose DOI now appears in the database -- i.e. it has been
  incorporated, so it no longer needs chasing.

The remaining papers are formatted into a plain-text email body. A paper stays on
the list (and in every monthly email) until its DOI shows up in the database.

Below the list, the email also names every *other* paper this month's search
found (the ``--found`` files) that did not pass the crustacean + neuropeptide
filter, ranked by relevance score, so nothing the search returns is hidden.

Incorporation is detected by **DOI match** against the database's ``DOI`` column,
so it only works if the source DOI is recorded when sequences are added.

CLI (run by the monthly workflow)::

    python -m DataCuration.notify_papers reconcile \
        --shortlist DataCuration/outputs/shortlist_2026-08-01.csv \
        --found DataCuration/outputs/new_papers_2026-08-01.csv \
        --list DataCuration/outputs/papers_to_check.csv \
        --body-out email_body.txt
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataCuration.cnpdb_qc import DEFAULT_DB, OUTPUTS_DIR, load_database  # noqa: E402
from DataCuration.lit_mining import known_dois, normalize_doi, paper_key  # noqa: E402

COLUMNS = ["doi", "pmid", "title", "first_flagged"]
DEFAULT_LIST = os.path.join(OUTPUTS_DIR, "papers_to_check.csv")


def load_list(path: str = DEFAULT_LIST) -> pd.DataFrame:
    if path and os.path.exists(path):
        df = pd.read_csv(path, dtype=str).fillna("")
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def reconcile(pending: pd.DataFrame, shortlist: pd.DataFrame,
              incorporated: set[str], today: str) -> pd.DataFrame:
    """Return the updated pending list.

    * Rows whose DOI is in ``incorporated`` are dropped (they're in the DB now).
    * Shortlist papers not already pending and not incorporated are appended,
      stamped with ``today`` as ``first_flagged``.

    Papers are matched by ``paper_key`` (DOI, else PMID, else title), so a paper
    without a DOI is still listed. It cannot drop off automatically, though,
    because incorporation is detected by DOI; a curator removes it by hand.
    """
    kept = []
    seen: set[str] = set()

    def _str(v) -> str:
        return "" if pd.isna(v) else str(v)

    for _, r in pending.iterrows():
        key = paper_key(r)
        if not key or key in incorporated or key in seen:
            continue
        seen.add(key)
        kept.append({"doi": _str(r.get("doi", "")), "pmid": _str(r.get("pmid", "")),
                     "title": _str(r.get("title", "")),
                     "first_flagged": _str(r.get("first_flagged", "")) or today})

    if not shortlist.empty:
        for _, r in shortlist.iterrows():
            key = paper_key(r)
            if not key or key in incorporated or key in seen:
                continue
            seen.add(key)
            kept.append({"doi": _str(r.get("doi", "")), "pmid": _str(r.get("pmid", "")),
                         "title": _str(r.get("title", "")),
                         "first_flagged": today})

    return pd.DataFrame(kept, columns=COLUMNS)


def other_papers(found: pd.DataFrame, pending: pd.DataFrame,
                 incorporated: set[str]) -> pd.DataFrame:
    """Papers the search found that are not on the to-check list, best score first."""
    if found.empty:
        return found
    skip = {paper_key(r) for _, r in pending.iterrows()} | set(incorporated)
    rows, keys = [], set()
    for _, r in found.iterrows():
        key = paper_key(r)
        if not key or key in skip or key in keys:
            continue
        keys.add(key)
        rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty and "score" in out.columns:
        order = pd.to_numeric(out["score"], errors="coerce").fillna(0)
        out = out.loc[order.sort_values(ascending=False, kind="stable").index]
    return out


def _clean_title(title) -> str:
    t = html.unescape(html.unescape(str(title or "")))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t)).strip()


def _ident(r) -> str:
    doi = str(r.get("doi", "") or "").strip()
    pmid = str(r.get("pmid", "") or "").split(".")[0].strip()
    if doi and doi.lower() != "nan":
        return f"DOI: {doi}"
    if pmid and pmid.lower() != "nan":
        return f"PMID: {pmid}"
    return "no DOI or PMID"


def _flag(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def format_email_body(pending: pd.DataFrame, others: pd.DataFrame | None = None) -> str:
    """Plain-text email body: papers still to check, then other papers found.

    Returns '' only when both are empty.
    """
    others = others if others is not None else pd.DataFrame()
    if pending.empty and others.empty:
        return ""
    lines = []
    if not pending.empty:
        lines += [
            f"{len(pending)} crustacean-neuropeptide paper(s) are waiting to be "
            f"checked for new sequences.",
            "",
            "A paper drops off this list automatically once its DOI appears in the "
            "database.",
            "",
        ]
        for _, r in pending.iterrows():
            title = _clean_title(r.get("title", "")) or "(no title)"
            lines.append(f"- {title}")
            lines.append(f"    {_ident(r)}    (flagged {r.get('first_flagged', '')})")
        lines.append("")
    else:
        lines += ["No papers are waiting on the to-check list.", ""]
    if not others.empty:
        lines += [
            f"{len(others)} other paper(s) found by the search did not mention both a "
            f"crustacean term and a neuropeptide term in the title or abstract. "
            f"Listed by relevance score, highest first:",
            "",
        ]
        for _, r in others.iterrows():
            title = _clean_title(r.get("title", "")) or "(no title)"
            matched = [name for name in ("crustacean", "neuropeptide")
                       if _flag(r.get(name, ""))]
            note = f"matched: {', '.join(matched)}" if matched else "matched: neither"
            lines.append(f"- {title}")
            lines.append(f"    {_ident(r)}    ({note})")
        lines.append("")
    lines.append("Source: monthly cNPDB literature scan.")
    return "\n".join(lines)


def _read_all(paths, columns) -> pd.DataFrame:
    """Concatenate the CSVs that exist; empty frame with ``columns`` if none do."""
    frames = [pd.read_csv(p, dtype=str) for p in paths or [] if p and os.path.exists(p)]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _today() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reconcile the papers-to-check list.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reconcile", help="update the list and emit an email body")
    r.add_argument("--shortlist", nargs="*", default=[],
                   help="shortlist CSV(s); their papers are added to the to-check list")
    r.add_argument("--found", nargs="*", default=[],
                   help="full ranked search output(s); papers not on the to-check "
                        "list are named in the email below it")
    r.add_argument("--db", default=DEFAULT_DB)
    r.add_argument("--list", default=DEFAULT_LIST, dest="list_path")
    r.add_argument("--body-out", default=None,
                   help="write the email body here (empty file if nothing to send)")
    r.add_argument("--date", default=None, help="first_flagged stamp (default: today)")

    args = ap.parse_args(argv)
    today = args.date or _today()

    pending = load_list(args.list_path)
    shortlist = _read_all(args.shortlist, ["doi", "pmid", "title"])
    found = _read_all(args.found, ["doi", "pmid", "title"])
    incorporated = known_dois(load_database(args.db))

    before = len(pending)
    updated = reconcile(pending, shortlist, incorporated, today)
    updated.to_csv(args.list_path, index=False)

    others = other_papers(found, updated, incorporated)
    body = format_email_body(updated, others)
    if args.body_out:
        with open(args.body_out, "w", encoding="utf-8") as fh:
            fh.write(body)

    print(f"papers-to-check: {before} -> {len(updated)} "
          f"(shortlist rows: {len(shortlist)}; other papers in email: {len(others)}; "
          f"DOIs already in DB: {len(incorporated)})")
    return updated


if __name__ == "__main__":
    main()
