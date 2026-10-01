# -*- coding: utf-8 -*-
"""Literature-mining pipeline to keep cNPDB current.

Two complementary modes:

* ``discover`` -- find *recent* crustacean neuropeptide papers (by topic/species
  keywords over a date window) whose DOI is **not** already cited in the
  database. This is the mode meant to be scheduled (e.g. monthly) so a curator
  gets a short digest of papers to screen for new sequences.

* ``references`` -- the original behaviour: for every peptide already in the
  database, query PubMed + Europe PMC for the exact sequence to collect
  supporting references. Useful when back-filling citations.

Network calls are isolated in thin functions; the parsing/filtering helpers are
pure so they can be unit-tested without hitting the network. Set an NCBI API key
via the ``NCBI_API_KEY`` environment variable to raise the PubMed rate limit.

Examples::

    python -m DataCuration.lit_mining discover --since 2025-01-01 --out new_papers.csv
    python -m DataCuration.lit_mining references --db Assets/20260418_cNPDB.xlsx --out refs.csv
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataCuration.cnpdb_qc import DEFAULT_DB, load_database  # noqa: E402

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PAPER_COLUMNS = ["title", "authors", "journal", "year", "doi", "pmid", "abstract",
                 "source_query"]
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "lafields2@wisc.edu")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")

# Topic query used by ``discover``. Combines neuropeptide vocabulary with the
# crustacean clade so we cast wide but stay on-topic.
DISCOVER_QUERY = (
    '(neuropeptide OR neuropeptidome OR peptidome OR "peptide hormone" OR allatostatin '
    'OR RFamide OR orcokinin OR "crustacean cardioactive peptide" OR "crustacean hyperglycemic hormone") '
    'AND (crustacean OR decapod OR crab OR lobster OR crayfish OR shrimp OR Crustacea)'
)

# --- Crustacean-relevance vocabulary (cNPDB scope is Crustacea only) ---------
# General crustacean clade terms.
CRUSTACEAN_TERMS = {
    "crustacean", "crustacea", "decapod", "decapoda", "malacostraca", "brachyura",
    "crab", "lobster", "crayfish", "shrimp", "prawn", "isopod", "amphipod",
    "copepod", "barnacle", "krill", "cladocera", "stomatogastric",
}
# Scientific names + genera of the 30 cNPDB species (from the Glossary).
CRUSTACEAN_SPECIES = {
    "cancer borealis", "carcinus maenas", "callinectes sapidus", "homarus americanus",
    "litopenaeus vannamei", "penaeus vannamei", "procambarus clarkii", "orconectes limosus",
    "ocypode ceratophthalm", "scylla paramamosain", "scylla serrata", "nephrops norvegicus",
    "panulirus", "cancer irroratus", "cancer magister", "metacarcinus magister",
    "cancer pagurus", "cancer productus", "daphnia pulex", "armadillidium", "eurydice",
    "lithodes maja", "metapenaeus ensis", "marsupenaeus japonicus", "macrobrachium",
    "penaeus aztecus", "pandalus borealis", "procambarus bouvieri", "pugettia producta",
    "sagmariasus verreauxi", "penaeus monodon",
}
NEUROPEPTIDE_TERMS = {
    "neuropeptide", "neuropeptidom", "peptidom", "peptide hormone", "allatostatin",
    "rfamide", "ryamide", "orcokinin", "tachykinin", "pyrokinin", "sulfakinin",
    "crustacean cardioactive", "hyperglycemic hormone", "hyperglycaemic hormone",
    "pigment dispersing", "corazonin", "bursicon", "sifamide", "leucokinin",
    "proctolin", "myosuppressin", "eclosion hormone", "ecdysis", "natalisin", "ccap",
    " chh", " mih", " gih", " vih",
    "red pigment-concentrating", "red pigment concentrating", "rpch",
}
DISCOVERY_TERMS = {
    "novel", "new ", "de novo", "identif", "characteriz", "characteris", "discover",
    "profiling", "neuropeptidom", "peptidom", "mass spectrom", "transcriptom",
    "sequenc", "genome", "prohormone", "precursor",
}
# Non-crustacean model organisms; used only to DROP a paper when NO crustacean
# signal is present (a crustacean-vs-insect comparison still counts as relevant).
EXCLUDE_ORGANISMS = {
    "drosophila", "mouse", "murine", " rat ", " rats", "human", "zebrafish", "bombyx",
    "aphid", "mosquito", "honeybee", "honey bee", "tribolium", "caenorhabditis",
    "c. elegans", "arabidopsis", "nematode", "tick", "spider", "locust", "cockroach",
}


def _contains_any(text: str, terms) -> bool:
    return any(t in text for t in terms)


def classify_relevance(title: str, abstract: str = "") -> dict:
    """Score a paper for crustacean-neuropeptide relevance from its text.

    Returns a dict with booleans (``crustacean``, ``neuropeptide``, ``discovery``,
    ``excluded_organism``), an integer ``score``, and ``keep`` -- True when the
    paper is crustacean AND neuropeptide-related (cNPDB scope). Pure/offline.
    """
    text = f"{title} {abstract}".lower()
    crust = _contains_any(text, CRUSTACEAN_TERMS) or _contains_any(text, CRUSTACEAN_SPECIES)
    npep = _contains_any(text, NEUROPEPTIDE_TERMS)
    disc = _contains_any(text, DISCOVERY_TERMS)
    excl = _contains_any(text, EXCLUDE_ORGANISMS)
    score = (3 if crust else 0) + (2 if npep else 0) + (1 if disc else 0)
    if excl and not crust:
        score = 0
    return {"crustacean": crust, "neuropeptide": npep, "discovery": disc,
            "excluded_organism": excl, "score": score, "keep": bool(crust and npep)}


def rank_by_relevance(df: pd.DataFrame, title_col: str = "title",
                      abstract_col: str = "abstract") -> pd.DataFrame:
    """Add relevance columns and return the frame sorted best-first, keepers on top.

    Scores title + abstract when an ``abstract_col`` column is present, so a paper
    is not dropped just because its title omits a crustacean or neuropeptide term.
    """
    df = df.reset_index(drop=True)
    titles = df[title_col].fillna("").astype(str)
    abstracts = (df[abstract_col].fillna("").astype(str) if abstract_col in df.columns
                 else pd.Series([""] * len(df)))
    rel = pd.DataFrame([classify_relevance(t, a) for t, a in zip(titles, abstracts)],
                       columns=["crustacean", "neuropeptide", "discovery",
                                "excluded_organism", "score", "keep"])
    out = pd.concat([df, rel], axis=1)
    return out.sort_values(["keep", "score", "year"], ascending=[False, False, False])


def shortlist_from_ranked(ranked: pd.DataFrame) -> pd.DataFrame:
    """Papers in cNPDB scope: crustacean AND neuropeptide terms (title or abstract).

    The ``discovery`` flag only affects ranking; it is not required.
    """
    return ranked[ranked["keep"].astype(bool)]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def normalize_doi(doi: str) -> str:
    """Lower-case and strip a DOI to a comparable bare form."""
    if doi is None:
        return ""
    d = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def known_dois(df: pd.DataFrame) -> set[str]:
    """Set of normalized DOIs already present in the database's DOI column.

    Cells may contain multiple DOIs separated by ``;`` or whitespace.
    """
    out: set[str] = set()
    if "DOI" not in df.columns:
        return out
    for cell in df["DOI"].dropna().astype(str):
        for token in cell.replace(",", ";").split(";"):
            nd = normalize_doi(token)
            if nd and nd not in ("nan", "none"):
                out.add(nd)
    return out


def normalize_title(title: str) -> str:
    """Lower-case a title and collapse punctuation/markup to single spaces."""
    t = html.unescape(html.unescape(str(title or ""))).lower()
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def paper_key(hit: dict) -> str:
    """Stable identity for a paper: DOI, else ``pmid:<id>``, else ``title:<title>``.

    Papers without a DOI used to be keyed by an empty value, so they were never
    recorded as seen and came back every month.
    """
    nd = normalize_doi(hit.get("doi", ""))
    if nd and nd not in ("nan", "none"):
        return nd
    pmid = str(hit.get("pmid", "") or "").strip()
    if pmid and pmid.lower() not in ("nan", "none"):
        return f"pmid:{pmid.split('.')[0]}"
    nt = normalize_title(hit.get("title", ""))
    return f"title:{nt}" if nt and nt not in ("nan", "none") else ""


def load_seen_dois(path: str) -> set[str]:
    """Read a persisted seen-list (one DOI or paper key per line; ``#`` comments ignored)."""
    if not path or not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(normalize_doi(line))
    return {d for d in out if d}


def append_seen_dois(path: str, dois) -> None:
    """Append new paper keys (see ``paper_key``) to the seen-list, sorted & unique."""
    combined = load_seen_dois(path) | {normalize_doi(d) for d in dois if normalize_doi(d)}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# DOIs already surfaced by lit_mining discover; do not edit by hand.\n")
        for d in sorted(combined):
            fh.write(d + "\n")


def filter_new_papers(hits: list[dict], seen: set[str]) -> list[dict]:
    """Return hits whose DOI is not in ``seen`` (deduped within the batch too)."""
    new, batch = [], set()
    for h in hits:
        key = paper_key(h)
        if not key or key in seen or key in batch:
            continue
        batch.add(key)
        new.append(h)
    return new


# ---------------------------------------------------------------------------
# Network functions
# ---------------------------------------------------------------------------
def europepmc_search(query: str, page_size: int = 100, max_pages: int = 5,
                     session: requests.Session | None = None) -> list[dict]:
    """Query Europe PMC and return normalized hit dicts.

    Uses cursorMark pagination. Each hit -> {title, doi, pmid, journal, year, source_query}.
    """
    session = session or requests.Session()
    results, cursor = [], "*"
    for _ in range(max_pages):
        params = {"query": query, "format": "json", "pageSize": page_size,
                  "cursorMark": cursor, "resultType": "core"}
        resp = session.get(EPMC_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for r in data.get("resultList", {}).get("result", []):
            results.append({
                "title": r.get("title", ""),
                "doi": r.get("doi", ""),
                "pmid": r.get("pmid", ""),
                "journal": r.get("journalTitle", ""),
                "year": r.get("pubYear", ""),
                "authors": r.get("authorString", ""),
                "abstract": r.get("abstractText", ""),
                "source_query": query,
            })
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.3)
    return results


def discover_recent_papers(df: pd.DataFrame, since: str, until: str | None = None,
                           query: str = DISCOVER_QUERY,
                           session: requests.Session | None = None,
                           extra_seen: set[str] | None = None) -> pd.DataFrame:
    """Find recent on-topic papers not already cited in the database.

    ``since``/``until`` are ISO dates (YYYY-MM-DD). ``extra_seen`` is a set of
    already-surfaced DOIs (e.g. from a persisted seen-list) to exclude on top of
    the DOIs already in the database -- this is what makes a scheduled monthly
    run show only *newly appeared* papers. Returns a DataFrame of candidate
    papers for a curator to screen for new sequences.
    """
    date_clause = f'(FIRST_PDATE:[{since} TO {until or "3000-12-31"}])'
    full_query = f"{query} AND {date_clause}"
    hits = europepmc_search(full_query, session=session)
    seen = known_dois(df) | (extra_seen or set())
    new = filter_new_papers(hits, seen)
    return pd.DataFrame(new, columns=PAPER_COLUMNS).sort_values("year", ascending=False)


def fetch_abstract(row: dict, session: requests.Session | None = None) -> str:
    """Look up one paper's abstract on Europe PMC by DOI, else PMID, else title."""
    nd = normalize_doi(row.get("doi", ""))
    pmid = str(row.get("pmid", "") or "").split(".")[0].strip()
    if nd and nd not in ("nan", "none"):
        query = f'DOI:"{nd}"'
    elif pmid and pmid.lower() not in ("nan", "none"):
        query = f"EXT_ID:{pmid} AND SRC:MED"
    else:
        title = re.sub(r'["<>]', " ", str(row.get("title", "") or "")).strip()
        if not title:
            return ""
        query = f'TITLE:"{title}"'
    hits = europepmc_search(query, page_size=1, max_pages=1, session=session)
    return hits[0].get("abstract", "") if hits else ""


def rescreen_papers(frames: list[pd.DataFrame], session: requests.Session | None = None,
                    sleep: float = 0.3) -> pd.DataFrame:
    """Re-score papers from earlier ``discover`` outputs using their abstracts.

    Recovers papers that were found but dropped by the old title-only filter;
    they are already on the seen-list, so ``discover`` will not return them
    again. Duplicates across inputs are collapsed by ``paper_key``.
    """
    session = session or requests.Session()
    rows, keys = [], set()
    for f in frames:
        for r in f.fillna("").to_dict("records"):
            k = paper_key(r)
            if not k or k in keys:
                continue
            keys.add(k)
            rows.append(r)
    for r in rows:
        try:
            r["abstract"] = fetch_abstract(r, session=session)
        except Exception as exc:  # pragma: no cover - network defensive
            print(f"abstract lookup failed for {r.get('doi') or r.get('title')}: {exc}")
            r["abstract"] = ""
        time.sleep(sleep)
    df = pd.DataFrame(rows)
    for c in PAPER_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[PAPER_COLUMNS]


def write_ranked(ranked: pd.DataFrame, out: str, shortlist: str | None) -> pd.DataFrame:
    """Write the ranked paper list (abstracts omitted) and, optionally, the shortlist."""
    ranked.drop(columns=["abstract"], errors="ignore").to_csv(out, index=False)
    short = shortlist_from_ranked(ranked)
    if shortlist:
        short.drop(columns=["abstract"], errors="ignore").to_csv(shortlist, index=False)
        print(f"Crustacean + neuropeptide shortlist: {len(short)} of {len(ranked)} "
              f"-> {shortlist}")
    return short


def references_for_sequences(sequences, session: requests.Session | None = None,
                             sleep: float = 0.4) -> pd.DataFrame:
    """For each peptide sequence, collect Europe PMC references mentioning it.

    A lightweight, robust rewrite of the original per-peptide search. PubMed
    E-utilities can be layered in via Biopython if desired; Europe PMC alone
    covers PubMed + PMC + preprints and needs no API key.
    """
    session = session or requests.Session()
    records = []
    for seq in sequences:
        try:
            hits = europepmc_search(f'"{seq}"', page_size=50, max_pages=1, session=session)
            for h in hits:
                records.append({"Peptide": seq, "DOI": h["doi"], "PMID": h["pmid"],
                                "Title": h["title"], "Source": "Europe PMC"})
        except Exception as exc:  # pragma: no cover - network defensive
            records.append({"Peptide": seq, "DOI": "", "PMID": "",
                            "Title": f"ERROR: {exc}", "Source": "error"})
        time.sleep(sleep)
    return pd.DataFrame(records, columns=["Peptide", "DOI", "PMID", "Title", "Source"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="cNPDB literature mining.")
    sub = ap.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("discover", help="find recent papers not yet cited in the DB")
    d.add_argument("--since", required=True, help="ISO date, e.g. 2025-01-01")
    d.add_argument("--until", default=None, help="ISO date (default: open-ended)")
    d.add_argument("--db", default=DEFAULT_DB)
    d.add_argument("--out", default="new_papers.csv")
    d.add_argument("--seen", default=None,
                   help="path to a persisted seen-DOI list; excludes already-surfaced "
                        "papers so each run shows only what is new")
    d.add_argument("--update-seen", action="store_true",
                   help="append this run's DOIs to the --seen file")
    d.add_argument("--shortlist", default=None,
                   help="also write the crustacean + neuropeptide shortlist here")

    s = sub.add_parser("rescreen",
                       help="re-score papers from earlier discover outputs using abstracts")
    s.add_argument("--inputs", nargs="+", required=True,
                   help="earlier new_papers_<date>.csv files")
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--out", required=True, help="ranked list of every re-screened paper")
    s.add_argument("--shortlist", default=None, help="crustacean + neuropeptide subset")

    r = sub.add_parser("references", help="collect references for existing peptides")
    r.add_argument("--db", default=DEFAULT_DB)
    r.add_argument("--out", default="peptide_references.csv")
    r.add_argument("--limit", type=int, default=None, help="only first N peptides (testing)")

    args = ap.parse_args(argv)
    df = load_database(args.db)

    if args.mode == "discover":
        extra_seen = load_seen_dois(args.seen) if args.seen else set()
        out = discover_recent_papers(df, since=args.since, until=args.until, extra_seen=extra_seen)
        print(f"Found {len(out)} candidate new papers since {args.since} "
              f"(excluding {len(extra_seen)} already-seen) -> {args.out}")
        write_ranked(rank_by_relevance(out), args.out, args.shortlist)
        if args.seen and args.update_seen:
            append_seen_dois(args.seen, [paper_key(r) for r in out.to_dict("records")])
            print(f"Updated seen-list -> {args.seen}")
    elif args.mode == "rescreen":
        frames = [pd.read_csv(p, dtype=str) for p in args.inputs]
        out = rescreen_papers(frames)
        print(f"Re-screened {len(out)} papers from {len(args.inputs)} file(s) -> {args.out}")
        write_ranked(rank_by_relevance(out), args.out, args.shortlist)
    else:
        seqs = df["Sequence"].dropna().astype(str).tolist()
        if args.limit:
            seqs = seqs[:args.limit]
        out = references_for_sequences(seqs)
        out.to_csv(args.out, index=False)
        print(f"Collected {len(out)} reference rows for {len(seqs)} peptides -> {args.out}")
    return out


if __name__ == "__main__":
    main()
