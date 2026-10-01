"""Unit tests for DataCuration.lit_mining pure helpers (no network)."""
import pandas as pd

from DataCuration import lit_mining as lm


def test_normalize_doi_variants():
    assert lm.normalize_doi("https://doi.org/10.1021/ABC") == "10.1021/abc"
    assert lm.normalize_doi("doi:10.1/X") == "10.1/x"
    assert lm.normalize_doi("  10.1/Y  ") == "10.1/y"
    assert lm.normalize_doi(None) == ""


def test_known_dois_splits_multi_value_cells():
    df = pd.DataFrame({"DOI": ["10.1/x; 10.2/y", "10.3/z", None, "nan"]})
    assert lm.known_dois(df) == {"10.1/x", "10.2/y", "10.3/z"}


def test_known_dois_missing_column():
    assert lm.known_dois(pd.DataFrame({"other": [1]})) == set()


def test_filter_new_papers_excludes_seen_and_dedupes_batch():
    hits = [
        {"doi": "10.1/x", "pmid": "1"},   # seen
        {"doi": "10.1/X", "pmid": "2"},   # same DOI (case) -> dropped as seen
        {"doi": "10.2/new", "pmid": "3"},  # new
        {"doi": "10.2/new", "pmid": "4"},  # dup of the new one within batch
        {"doi": "", "pmid": "99"},         # no doi -> keyed by pmid, kept
    ]
    new = lm.filter_new_papers(hits, seen={"10.1/x"})
    pmids = [h["pmid"] for h in new]
    assert pmids == ["3", "99"]


def test_filter_new_papers_all_seen():
    hits = [{"doi": "10.1/x", "pmid": "1"}]
    assert lm.filter_new_papers(hits, {"10.1/x"}) == []


def test_seen_list_roundtrip(tmp_path):
    path = str(tmp_path / "seen.txt")
    assert lm.load_seen_dois(path) == set()          # missing file -> empty
    lm.append_seen_dois(path, ["10.1/A", "https://doi.org/10.2/B"])
    assert lm.load_seen_dois(path) == {"10.1/a", "10.2/b"}
    # Appending is idempotent + additive, and normalizes.
    lm.append_seen_dois(path, ["10.2/B", "10.3/c"])
    assert lm.load_seen_dois(path) == {"10.1/a", "10.2/b", "10.3/c"}


def test_seen_list_ignores_comments_and_blanks(tmp_path):
    path = tmp_path / "seen.txt"
    path.write_text("# header\n10.1/x\n\n  10.2/y  \n", encoding="utf-8")
    assert lm.load_seen_dois(str(path)) == {"10.1/x", "10.2/y"}


# --- crustacean relevance (cNPDB scope is Crustacea only) -------------------
def test_relevance_keeps_crustacean_neuropeptide_paper():
    r = lm.classify_relevance("Novel neuropeptides identified in Cancer borealis")
    assert r["crustacean"] and r["neuropeptide"] and r["discovery"]
    assert r["keep"] and r["score"] == 6


def test_relevance_drops_non_crustacean_neuropeptide_paper():
    r = lm.classify_relevance("Neuropeptide signaling in Drosophila melanogaster")
    assert not r["crustacean"]
    assert r["excluded_organism"]
    assert not r["keep"] and r["score"] == 0


def test_relevance_drops_crustacean_paper_without_neuropeptides():
    r = lm.classify_relevance("Dietary calcium effects on growth in shrimp")
    assert r["crustacean"] and not r["neuropeptide"]
    assert not r["keep"]


def test_relevance_keeps_crustacean_vs_insect_comparison():
    # An excluded organism must NOT veto a paper that is also crustacean.
    r = lm.classify_relevance("Comparing neuropeptides of Drosophila and the crab")
    assert r["crustacean"] and r["excluded_organism"]
    assert r["keep"] and r["score"] > 0


def test_relevance_detects_species_by_scientific_name():
    assert lm.classify_relevance("Peptidome of Homarus americanus")["crustacean"]
    assert lm.classify_relevance("Scylla paramamosain peptide hormone")["crustacean"]


def test_rank_by_relevance_sorts_keepers_first():
    df = pd.DataFrame({
        "title": ["Dietary protein in tilapia",                       # drop
                  "Novel neuropeptides in the blue crab",             # keep, high
                  "Neuropeptide receptors in mice"],                  # drop
        "year": [2025, 2026, 2025],
    })
    ranked = lm.rank_by_relevance(df)
    assert ranked.iloc[0]["title"] == "Novel neuropeptides in the blue crab"
    assert bool(ranked.iloc[0]["keep"]) is True
    assert int(ranked["keep"].sum()) == 1


def test_rank_uses_abstract_when_title_lacks_terms():
    df = pd.DataFrame({
        "title": ["Feeding behaviour study"],
        "abstract": ["We profiled neuropeptides in the Jonah crab."],
        "year": [2026],
    })
    ranked = lm.rank_by_relevance(df)
    assert bool(ranked.iloc[0]["keep"]) is True


def test_shortlist_does_not_require_a_discovery_term():
    df = pd.DataFrame({"title": ["Neuropeptide F alters feeding genes in a prawn"],
                       "year": [2026]})
    ranked = lm.rank_by_relevance(df)
    assert not bool(ranked.iloc[0]["discovery"])
    assert len(lm.shortlist_from_ranked(ranked)) == 1


def test_rpch_counts_as_neuropeptide():
    r = lm.classify_relevance("Red pigment-concentrating hormone knockdown in Litopenaeus vannamei")
    assert r["neuropeptide"] and r["keep"]


def test_paper_key_falls_back_to_pmid_then_title():
    assert lm.paper_key({"doi": "10.1/X", "pmid": "5"}) == "10.1/x"
    assert lm.paper_key({"doi": "", "pmid": "5.0"}) == "pmid:5"
    assert lm.paper_key({"doi": "nan", "pmid": "nan",
                         "title": "Crab &lt;i&gt;Hemolymph&lt;/i&gt;!"}) == "title:crab hemolymph"
    assert lm.paper_key({"doi": "", "pmid": "", "title": ""}) == ""


def test_paper_without_doi_or_pmid_is_seen_by_title():
    hit = {"doi": "", "pmid": "", "title": "Approach for hemolymph collection"}
    key = lm.paper_key(hit)
    assert lm.filter_new_papers([hit], seen=set()) == [hit]
    assert lm.filter_new_papers([hit], seen={key}) == []


def test_seen_list_stores_non_doi_keys(tmp_path):
    path = str(tmp_path / "seen.txt")
    lm.append_seen_dois(path, ["10.1/a", "pmid:7", "title:crab hemolymph"])
    assert lm.load_seen_dois(path) == {"10.1/a", "pmid:7", "title:crab hemolymph"}
