import seed


def test_s2_to_raw():
    rec = {"corpusId": 1001, "title": "BadNets", "abstract": "abs", "year": 2019,
           "venue": "IEEE", "citationCount": 980, "tldr": {"text": "tl"},
           "externalIds": {"DOI": "10.1/a", "ArXiv": "1708.06733"},
           "fieldsOfStudy": ["Computer Science"],
           "embedding": {"vector": [0.0] * 768},
           "references": [{"corpusId": 1002}, {"corpusId": None}]}
    r = seed._s2_to_raw(rec)
    assert r["paper_id"] == "S2:1001" and r["tldr"] == "tl"
    assert r["refs"] == ["S2:1002"]                      # None corpusId dropped
    assert len(r["embedding"]) == 768
    assert {"scheme": "doi", "value": "10.1/a"} in r["ext_ids"]
    assert {"scheme": "s2", "value": "1001"} in r["ext_ids"]


def test_s2_to_raw_skips_incomplete():
    assert seed._s2_to_raw({"title": "x"}) is None        # no corpusId
    assert seed._s2_to_raw({"corpusId": 1}) is None       # no title
    assert seed._s2_to_raw(None) is None


def test_oa_to_raw():
    w = {"id": "https://openalex.org/W42", "title": "Trojaning",
         "abstract_inverted_index": {"Implant": [0], "triggers": [1]},
         "publication_year": 2021, "cited_by_count": 410,
         "primary_location": {"source": {"display_name": "NDSS"}},
         "ids": {"doi": "https://doi.org/10.2/y", "mag": 55},
         "concepts": [{"display_name": "Backdoor"}],
         "referenced_works": ["https://openalex.org/W7"]}
    r = seed._oa_to_raw(w)
    assert r["paper_id"] == "OA:W42" and r["abstract"] == "Implant triggers"
    assert r["venue"] == "NDSS" and r["refs"] == ["OA:W7"]
    assert {"scheme": "doi", "value": "10.2/y"} in r["ext_ids"]        # doi.org stripped


def test_reconstruct_abstract():
    assert seed._reconstruct_abstract(None) is None
    assert seed._reconstruct_abstract({"a": [0], "b": [1], "c": [2]}) == "a b c"
    assert seed._reconstruct_abstract({"world": [1], "hello": [0]}) == "hello world"
