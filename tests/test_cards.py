import cards


def test_header_and_fields_flag():
    rows = [
        {"paper_id": "S2:14", "year": 2019, "n_citations": 980, "cluster_id": 3,
         "fields_status": "extracted", "title": "BadNets", "tldr": "t", "tags": "backdoor,dnn"},
        {"paper_id": "S2:22", "year": 2021, "n_citations": 410, "cluster_id": None,
         "fields_status": "none", "title": "Trojan", "tldr": "u", "tags": "trojan"},
    ]
    lines = cards.render_cards(rows).splitlines()
    assert lines[0] == "cards[2]{paper_id,year,cit,cl,f,title,tldr,tags}:"
    assert lines[1].split("\t")[4] == "Y"    # extracted -> Y
    assert lines[2].split("\t")[4] == "N"    # none -> N
    assert lines[2].split("\t")[3] == ""     # None cluster -> empty cell


def test_tab_safe_and_score_column():
    rows = [{"paper_id": "p", "year": None, "n_citations": 0, "cluster_id": None,
             "fields_status": "none", "title": "a\tb\nc", "tldr": None, "tags": None,
             "score": 0.5}]
    lines = cards.render_cards(rows, extra=["score"]).splitlines()
    assert lines[0].endswith("score}:")
    cells = lines[1].split("\t")
    assert len(cells) == 9                    # 8 card cols + score, no extra tabs from title
    assert cells[5] == "a b c"                # tab/newline flattened to spaces
    assert cells[-1] == "0.5000"              # float formatted
