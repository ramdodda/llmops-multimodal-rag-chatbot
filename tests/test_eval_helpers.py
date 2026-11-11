from run_eval import context_recall, is_not_found, parse_json


def test_context_recall_full_and_partial():
    reference = "Vector dimensions: 768"
    assert context_recall(reference, ["... Vector dimensions: 768 ..."]) == 1.0
    assert context_recall(reference, ["vector size unknown"]) < 0.5


def test_is_not_found_handles_apostrophes():
    assert is_not_found("I couldn't find that information.")
    assert is_not_found("I couldn\u2019t find that.")
    assert not is_not_found("It uses Qdrant.")


def test_parse_json_strips_markdown_fences():
    text = '```json\n{"correctness": 5, "faithfulness": 4, "reason": "ok"}\n```'
    assert parse_json(text)["faithfulness"] == 4
