"""Answer-prompt comparison on synthetic verdicts."""

from evals.failures.compare_answers import compare, large_result_ids


def test_compare_paired_by_pass_and_skips_no_reference() -> None:
    v1 = [{1: "correct", 2: "incorrect", 3: "no_reference"}, {1: "correct", 2: "incorrect"}]
    v2 = [{1: "correct", 2: "correct", 3: "correct"}, {1: "correct", 2: "incorrect"}]
    out = compare(v1, v2, {"all": {1, 2, 3}, "two": {2}, "none": {9}})
    assert set(out) == {"all", "two"}
    assert out["all"]["n_questions"] == 2
    assert out["all"]["accepted_per_pass"] == {"v1": [1, 1], "v2": [2, 1]}
    assert out["two"]["v2_minus_v1"]["diff"] == 0.5


def test_large_result_ids() -> None:
    def pred(qid: int, row_count: str) -> dict:
        return {
            "question_id": qid,
            "pred_sql": "S",
            "attempts": [{"sql": "S", "row_count": row_count}],
        }

    preds = [pred(1, "20"), pred(2, "21"), pred(3, ">1000"), pred(4, "n/a")]
    assert large_result_ids(preds) == {2, 3}
