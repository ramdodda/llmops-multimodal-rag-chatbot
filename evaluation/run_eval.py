"""Evaluate the RAG chatbot against evaluation_dataset.json.

Measures, for every question:
  * Retrieval  - context recall: how much of the reference context appears
                 in the retrieved chunks (no LLM needed).
  * Generation - correctness vs. ground truth and faithfulness to the
                 retrieved context, scored by Gemini as a judge.
  * Guardrail  - for questions marked "expect_not_found", the system must
                 refuse instead of inventing an answer.

Exits with code 1 if any score is below its threshold, so it can run in CI.

Usage (from the project root, with the stack running):
    python evaluation/run_eval.py
    python evaluation/run_eval.py --no-judge          # retrieval + guardrail only, no LLM cost
    python evaluation/run_eval.py --document-id <id>  # evaluate one document
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

# When running outside Docker, point the Google SDK at the local key file.
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    key_file = PROJECT_ROOT / "vertex-credentials.json"
    if key_file.is_file():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_file)

NOT_FOUND_MARKERS = ("couldn't find", "could not find", "couldn\u2019t find")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "its", "of", "on", "or", "the", "to", "using",
    "with", "named", "uses", "use",
}

JUDGE_PROMPT = """
You are grading the answer of a document question-answering system.

Score two things from 1 to 5:

correctness: does the ANSWER convey the same facts as the GROUND TRUTH?
  5 = fully correct, 3 = partially correct or missing key facts, 1 = wrong.
  Extra correct detail is fine. Wording does not matter, meaning does.

faithfulness: is every claim in the ANSWER supported by the CONTEXT?
  5 = everything is supported, 3 = some unsupported claims, 1 = mostly invented.

Return ONLY a JSON object, with no markdown:
{{"correctness": <1-5>, "faithfulness": <1-5>, "reason": "<one short sentence>"}}

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

CONTEXT (retrieved chunks):
{context}

ANSWER:
{answer}
"""


# --------------------------------------------------
# Scoring helpers
# --------------------------------------------------
def tokens(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9_]+", text.lower())
        if word not in STOPWORDS
    }


def context_recall(reference: str, retrieved_texts: list[str]) -> float:
    """Fraction of the reference context's words found in retrieved chunks."""
    reference_tokens = tokens(reference)
    if not reference_tokens:
        return 0.0
    retrieved_tokens = tokens(" ".join(retrieved_texts))
    return len(reference_tokens & retrieved_tokens) / len(reference_tokens)


def is_not_found(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in NOT_FOUND_MARKERS)


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return json.loads(match.group(0) if match else cleaned)


# --------------------------------------------------
# LLM judge
# --------------------------------------------------
class GeminiJudge:
    def __init__(self, model: str):
        from google import genai
        from google.genai import types

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise SystemExit("GOOGLE_CLOUD_PROJECT is missing from .env")

        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        )
        self.model = model
        self.config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )
        self.tokens_used = 0

    def score(self, question, ground_truth, context, answer) -> dict:
        prompt = JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            context=context or "(no context retrieved)",
            answer=answer,
        )

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self.config,
                )
                usage = getattr(response, "usage_metadata", None)
                self.tokens_used += getattr(usage, "total_token_count", 0) or 0

                result = parse_json(response.text or "")
                return {
                    "correctness": int(result["correctness"]),
                    "faithfulness": int(result["faithfulness"]),
                    "reason": str(result.get("reason", "")),
                }
            except Exception as exc:  # noqa: BLE001 - judge must not crash the run
                if attempt == 2:
                    return {
                        "correctness": None,
                        "faithfulness": None,
                        "reason": f"judge failed: {exc}",
                    }
                time.sleep(2 ** attempt)


# --------------------------------------------------
# Runner
# --------------------------------------------------
def ask(api_url: str, question: str, document_id: str, top_k: int) -> tuple[dict, float]:
    start = time.perf_counter()
    response = requests.post(
        f"{api_url}/ask-file",
        json={"question": question, "top_k": top_k, "document_id": document_id},
        timeout=180,
    )
    latency = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), latency


def evaluate_item(item, args, judge) -> dict:
    data, latency = ask(args.api_url, item["question"], args.document_id, args.top_k)

    answer = data.get("answer") or ""
    chunks = data.get("retrieved_chunks") or []
    chunk_texts = [chunk.get("text", "") for chunk in chunks]
    usage = data.get("token_usage") or {}

    result = {
        "id": item.get("id", item["question"][:30]),
        "question": item["question"],
        "answer": answer,
        "latency_s": round(latency, 2),
        "tokens": usage.get("total_token_count", 0) or 0,
        "top_similarity": max((c.get("similarity", 0) for c in chunks), default=None),
        "prompt_version": data.get("prompt_version"),
    }

    if item.get("expect_not_found"):
        result["type"] = "negative"
        result["refused"] = is_not_found(answer)
        result["passed"] = result["refused"]
        return result

    result["type"] = "answerable"
    recall = context_recall(item.get("reference_context", ""), chunk_texts)
    result["context_recall"] = round(recall, 2)
    result["retrieval_hit"] = recall >= args.recall_threshold

    if judge:
        # Give the judge the same source filenames the answering model saw,
        # otherwise correct filename citations look "invented".
        judge_context = "\n\n".join(
            f"[Source: {chunk.get('filename', 'unknown')}]\n{chunk.get('text', '')}"
            for chunk in chunks
        )
        scores = judge.score(
            item["question"],
            item.get("ground_truth", ""),
            judge_context,
            answer,
        )
        result.update(scores)
        result["correct"] = (scores["correctness"] or 0) >= 4
        result["passed"] = result["retrieval_hit"] and result["correct"]
    else:
        result["passed"] = result["retrieval_hit"]

    return result


def rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def summarize(results: list[dict], use_judge: bool) -> dict:
    answerable = [r for r in results if r["type"] == "answerable"]
    negatives = [r for r in results if r["type"] == "negative"]
    latencies = sorted(r["latency_s"] for r in results)

    summary = {
        "questions": len(results),
        "retrieval_hit_rate": rate([r["retrieval_hit"] for r in answerable]),
        "mean_context_recall": (
            round(statistics.mean(r["context_recall"] for r in answerable), 2)
            if answerable else None
        ),
        "negative_rejection_rate": rate([r["refused"] for r in negatives]),
        "avg_latency_s": round(statistics.mean(latencies), 2) if latencies else None,
        "p95_latency_s": (
            latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            if latencies else None
        ),
        "total_app_tokens": sum(r["tokens"] for r in results),
    }

    if use_judge:
        judged = [r for r in answerable if r.get("correctness") is not None]
        summary["answer_correctness_rate"] = rate([r["correct"] for r in judged])
        summary["mean_faithfulness"] = (
            round(statistics.mean(r["faithfulness"] for r in judged), 2)
            if judged else None
        )
        summary["judge_failures"] = len(answerable) - len(judged)

    return summary


def print_report(results, summary, config):
    print("\n" + "=" * 92)
    print(
        f"RAG evaluation  |  prompt {config.get('rag_prompt_version')}  |  "
        f"min_similarity {config.get('min_similarity')}"
    )
    print("=" * 92)
    print(f"{'id':<17}{'type':<11}{'recall':>7}{'sim':>7}{'corr':>6}{'faith':>6}"
          f"{'tokens':>8}{'sec':>7}  result")
    print("-" * 92)

    for r in results:
        recall = f"{r['context_recall']:.2f}" if "context_recall" in r else "-"
        sim = f"{r['top_similarity']:.2f}" if r.get("top_similarity") else "-"
        corr = r.get("correctness") or "-"
        faith = r.get("faithfulness") or "-"
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id'][:16]:<17}{r['type']:<11}{recall:>7}{sim:>7}{corr!s:>6}"
              f"{faith!s:>6}{r['tokens']:>8}{r['latency_s']:>7}  {status}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\nFailures:")
        for r in failures:
            reason = r.get("reason") or (
                "did not refuse" if r["type"] == "negative" else "retrieval miss"
            )
            print(f"  - {r['id']}: {reason}")
            print(f"    answer: {r['answer'][:150]!r}")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key:<26}{value}")


def check_gates(summary, args) -> list[str]:
    gates = [
        ("retrieval_hit_rate", args.min_hit_rate),
        ("negative_rejection_rate", args.min_rejection),
        ("answer_correctness_rate", args.min_correctness),
        ("mean_faithfulness", args.min_faithfulness),
    ]
    problems = []
    for metric, threshold in gates:
        value = summary.get(metric)
        if value is not None and value < threshold:
            problems.append(f"{metric} = {value} (minimum {threshold})")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG chatbot.")
    parser.add_argument("--api-url", default=os.getenv("EVAL_API_URL", "http://localhost:8000"))
    parser.add_argument("--dataset", default=str(EVAL_DIR / "evaluation_dataset.json"))
    parser.add_argument("--document-id", default="__all__")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL") or os.getenv(
        "GOOGLE_GENAI_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--recall-threshold", type=float, default=0.6)
    parser.add_argument("--min-hit-rate", type=float, default=0.8)
    parser.add_argument("--min-correctness", type=float, default=0.8)
    parser.add_argument("--min-rejection", type=float, default=0.8)
    parser.add_argument("--min-faithfulness", type=float, default=4.0)
    args = parser.parse_args()

    args.api_url = args.api_url.rstrip("/")
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))

    try:
        config = requests.get(f"{args.api_url}/", timeout=10).json()
    except requests.RequestException as exc:
        raise SystemExit(f"Cannot reach the API at {args.api_url}: {exc}")

    judge = None if args.no_judge else GeminiJudge(args.judge_model)

    results = []
    for number, item in enumerate(dataset, start=1):
        print(f"[{number}/{len(dataset)}] {item['question']}")
        try:
            results.append(evaluate_item(item, args, judge))
        except requests.RequestException as exc:
            results.append({
                "id": item.get("id", str(number)),
                "question": item["question"],
                "type": "negative" if item.get("expect_not_found") else "answerable",
                "answer": f"API error: {exc}",
                "latency_s": 0, "tokens": 0, "passed": False,
                "retrieval_hit": False, "refused": False, "context_recall": 0.0,
            })

    summary = summarize(results, use_judge=judge is not None)
    if judge:
        summary["judge_tokens"] = judge.tokens_used

    print_report(results, summary, config)

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_file = out_dir / f"eval_{stamp}.json"
    out_file.write_text(json.dumps({
        "timestamp_utc": stamp,
        "api_config": config,
        "settings": {k: v for k, v in vars(args).items() if k != "dataset"},
        "summary": summary,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_file}")

    problems = check_gates(summary, args)
    if problems:
        print("\nQUALITY GATE FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    print("\nQUALITY GATE PASSED")


if __name__ == "__main__":
    main()