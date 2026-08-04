import json
import requests
from collections import defaultdict

BASE_URL = "http://localhost:8000"

QUESTION_FILE = "benchmark_output/benchmark_questions.json"


def query(q):

    r = requests.post(
        f"{BASE_URL}/memory/query",
        json={"text": q},
        timeout=120
    )

    if not r.ok:

        return []

    data = r.json()

    return data.get("results", [])


def run_report():

    with open(QUESTION_FILE, "r") as f:

        questions = json.load(f)

    total = len(questions)

    top1 = 0

    top3 = 0

    failed = 0

    type_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})

    failure_examples = []

    for i, q in enumerate(questions):

        query_text = q["query"]

        expected = q.get("expected")

        if not isinstance(expected, str):

            continue

        expected = expected.lower()

        results = query(query_text)

        if not results:

            failed += 1

            continue

        top_texts = [r["text"].lower() for r in results[:3]]

        # -------------------------
        # GLOBAL STATS
        # -------------------------

        if expected in top_texts[0]:

            top1 += 1

        if any(expected in t for t in top_texts):

            top3 += 1

        # -------------------------
        # TYPE STATS (IMPORTANT)
        # -------------------------

        memory_type = None

        if results and isinstance(results[0], dict):

            memory_type = results[0].get("type", "unknown")

        type_stats[memory_type]["total"] += 1

        if expected in top_texts[0]:

            type_stats[memory_type]["top1"] += 1

        if any(expected in t for t in top_texts):

            type_stats[memory_type]["top3"] += 1

        # -------------------------
        # FAILURE LOGGING
        # -------------------------

        if expected not in top_texts[0]:

            failure_examples.append({

                "query": query_text,

                "expected": expected,

                "top": top_texts[:3]

            })

        if i % 500 == 0:

            print(f"Processed {i}/{total}")

    # -------------------------
    # REPORT OUTPUT
    # -------------------------

    print("\n================ REGRESSION REPORT ================\n")

    print(f"Total: {total}")

    print(f"Top-1: {top1/total:.3f}")

    print(f"Top-3: {top3/total:.3f}")

    print(f"Failures: {failed}")

    print("\n--- TYPE BREAKDOWN ---\n")

    for t, stats in type_stats.items():

        if stats["total"] == 0:

            continue

        print(

            f"{t}: "

            f"Top1={stats['top1']/stats['total']:.2f} "

            f"Top3={stats['top3']/stats['total']:.2f} "

            f"n={stats['total']}"

        )

    print("\n--- SAMPLE FAILURES ---\n")

    for f in failure_examples[:10]:

        print(f["query"])

        print("EXPECTED:", f["expected"])

        print("TOP:", f["top"])

        print("-----")

    print("\n====================================================\n")


if __name__ == "__main__":

    run_report()
