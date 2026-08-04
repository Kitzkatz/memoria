import json
import requests
import time


BASE_URL = "http://localhost:8000"
QUESTION_FILE = "benchmark_output/benchmark_questions.json"


# -----------------------------------------
# DEBUG SETTINGS
# -----------------------------------------

VERBOSE = False

SHOW_FAILURES_ONLY = True

TOP_K_DISPLAY = 5


# -----------------------------------------
# API CALL
# -----------------------------------------

def query_memory(question):

    try:

        r = requests.post(
            f"{BASE_URL}/memory/query",
            json={"text": question},
            timeout=120
        )

    except Exception as e:

        print("[REQUEST ERROR]", e)

        return []


    if not r.ok:

        print("[API ERROR]", r.text)

        return []


    response = r.json()


    if "results" not in response:

        print("[BAD RESPONSE]", response)

        return []


    return response["results"]



# -----------------------------------------
# NORMALIZE
# -----------------------------------------

def normalize_text(result):

    if not isinstance(result, dict):

        return str(result).lower()


    return (

        result.get("normalized_text")

        or result.get("text")

        or ""

    ).lower()



# -----------------------------------------
# FIND EXPECTED MEMORY
# -----------------------------------------

def find_expected_rank(results, expected):

    expected = str(expected).lower()

    for result in results:

        text = normalize_text(result)

        if expected in text:

            return result.get("rank")


    return None



# -----------------------------------------
# PRINT DIAGNOSTICS
# -----------------------------------------

def print_result_debug(
        query,
        expected,
        results,
        expected_rank
):

    print("\n")
    print("=" * 70)

    print("QUERY:")
    print(query)

    print()

    print("EXPECTED:")
    print(expected)

    print()

    if expected_rank:

        print(
            "FOUND AT RANK:",
            expected_rank
        )

    else:

        print(
            "FOUND:",
            "NO"
        )


    print()

    print(
        "RETURNED:",
        len(results)
    )

    print("-" * 70)


    for result in results[:TOP_K_DISPLAY]:

        ranking = result.get(
            "ranking",
            {}
        )


        print()

        print(
            "RANK:",
            result.get("rank")
        )

        print(
            "ID:",
            result.get("id")
        )

        print(
            "TYPE:",
            result.get("memory_type")
        )

        print(
            "SCORE:",
            round(
                float(result.get("score",0)),
                5
            )
        )


        print()

        print("COMPONENTS")

        print(
            " semantic   :",
            ranking.get(
                "semantic",
                "N/A"
            )
        )

        print(
            " importance :",
            ranking.get(
                "importance",
                "N/A"
            )
        )

        print(
            " recency    :",
            ranking.get(
                "recency",
                "N/A"
            )
        )

        print(
            " token      :",
            ranking.get(
                "token",
                "N/A"
            )
        )

        print(
            " feedback   :",
            ranking.get(
                "feedback",
                "N/A"
            )
        )

        print(
            " final      :",
            ranking.get(
                "final",
                "N/A"
            )
        )


        print()

        print("TEXT:")

        print(
            result.get(
                "text",
                ""
            )
        )

        print("-" * 40)



    print("=" * 70)



# -----------------------------------------
# BENCHMARK
# -----------------------------------------

def evaluate():

    print()
    print("=" * 70)
    print("[BENCHMARK START]")
    print("=" * 70)


    with open(
        QUESTION_FILE,
        "r"
    ) as f:

        questions = json.load(f)



    total = len(questions)

    print(
        "[QUESTIONS]",
        total
    )


    top1 = 0
    top3 = 0

    failed = 0

    retrieved_correct = 0

    total_results = 0

    start = time.time()

    next_report = 5



    for i, item in enumerate(
        questions,
        start=1
    ):


        query = item.get("query")

        expected = item.get("expected")


        if not query or not expected:

            failed += 1

            continue



        results = query_memory(query)



        if not results:

            failed += 1

            continue



        total_results += len(results)



        expected_rank = find_expected_rank(
            results,
            expected
        )



        if expected_rank:

            retrieved_correct += 1



        if expected_rank == 1:

            top1 += 1



        if expected_rank and expected_rank <= 3:

            top3 += 1



        is_failure = (

            expected_rank is None

            or expected_rank > 3

        )



        if VERBOSE:

            print_result_debug(
                query,
                expected,
                results,
                expected_rank
            )


        elif SHOW_FAILURES_ONLY and is_failure:

            print_result_debug(
                query,
                expected,
                results,
                expected_rank
            )



        percent = int(
            (i / total) * 100
        )


        if percent >= next_report:

            elapsed = time.time() - start

            rate = i / max(
                elapsed,
                0.001
            )

            eta = (
                total-i
            ) / max(
                rate,
                0.001
            )


            print(
                f"[PROGRESS] {percent}% "
                f"{i}/{total} "
                f"ETA {eta:.1f}s"
            )


            next_report += 5




    elapsed = time.time() - start



    print()
    print("=" * 70)
    print("[BENCHMARK COMPLETE]")
    print("=" * 70)


    print(
        "Questions:",
        total
    )

    print(
        "Failed:",
        failed
    )

    print()


    print(
        "Retrieved Correct:",
        retrieved_correct
    )


    print(
        "Top 1:",
        top1,
        f"({top1/max(total,1)*100:.2f}%)"
    )


    print(
        "Top 3:",
        top3,
        f"({top3/max(total,1)*100:.2f}%)"
    )


    print()

    print(
        "Average Candidates Returned:",
        total_results/max(total,1)
    )


    print(
        "Runtime:",
        round(
            elapsed,
            2
        ),
        "seconds"
    )


    print("=" * 70)




if __name__ == "__main__":

    evaluate()

##import json
##import requests
##import time
##
##
##BASE_URL = "http://localhost:8000"
##QUESTION_FILE = "benchmark_output/benchmark_questions.json"
##
##
### -----------------------------------------
### API CALL
### -----------------------------------------
##
##def query_memory(question: str):
##
##    try:
##        r = requests.post(
##            f"{BASE_URL}/memory/query",
##            json={"text": question},
##            timeout=120
##        )
##
##    except Exception as e:
##        print("[REQUEST ERROR]", e)
##        return []
##
##    if not r.ok:
##        print("[API ERROR]", r.text)
##        return []
##
##    response = r.json()
##
##    if "results" not in response:
##        print("[BAD RESPONSE SHAPE]", response)
##        return []
##
##    return response["results"]
##
##
### -----------------------------------------
### RESULT NORMALIZATION
### -----------------------------------------
##
##def normalize_result_text(r):
##
##    if isinstance(r, dict):
##
##        return (
##            r.get("normalized_text")
##            or r.get("text")
##            or ""
##        ).lower()
##
##    return str(r).lower()
##
##
### -----------------------------------------
### BENCHMARK
### -----------------------------------------
##
##def evaluate():
##
##    print("\n==============================")
##    print("[BENCHMARK] STARTING")
##    print("==============================")
##
##    print("[LOAD] Reading questions...")
##
##    with open(QUESTION_FILE, "r") as f:
##        questions = json.load(f)
##
##
##    total = len(questions)
##
##    print(f"[LOAD] Questions loaded: {total}")
##
##    if total == 0:
##        print("[ERROR] Empty benchmark")
##        return
##
##
##    print("[CHECK] Memory API:")
##    
##    try:
##
##        requests.post(
##            f"{BASE_URL}/memory/query",
##            json={"text": "startup check"},
##            timeout=10
##        )
##
##        print("[CHECK] API reachable")
##
##    except Exception as e:
##
##        print("[CHECK FAILED]", e)
##
##
##
##    top1 = 0
##    top3 = 0
##    failed = 0
##
##
##    # -----------------------------------------
##    # 5% progress tracker
##    # -----------------------------------------
##
##    next_report = 5
##
##    start = time.time()
##
##
##    print("\n[BENCHMARK] RUNNING\n")
##
##
##    for i, q in enumerate(questions, start=1):
##
##
##        query = q.get("query")
##        expected = q.get("expected")
##
##
##        if not query or not expected:
##
##            failed += 1
##            continue
##
##
##        payload = query_memory(query)
##
##        results = payload["results"]
##
##        retrieved = payload["retrieved"]
##
##        returned = payload["returned"]
##
##
##        if not results:
##
##            failed += 1
##            continue
##
##
##
##        texts = [
##
##            normalize_result_text(r)
##
##            for r in results[:3]
##
##        ]
##
##        for result in results:
##
##            print()
##
##            print("=" * 60)
##
##            print("Rank :", result["rank"])
##
##            print("ID   :", result["id"])
##
##            print("Score:", round(result["score"], 4))
##
##            print()
##
##            ranking = result.get("ranking", {})
##
##            print("Semantic  :", ranking.get("semantic"))
##
##            print("Importance:", ranking.get("importance"))
##
##            print("Recency   :", ranking.get("recency"))
##
##            print("Token     :", ranking.get("token"))
##
##            print("Feedback  :", ranking.get("feedback"))
##
##            print("Final     :", ranking.get("final"))
##
##            print()
##
##            print(result["text"])
##        expected = str(expected).lower()
##
##
##        # top 1
##
##        if texts and expected in texts[0]:
##
##            top1 += 1
##
##
##
##        # top 3
##
##        if any(expected in t for t in texts):
##
##            top3 += 1
##
##
##
##        # -----------------------------------------
##        # Progress every 5%
##        # -----------------------------------------
##
##        percent = int((i / total) * 100)
##
##
##        if percent >= next_report:
##
##
##            elapsed = time.time() - start
##
##            rate = i / max(elapsed, 0.001)
##
##            remaining = (total - i) / max(rate, 0.001)
##
##
##            print(
##                f"[PROGRESS] {percent}% "
##                f"({i}/{total}) "
##                f"| "
##                f"ETA {remaining:.1f}s"
##            )
##
##
##            next_report += 5
##
##
##
##    # -----------------------------------------
##    # RESULTS
##    # -----------------------------------------
##
##    elapsed = time.time() - start
##
##
##    print("\n==============================")
##    print("[BENCHMARK COMPLETE]")
##    print("==============================")
##
##    print(f"Questions : {total}")
##    print(f"Failed    : {failed}")
##
##    print()
##
##    print(f"Top1      : {top1}")
##    print(f"Top3      : {top3}")
##
##    print()
##
##    print(
##        f"Top1 %    : {(top1/max(total,1))*100:.2f}"
##    )
##
##    print(
##        f"Top3 %    : {(top3/max(total,1))*100:.2f}"
##    )
##
##    print()
##
##    print(
##        f"Runtime   : {elapsed:.2f}s"
##    )
##
##    print("==============================")
##
##
##
##if __name__ == "__main__":
##
##    evaluate()
##
##
####import json
####import requests
####
####
####BASE_URL = "http://localhost:8000"
####QUESTION_FILE = "benchmark_output/benchmark_questions.json"
####
####
##### -----------------------------------------
##### API CALL
##### -----------------------------------------
####
####def query_memory(question: str):
####
####    try:
####        r = requests.post(
####            f"{BASE_URL}/memory/query",
####            json={"text": question},
####            timeout=120
####        )
####    except Exception as e:
####        print("[REQUEST ERROR]", e)
####        return []
####
####    if not r.ok:
####        print("[API ERROR]", r.text)
####        return []
####
####    response = r.json()
####
####    # strict contract
####    if "results" not in response:
####        print("[BAD RESPONSE SHAPE]", response)
####        return []
####
####    return response["results"]
####
####
##### -----------------------------------------
##### NORMALIZATION (NO TEXT CLEANING)
##### -----------------------------------------
####
####def normalize_result_text(r):
####
####    # API contract: memory system already returns normalized + structured text
####    if isinstance(r, dict):
####
####        # prefer normalized_text if present (your system contract)
####        return (
####            r.get("normalized_text")
####            or r.get("text")
####            or ""
####        ).lower()
####
####    return str(r).lower()
####
####
##### -----------------------------------------
##### EVALUATION CORE
##### -----------------------------------------
####
####def evaluate():
####
####    with open(QUESTION_FILE, "r") as f:
####        questions = json.load(f)
####
####    total = len(questions)
####
####    if total == 0:
####        print("[EMPTY DATASET]")
####        return
####
####    top1 = 0
####    top3 = 0
####    failed = 0
####
####    for i, q in enumerate(questions, start=1):
####
####        query = q.get("query")
####        expected = q.get("expected")
####
####        if not query or not expected:
####            failed += 1
####            continue
####
####        results = query_memory(query)
####
####        if not results:
####            failed += 1
####            continue
####
####        texts = [normalize_result_text(r) for r in results[:3]]
####
####        expected = str(expected).lower()
####
####        # -----------------------------
####        # TOP 1
####        # -----------------------------
####        if texts and expected in texts[0]:
####            top1 += 1
####
####        # -----------------------------
####        # TOP 3
####        # -----------------------------
####        if any(expected in t for t in texts):
####            top3 += 1
####
####        # progress log
####        if i % 250 == 0:
####            print(f"[PROGRESS] {i}/{total}")
####
####    # -----------------------------------------
####    # FINAL REPORT
####    # -----------------------------------------
####
####    print("\n" + "=" * 50)
####    print(f"Questions : {total}")
####    print(f"Failed    : {failed}")
####    print()
####    print(f"Top1      : {top1}")
####    print(f"Top3      : {top3}")
####    print()
####
####    denom = max(total, 1)
####
####    print(f"Top1 %    : {(top1 / denom) * 100:.2f}")
####    print(f"Top3 %    : {(top3 / denom) * 100:.2f}")
####    print("=" * 50)
####
####
##### -----------------------------------------
##### ENTRYPOINT
##### -----------------------------------------
####
####if __name__ == "__main__":
####    evaluate()
####
######def evaluate():
######
######    with open(QUESTION_FILE, "r") as f:
######        questions = json.load(f)
######
######    total = len(questions)
######
######    top1 = 0
######    top3 = 0
######    failed = 0
######
######    for index, q in enumerate(questions, start=1):
######
######        query = q["query"]
######        expected = q.get("expected")
######
######        if not isinstance(expected, str):
######            failed += 1
######            continue
######
######        results = query_memory(query)
######
######        if not results:
######            failed += 1
######            continue
######
######        texts = []
######
######        for r in results[:3]:
######
######            if isinstance(r, dict):
######                texts.append(r.get("text", "").lower())
######            else:
######                texts.append(str(r).lower())
######
######        expected = expected.lower().strip()
######
######        # -------------------
######        # top1 (strict match)
######        # -------------------
######
######        if texts and expected == texts[0].strip():
######            top1 += 1
######
######        # -------------------
######        # top3 (loose match)
######        # -------------------
######
######        if any(expected in t for t in texts):
######            top3 += 1
######
######        if index % 250 == 0:
######            print(f"{index}/{total}")
######
######    print()
######    print("=" * 40)
######    print(f"Questions : {total}")
######    print(f"Failed    : {failed}")
######    print()
######    print(f"Top1      : {top1}")
######    print(f"Top3      : {top3}")
######    print()
######
######    if total > 0:
######        print(f"Top1 %    : {(top1/total)*100:.2f}")
######        print(f"Top3 %    : {(top3/total)*100:.2f}")
######    else:
######        print("Top1 %    : 0.00")
######        print("Top3 %    : 0.00")
######
######    print("=" * 40)
######
####
####
######import json
######import requests
######
######BASE_URL = "http://localhost:8000"
######
######QUESTION_FILE = "benchmark_output/benchmark_questions.json"
######
######
######def query_memory(question):
######
######    r = requests.post(
######        f"{BASE_URL}/memory/query",
######        json={"text": question},
######        timeout=120
######    )
######
######    if not r.ok:
######
######        print(r.text)
######
######        return []
######
######    response = r.json()
######
######    if "results" not in response:
######        print("Unexpected response:", response)
######        return []
######
######    return response["results"]
######
######
######def evaluate():
######
######    with open(QUESTION_FILE, "r") as f:
######
######        questions = json.load(f)
######
######    total = len(questions)
######
######    top1 = 0
######
######    top3 = 0
######
######    failed = 0
######
######    for index, q in enumerate(questions, start=1):
######
######        query = q["query"]
######
######        expected = q["expected"]
######
######        results = query_memory(query)
######
######        if not results:
######
######            failed += 1
######
######            continue
######
######        #
######        # normalize results
######        #
######
######        texts = []
######
######        for r in results[:3]:
######
######            if isinstance(r, dict):
######
######                texts.append(r["text"].lower())
######
######            else:
######
######                texts.append(str(r).lower())
######
######        expected = q.get("expected")
######
######        if not isinstance(expected, str):
######            failed += 1
######            continue
######
######        expected = expected.lower()
######
######        #
######        # top1
######        #
######
######        if expected in texts[0]:
######
######            top1 += 1
######
######        #
######        # top3
######        #
######
######        if any(expected in t for t in texts):
######
######            top3 += 1
######
######        if index % 250 == 0:
######
######            print(f"{index}/{total}")
######
######    print()
######
######    print("=" * 40)
######
######    print(f"Questions : {total}")
######
######    print(f"Failed    : {failed}")
######
######    print()
######
######    print(f"Top1      : {top1}")
######
######    print(f"Top3      : {top3}")
######
######    print()
######
######    print(f"Top1 %    : {(top1/total)*100:.2f}")
######
######    print(f"Top3 %    : {(top3/total)*100:.2f}")
######
######    print("=" * 40)
######
######
######if __name__ == "__main__":
######
######    evaluate()
