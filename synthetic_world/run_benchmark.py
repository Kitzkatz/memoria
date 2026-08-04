import json
import csv
import argparse




# --------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------

def load_data():

    with open("benchmark_output/benchmark_memories.txt", "r") as f:
        memories = [l.strip() for l in f if l.strip()]

    with open("benchmark_output/benchmark_questions.json", "r") as f:
        questions = json.load(f)

    return memories, questions


# --------------------------------------------------------
# STORE VIA DAEMON (THIS IS THE KEY FIX)
# --------------------------------------------------------
##
##def store_memories(memories):
##    from shared.memory_interface import MemoryInterface
##    memory = MemoryInterface()
##    return memory.remember_many(memories)
##
### --------------------------------------------------------
### QUERY VIA DAEMON
### --------------------------------------------------------
##
##def query_memory(q):
##
##    r = requests.post(
##        f"{BASE_URL}/memory/query",
##        json={"text": q},
##        timeout=60
##    )
##
##    if not r.ok:
##
##        return []
##
##    return r.json()
##
##
### --------------------------------------------------------
### EVAL LOOP
### --------------------------------------------------------
##
##def run_eval(memories, questions):
##
##    print("\n[1] Storing memories via daemon...\n")
##
##    ids = store_memories(memories)
##
##    print(f"Stored {len(ids)} memories")
##
##    print("\n[2] Running queries...\n")
##
##    total = len(questions)
##
##    top1 = 0
##
##    top3 = 0
##
##    for i, q in enumerate(questions):
##
##        results = query_memory(q["query"])
##
##        expected = q.get("expected")
##
##        if not results:
##
##            continue
##
##        top_texts = [r["text"] for r in results[:3]]
##
##        if expected and expected in results[0]["text"]:
##
##            top1 += 1
##
##        if expected and any(expected in t for t in top_texts):
##
##            top3 += 1
##
##        if i % 500 == 0:
##
##            print(f"Queried {i}/{total}")
##
##    print("\n========== RESULTS ==========\n")
##
##    print(f"Top-1 Accuracy: {top1 / total:.4f}")
##
##    print(f"Top-3 Accuracy: {top3 / total:.4f}")
##
##    print("\n=============================\n")
##

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--people", type=int, default=500)

    args = parser.parse_args()

    print("\n[RUNNING GENERATOR]\n")

    # still generate dataset files
    from benchmark_generator import build_world

    build_world(args.people)

    memories, questions = load_data()

    #run_eval(memories, questions)


if __name__ == "__main__":

    main()
