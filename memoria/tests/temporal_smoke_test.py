from datetime import datetime, timezone

from blackboard.temporal import TemporalWorker
from blackboard.temporal_index import TemporalIndex


UTC = timezone.utc
REFERENCE = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


RECORDS = [
    {"id": 1, "created_at": "2026-09-28T10:00:00Z", "updated_at": "2026-09-28T10:00:00Z"},
    {"id": 2, "created_at": "2026-09-27T10:00:00Z", "updated_at": "2026-09-27T10:00:00Z"},
    {"id": 3, "created_at": "2026-09-25T10:00:00Z", "updated_at": "2026-09-25T10:00:00Z"},
    {"id": 4, "created_at": "2026-09-20T10:00:00Z", "updated_at": "2026-09-20T10:00:00Z"},
    {"id": 5, "created_at": "2026-01-15T10:00:00Z", "updated_at": "2026-01-15T10:00:00Z"},
    {"id": 6, "created_at": "2025-12-31T10:00:00Z", "updated_at": "2025-12-31T10:00:00Z"},
]


index = TemporalIndex()

print("\n" + "=" * 60)
print("TEMPORAL INDEX")
print("=" * 60)

built = index.build(RECORDS)
print("records built:", built)
print("index length:", len(index))

print("\nexact date: 2026-09-28")
print(index.search_exact_date(REFERENCE))

print("\nexact date: 2026-09-27")
print(index.search_exact_date(
    datetime(2026, 9, 27, tzinfo=UTC)
))

print("\nyear: 2026")
print(index.search_year(2026))

print("\nbefore: 2026-09-27")
print(index.search_before(
    datetime(2026, 9, 27, tzinfo=UTC),
    10,
))

print("\nafter: 2026-09-27")
print(index.search_after(
    datetime(2026, 9, 27, tzinfo=UTC),
    10,
))


worker = TemporalWorker(index)


def run(query):
    result = worker.process({
        "query": query,
        "top_k": 10,
        "reference_time": REFERENCE,
    })

    print("\n" + "-" * 60)
    print(f"QUERY: {query}")
    print("-" * 60)
    print("active:", result.get("active"))
    print("candidates:", result.get("candidates"))

    return result


print("\n" + "=" * 60)
print("TEMPORAL WORKER")
print("=" * 60)

run("what happened today")
run("what happened yesterday")
run("what happened 3 days ago")
run("what happened in 2026")
run("what happened before 2026-09-27")
run("what happened after 2026-09-27")
run("what happened between 2026-09-20 and 2026-09-28")
run("what is the latest")

print("\n" + "=" * 60)
print("NON-TEMPORAL")
print("=" * 60)

run("who is Bob")
run("when did I meet Bob")

print("\n" + "=" * 60)
print("TEMPORAL SMOKE TEST COMPLETE")
print("=" * 60)
