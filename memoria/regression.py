from core.logger import debug
import requests

BASE = "http://localhost:8000"


TESTS = [
    {
        "text": "Alice likes tacos",
        "query": "Alice likes",
        "expect": "tacos"
    },
    {
        "text": "Bob drives a blue truck",
        "query": "Bob drives",
        "expect": "truck"
    },
    {
        "text": "Charlie lives in Detroit",
        "query": "Charlie lives",
        "expect": "detroit"
    }
]


def run():

    debug("\n[REGRESSION START]\n")

    for t in TESTS:

        # SAFE INSERT
        r = requests.post(
            f"{BASE}/memory/test_store",
            json={"text": t["text"]}
        )

        assert r.status_code == 200

        # QUERY
        r = requests.post(
            f"{BASE}/memory/query",
            json={"text": t["query"]}
        )

        data = r.json()
        results = data.get("results", [])

        joined = " ".join(
            str(x) for x in results
        ).lower()

        ok = t["expect"].lower() in joined

        debug("TEST:", t["text"])
        debug("PASS" if ok else "FAIL")
        debug()

    # ALWAYS REPAIR INDEX AFTER TESTS
    # CLEANUP AT END
    requests.post(
        f"{BASE}/memory/test_cleanup",
        json={"texts": created_ids}
    )
    requests.post(f"{BASE}/memory/rebuild_index")

    debug("\n[REGRESSION DONE]\n")


if __name__ == "__main__":
    run()
