from core.logger import debug
import requests


BASE = "http://localhost:8000"


TESTS = [
    {
        "text": "Alice likes tacos",
        "query": "Alice likes",
        "expect": "tacos",
    },
    {
        "text": "Bob drives a blue truck",
        "query": "Bob drives",
        "expect": "truck",
    },
    {
        "text": "Charlie lives in Detroit",
        "query": "Charlie lives",
        "expect": "detroit",
    },
]


def run():
    debug("\n[REGRESSION START]\n")

    created_texts = []

    for test in TESTS:

        # SAFE INSERT
        response = requests.post(
            f"{BASE}/memory/test_store",
            json={"text": test["text"]},
        )

        assert response.status_code == 200
        created_texts.append(test["text"])

        # QUERY
        response = requests.post(
            f"{BASE}/memory/query",
            json={"text": test["query"]},
        )

        assert response.status_code == 200

        data = response.json()
        results = data.get("results", [])

        joined = " ".join(
            str(result) for result in results
        ).lower()

        ok = test["expect"].lower() in joined

        debug("TEST:", test["text"])
        debug("PASS" if ok else "FAIL")
        debug()

    # CLEANUP AT END
    requests.post(
        f"{BASE}/memory/test_cleanup",
        json={"texts": created_texts},
    )

    # ALWAYS REPAIR INDEX AFTER TESTS
    requests.post(
        f"{BASE}/memory/rebuild_index"
    )

    debug("\n[REGRESSION DONE]\n")


if __name__ == "__main__":
    run()
