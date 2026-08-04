"""
benchmark_runner.py

Runs benchmark questions against the local memory system.

No HTTP.
No API.
Writes flight recorder output.
"""

import json
import time
import argparse

from pathlib import Path

from benchmark.benchmark_writer import BenchmarkWriter
from shared.memory_interface import MemoryInterface


QUESTION_FILE = (
    "benchmark_output/"
    "benchmark_questions.json"
)


# -----------------------------------------
# SETTINGS
# -----------------------------------------

PROGRESS_INTERVAL = 5


# -----------------------------------------
# RUNNER
# -----------------------------------------

class BenchmarkRunner:


    def __init__(self):

        self.memory = MemoryInterface()

        self.writer = BenchmarkWriter()


    # -------------------------------------
    # LOAD QUESTIONS
    # -------------------------------------

    def load_questions(self):

        with open(
            QUESTION_FILE,
            "r",
            encoding="utf8"
        ) as f:

            return json.load(f)



    # -------------------------------------
    # FIND EXPECTED
    # -------------------------------------

    def find_expected_rank(
        self,
        results,
        expected
    ):

        expected = str(expected).lower()


        for result in results:

            text = (

                result.get(
                    "normalized_text"
                )

                or result.get(
                    "text",
                    ""
                )

            ).lower()


            if expected in text:

                return result.get(
                    "rank"
                )


        return None



    # -------------------------------------
    # RUN
    # -------------------------------------

    def run(self, limit=None):

        print()
        print("=" * 60)
        print("[BENCHMARK START]")
        print("=" * 60)


        questions = self.load_questions()
        if limit:
            questions = questions[:limit]


        total = len(questions)


        print(
            "[QUESTIONS]",
            total
        )


        start = time.perf_counter()


        next_progress = PROGRESS_INTERVAL



        for index, item in enumerate(
            questions,
            start=1
        ):

            query = item.get(
                "query"
            )

            expected = item.get(
                "expected"
            )


            if not query or not expected:

                continue



            query_start = time.perf_counter()


            response = self.memory.recall(
                query
            )


            query_time = (

                time.perf_counter()

                -

                query_start

            ) * 1000



            #
            # Allow diagnostics from system
            #

            if isinstance(
                response,
                dict
            ):

                results = response.get(
                    "results",
                    []
                )

                diagnostics = response.get(
                    "diagnostics",
                    {}
                )

            else:

                results = response

                diagnostics = {}



            expected_rank = (
                self.find_expected_rank(
                    results,
                    expected
                )
            )
##            print("\n===== BENCH DIAGNOSTICS =====")
##            print(diagnostics)
##            print("=============================\n")
##

            

            candidates=results

            self.writer.record(
                query = query,
                expected = expected,
                expected_rank = expected_rank,
                retrieved = (expected_rank is not None),
                candidates = results,
                runtime_ms = query_time,
                diagnostics = diagnostics

                )

            



            percent = int(
                index /
                total *
                100
            )


            if percent >= next_progress:


                elapsed = (
                    time.perf_counter()
                    -
                    start
                )


                rate = (
                    index /
                    max(
                        elapsed,
                        0.001
                    )
                )


                eta = (

                    total-index

                ) / rate



                print(

                    f"[PROGRESS] "
                    f"{percent}% "
                    f"{index}/{total} "
                    f"ETA {eta:.1f}s"

                )


                next_progress += (
                    PROGRESS_INTERVAL
                )



        runtime = (

            time.perf_counter()

            -

            start

        )


        outfile = self.writer.write()


        print()

        print(
            "[TOTAL RUNTIME]",
            round(runtime,2),
            "seconds"
        )


        return outfile



# -----------------------------------------
# MAIN
# -----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run first N benchmark questions"

        )
    args = parser.parse_args()

    runner = BenchmarkRunner()

    runner.run(
        limit=args.limit

        )
