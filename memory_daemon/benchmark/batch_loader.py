"""
batch_loader.py

Loads memory datasets into the memory system.

Designed for:
- synthetic generators
- dataset imports
- benchmark preparation

Avoids thousands of HTTP calls.
"""

import json
import time

from pathlib import Path


class BatchLoader:


    def __init__(self, memory_interface):

        self.memory = memory_interface



    # -----------------------------------------
    # LOAD FILE
    # -----------------------------------------

    def load_file(
        self,
        filepath
    ):

        path = Path(filepath)


        with open(
            path,
            "r",
            encoding="utf8"
        ) as f:

            data = json.load(f)


        return data



    # -----------------------------------------
    # NORMALIZE INPUT
    # -----------------------------------------

    def extract_texts(
        self,
        data
    ):

        texts = []


        for item in data:


            if isinstance(
                item,
                str
            ):

                texts.append(item)


            elif isinstance(
                item,
                dict
            ):

                text = (

                    item.get("text")

                    or

                    item.get("memory")

                )


                if text:

                    texts.append(text)


        return texts



    # -----------------------------------------
    # BATCH INSERT
    # -----------------------------------------

    def insert_batch(
        self,
        texts,
        batch_size=100
    ):


        total = len(texts)


        stored = 0


        start = time.perf_counter()



        print()

        print("=" * 60)

        print("[BATCH LOAD START]")

        print()

        print(
            "Memories:",
            total
        )

        print("=" * 60)



        for i in range(
            0,
            total,
            batch_size
        ):


            batch = texts[
                i:i+batch_size
            ]



            self.memory.store_many(
                batch
            )


            stored += len(batch)



            percent = (

                stored /
                total *
                100

            )


            print(

                f"[PROGRESS] "
                f"{stored}/{total} "
                f"({percent:.1f}%)"

            )



        elapsed = (

            time.perf_counter()

            -

            start

        )


        print()

        print("=" * 60)

        print("[BATCH COMPLETE]")

        print()

        print(
            "Stored:",
            stored
        )

        print(

            "Runtime:",

            round(
                elapsed,
                2
            ),

            "seconds"

        )

        print("=" * 60)



        return stored

##    def store_many(self, texts):
##        return self.rememeber_many(texts)

# -----------------------------------------
# CLI TEST
# -----------------------------------------

if __name__ == "__main__":

    print(
        "BatchLoader module loaded."
    )
