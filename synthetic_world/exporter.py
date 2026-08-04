import json
import csv
import os


class Exporter:

    def __init__(self, world, output_dir="benchmark_output"):

        self.world = world

        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------------

    def write(self):

        self._write_memories()

        self._write_questions()

        self._write_truth()

        self._write_summary()
        self._write_txt()

    # --------------------------------------------------------
    # MEMORIES (TXT)
    # --------------------------------------------------------
    def _write_txt(self):

        path = os.path.join(self.output_dir, "benchmark_memories.txt")

        with open(path, "w", encoding="utf-8") as f:

            for mem in self.world.memories:

                f.write(mem + "\n")

        print(f"[EXPORT] Memories -> {path}")
    # --------------------------------------------------------
    # MEMORIES (JSON)
    # --------------------------------------------------------

    def _write_memories(self):

        path = os.path.join(
            self.output_dir,
            "benchmark_memories.json"
        )


        memories = []


        for memory in self.world.memories:

            if isinstance(memory, str):

                memories.append({
                    "text": memory
                })

            elif isinstance(memory, dict):

                memories.append(memory)

            else:

                memories.append({
                    "text": str(memory)
                })


        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                memories,
                f,
                indent=4,
                ensure_ascii=False
            )


        print(
            f"[EXPORT] Memories -> {path}"
        )

    # --------------------------------------------------------
    # QUESTIONS (JSON)
    # --------------------------------------------------------

    def _write_questions(self):

        path = os.path.join(self.output_dir, "benchmark_questions.json")

        with open(path, "w", encoding="utf-8") as f:

            json.dump(self.world.questions, f, indent=2)

        print(f"[EXPORT] Questions -> {path}")

    # --------------------------------------------------------
    # TRUTH TABLE (CSV)
    # --------------------------------------------------------

    def _write_truth(self):

        path = os.path.join(self.output_dir, "benchmark_truth.csv")

        with open(path, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                "person",
                "food",
                "vehicle",
                "city",
                "job",
                "phone",
                "birthday"
            ])

            for p in self.world.people:

                writer.writerow([
                    p.name,
                    p.food,
                    p.vehicle,
                    p.city,
                    p.job,
                    p.phone,
                    p.birthday
                ])

        print(f"[EXPORT] Truth -> {path}")

    # --------------------------------------------------------
    # SUMMARY METADATA
    # --------------------------------------------------------

    def _write_summary(self):

        path = os.path.join(self.output_dir, "benchmark_summary.json")

        summary = {

            "people": len(self.world.people),

            "memories": len(self.world.memories),

            "questions": len(self.world.questions),

            "stats": dict(self.world.stats)

        }

        with open(path, "w", encoding="utf-8") as f:

            json.dump(summary, f, indent=2)

        print(f"[EXPORT] Summary -> {path}")
