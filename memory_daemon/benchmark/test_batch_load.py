from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader

memory = MemoryInterface()
loader = BatchLoader(memory)

data = loader.load_file(
    "benchmark_output/benchmark_memories.json"

    )
texts = loader.extract_texts(data)
loader.insert_batch(texts)
