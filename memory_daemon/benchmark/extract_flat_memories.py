#!/usr/bin/env python3
"""
Extract all memory texts from LongMemEval dataset into flat JSON array.
Optionally create a questions file for benchmarking.
"""

import json
import argparse
from pathlib import Path

def extract_memories_and_questions(input_path, output_memories, output_questions=None, limit=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
        print(f"Limiting to first {limit} questions")

    memories = []
    questions = []

    for item in data:
        # Extract memories from all turns
        for session in item.get('haystack_sessions', []):
            for turn in session:
                content = turn.get('content', '')
                if content and content.strip():
                    memories.append({"text": content})

        # Build question entry (only if we have expected answer)
        question_text = item.get('question', '')
        answer_ids = item.get('answer_session_ids', [])
        if question_text and answer_ids:
            # Find answer content from the corresponding session
            sessions = item.get('haystack_sessions', [])
            answer_content = ""
            for idx, session in enumerate(sessions):
                if idx < len(answer_ids) and answer_ids[idx]:
                    if session and session[0].get('content'):
                        answer_content = session[0]['content']
                        break

            if answer_content:
                questions.append({
                    "query": question_text,
                    "expected": answer_content,
                    "expected_ids": answer_ids
                })

    # Write memories
    with open(output_memories, 'w', encoding='utf-8') as f:
        json.dump(memories, f, indent=2)

    print(f"Extracted {len(memories)} memories to {output_memories}")

    if output_questions and questions:
        with open(output_questions, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2)
        print(f"Extracted {len(questions)} questions to {output_questions}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="longmemeval_s_cleaned.json", help="Input LongMemEval JSON file")
    parser.add_argument("--output-memories", default="benchmark_output/flat_memories.json", help="Output JSON for memories")
    parser.add_argument("--output-questions", default=None, help="Optional output JSON for questions")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N questions")
    args = parser.parse_args()

    extract_memories_and_questions(args.input, args.output_memories, args.output_questions, args.limit)
