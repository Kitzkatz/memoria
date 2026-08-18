#!/usr/bin/env python3
"""
Extract questions from LongMemEval dataset into benchmark runner/analyzer format.
Includes question_id for robust matching.
"""

import json
import argparse
from pathlib import Path

def extract_questions(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []

    for idx, item in enumerate(data):
        q_text = item.get('question', '')
        if not q_text:
            continue

        question_id = item.get('question_id', f'q_{idx}')

        answer_ids = item.get('answer_session_ids', [])
        sessions = item.get('haystack_sessions', [])
        expected_text = ""
        for sess_idx, session in enumerate(sessions):
            if sess_idx < len(answer_ids) and answer_ids[sess_idx]:
                if session and session[0].get('content'):
                    expected_text = session[0]['content']
                    break

        if not expected_text:
            print(f"Warning: No answer content found for question: {q_text[:60]}...")
            continue

        questions.append({
            "question_id": question_id,
            "query": q_text,
            "expected": expected_text,
            "expected_ids": answer_ids   # list of session IDs
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2)

    print(f"Extracted {len(questions)} questions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="longmemeval_s_cleaned.json")
    parser.add_argument("--output", default="benchmark_output/longmemeval_questions.json")
    args = parser.parse_args()
    extract_questions(args.input, args.output)
