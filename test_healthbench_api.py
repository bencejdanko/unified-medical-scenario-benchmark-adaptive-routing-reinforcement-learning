"""
Test script: HealthBench API call diagnostics
==============================================
Tests each model's response + grader JSON parsing for HealthBench.
Run: python test_healthbench_api.py
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

MODELS = {
    "gemini-3-flash-preview": {
        "model": "google/gemini-3-flash-preview",
        "api_key": OPENROUTER_API_KEY,
        "api_base": OPENROUTER_API_URL,
    },
    "gemma-4-31b-it": {
        "model": "google/gemma-4-31b-it",
        "api_key": OPENROUTER_API_KEY,
        "api_base": OPENROUTER_API_URL,
    },
    "gpt-oss-120b": {
        "model": "gpt-oss-120b",
        "api_key": CEREBRAS_API_KEY,
        "api_base": "https://api.cerebras.ai/v1",
    },
}

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def parse_json_to_dict(json_string: str) -> dict:
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError:
        return {}


def main():
    # Load first HealthBench example
    with open("healthbench_cube/data/healthbench.jsonl") as f:
        example = json.loads(f.readline())

    prompt = example["prompt"]
    rubrics = example["rubrics"]

    print("=" * 60)
    print("HEALTHBENCH API DIAGNOSTIC TEST")
    print("=" * 60)
    print(f"\nPrompt ({len(prompt)} messages):")
    for m in prompt:
        print(f"  [{m['role']}] {m['content'][:120]}...")
    print(f"\nRubrics ({len(rubrics)} items):")
    for i, r in enumerate(rubrics):
        print(f"  [{i}] ({r['points']:+.0f}pts) {r['criterion'][:100]}...")
    print()

    # --- Step 1: Test each model's response ---
    model_responses = {}
    for name, cfg in MODELS.items():
        print(f"--- Model: {name} ---")
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=prompt,
                temperature=0.5,
                max_tokens=4096,
            )
            text = resp.choices[0].message.content or ""
            model_responses[name] = text
            print(f"  finish_reason: {resp.choices[0].finish_reason}")
            print(f"  length: {len(text)} chars")
            print(f"  preview: {text[:200]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            model_responses[name] = None
        print()

    # --- Step 2: Test grader on first model's response against first 3 rubric items ---
    print("=" * 60)
    print("GRADER TEST (gemini-3-flash-preview as judge)")
    print("=" * 60)

    # Pick first model that has a response
    test_model = next((n for n, r in model_responses.items() if r), None)
    if not test_model:
        print("No model responses available, cannot test grader")
        return

    test_response = model_responses[test_model]
    print(f"\nGrading {test_model}'s response against first 3 rubric items:\n")

    grader_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_API_URL)
    convo_with_response = prompt + [{"role": "assistant", "content": test_response}]
    convo_str = "\n\n".join([f"{m['role']}: {m['content']}" for m in convo_with_response])

    for i, rubric in enumerate(rubrics[:3]):
        grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
            "<<rubric_item>>", rubric["criterion"]
        )

        print(f"  Rubric [{i}] ({rubric['points']:+.0f}pts): {rubric['criterion'][:80]}...")
        try:
            grader_resp = grader_client.chat.completions.create(
                model="google/gemini-3-flash-preview",
                messages=[{"role": "user", "content": grader_prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = grader_resp.choices[0].message.content or ""
            print(f"  Raw response ({len(raw)} chars):")
            print(f"    {repr(raw[:300])}")

            parsed = parse_json_to_dict(raw)
            if "criteria_met" in parsed:
                print(f"  PARSED OK: criteria_met={parsed['criteria_met']}")
                print(f"    explanation: {parsed.get('explanation', '')[:150]}...")
            else:
                print(f"  PARSE FAILED - keys: {list(parsed.keys())}")
                print(f"    Full raw: {raw}")
        except Exception as e:
            print(f"  GRADER ERROR: {e}")
        print()

    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
