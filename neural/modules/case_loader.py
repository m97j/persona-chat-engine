import json
import os

from inference import run_inference
from webtest_prompt import build_webtest_prompt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_CASES_PATH = os.path.join(BASE_DIR, "test_cases.json")

with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
    TEST_CASES = json.load(f)

def get_case_names():
    return [f"{i+1}. {c['input'].get('description','')}" for i, c in enumerate(TEST_CASES)]

def load_cases():
    return TEST_CASES

def load_case(idx):
    cases = load_cases()
    return cases[idx]

def run_case(idx, player_utt):
    case = TEST_CASES[idx]["input"].copy()
    case["player_utterance"] = player_utt
    prompt = build_webtest_prompt(case["npc_id"], case["npc_location"], player_utt)
    result = run_inference(prompt)
    return result["npc_output_text"], result["deltas"], result["flags_prob"]
