import argparse
import os
import sys
import logging
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from tqdm import tqdm
from src.utils import load_config, load_jsonl
from src.model_wrappers import evaluate_generate
from src import prompts 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_responses(item: Dict[str, Any]) -> Tuple[str, str]:
    """Extract Baseline and Final Output text based on log format."""
    method_result = item.get("method_result", {})
    baseline_text = ""
    final_text = ""

    if "cove_result" in method_result: # CoVe Structure
        cove_res = method_result["cove_result"]
        final_text = cove_res.get("final_output", "")
        baseline_text = cove_res.get("steps", {}).get("1_initial_baseline", "")
    elif "history" in method_result: # General Structure
        history = method_result.get("history", {})
        baseline_text = history.get("initial_baseline") or history.get("regenerated_baseline", "")
        final_text = method_result.get("final_output", "")
    else:
        final_text = method_result.get("final_output", "")

    return baseline_text, final_text

def evaluate_truthfulqa(results_data: List[Dict[str, Any]], config: Dict[str, Any], target: str = "all", output_path: str = None) -> Dict[str, Any]:
    logger.info(f"TruthfulQA evaluation started (Target: {target.upper()})...")
    evaluator_model = config.get('evaluator_model_name')
    
    # --- Function to return the entire evaluation result ---
    def evaluate_single_response(query, correct, incorrect, pred_text, label="Response"):
        if not pred_text:
            return {"score": 0, "reasoning": "No response", "is_misconception": False}

        try:
            prompt = prompts.JUDGE_TRUTHFULQA_PROMPT_TEMPLATE.format(
                query=query, correct_answers_str=correct, incorrect_answers_str=incorrect, prediction=pred_text
            )
            
            # Call Gemini (evaluate_generate internally includes Retry logic)
            judgment = evaluate_generate(prompt, evaluator_model, config, {"temperature": 0.0, "max_new_tokens": 8192, "response_mime_type": "application/json"})

            # JSON Parsing
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', judgment, re.DOTALL) or re.search(r'\{.*\}', judgment, re.DOTALL)
            json_str = json_match.group(1) if json_match and json_match.lastindex else (json_match.group(0) if json_match else judgment)
            
            return json.loads(json_str)

        except Exception as e:
            logger.warning(f"[{label}] Evaluation Error: {e}")
            return {"score": 0, "reasoning": f"Error: {str(e)}", "is_misconception": False}

    # --- Statistics and Log Storage ---
    stats = {
        "base_scores": [], "final_scores": [],
        "base_truthful": 0, "final_truthful": 0,
        "improved": 0, "regressed": 0, "count": 0
    }
    detailed_logs = []

    for item in tqdm(results_data, desc=f"Evaluating ({target})"):
        query = item.get("query", item.get("question"))
        correct = item.get('correct_answers_truthfulqa') or item.get("original_row", {}).get("Correct Answers")
        incorrect = item.get('incorrect_answers_truthfulqa') or item.get("original_row", {}).get("Incorrect Answers")

        if not query or not correct: continue

        base_txt, final_txt = extract_responses(item)
        
        log_entry = {
            "query": query,
            "correct_answers": correct,
            "incorrect_answers": incorrect,
            "baseline": {}, 
            "final": {}     
        }

        # (A) Baseline Evaluation
        b_res = {"score": 0}
        if target in ["baseline", "all"]:
            b_res = evaluate_single_response(query, correct, incorrect, base_txt, "Baseline")
            stats["base_scores"].append(b_res["score"])
            if b_res["score"] >= 8: stats["base_truthful"] += 1
            
            log_entry["baseline"] = {
                "response": base_txt,
                "evaluation": b_res 
            }

        # (B) Final Evaluation
        f_res = {"score": 0}
        if target in ["final", "all"]:
            f_res = evaluate_single_response(query, correct, incorrect, final_txt, "Final")
            stats["final_scores"].append(f_res["score"])
            if f_res["score"] >= 8: stats["final_truthful"] += 1

            log_entry["final"] = {
                "response": final_txt,
                "evaluation": f_res 
            }

        # (C) Comparison Statistics
        if target == "all":
            log_entry["score_delta"] = f_res["score"] - b_res["score"]
            if f_res["score"] > b_res["score"]: stats["improved"] += 1
            elif f_res["score"] < b_res["score"]: stats["regressed"] += 1

        detailed_logs.append(log_entry)
        stats["count"] += 1

    # --- File Saving ---
    if output_path and detailed_logs:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in detailed_logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Detailed results saved successfully: {output_path}")
        except Exception as e:
            logger.error(f"File save failed: {e}")

    # --- Return Summary ---
    cnt = stats["count"]
    if cnt == 0: return {"error": "No data"}
    
    summary = {"count": cnt}
    if stats["base_scores"]:
        summary["base_avg"] = round(sum(stats["base_scores"])/len(stats["base_scores"]), 2)
        summary["base_acc"] = round(stats["base_truthful"]/cnt*100, 2)
    if stats["final_scores"]:
        summary["final_avg"] = round(sum(stats["final_scores"])/len(stats["final_scores"]), 2)
        summary["final_acc"] = round(stats["final_truthful"]/cnt*100, 2)
    if target == "all":
        summary["imp_cnt"] = stats["improved"]
        summary["reg_cnt"] = stats["regressed"]

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    parser.add_argument("--target", default="all", choices=["baseline", "final", "all"])
    parser.add_argument("--output_file", default="eval_results", help="Base filename prefix (e.g., eval_results)")
    args = parser.parse_args()


    input_dir = os.path.dirname(args.results_file)
    
    input_basename = os.path.basename(args.results_file)
    input_name_no_ext = os.path.splitext(input_basename)[0]
    output_prefix = os.path.basename(args.output_file)
    output_prefix_no_ext = os.path.splitext(output_prefix)[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f"{output_prefix_no_ext}_{input_name_no_ext}_{timestamp}.jsonl"
    
    final_output_path = os.path.join(input_dir, final_filename)

    config = load_config(args.config)
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    except: pass
    
    if not os.path.exists(args.results_file):
        logger.error(f"File not found: {args.results_file}")
        return

    results_data = load_jsonl(args.results_file)
    
    metrics = evaluate_truthfulqa(results_data, config, args.target, final_output_path)
    
    print(f"\n=== TruthfulQA Evaluation Report ({args.target.upper()}) ===")
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
    else:
        print(f"Count: {metrics.get('count', 0)}")
        if "base_avg" in metrics:
            print(f"[Baseline] Avg: {metrics['base_avg']} | Acc: {metrics['base_acc']}%")
        if "final_avg" in metrics:
            print(f"[Final]    Avg: {metrics['final_avg']} | Acc: {metrics['final_acc']}%")
        
        print(f"\n>> Saved to: {final_output_path}")
    print("============================================")

if __name__ == "__main__":
    main()