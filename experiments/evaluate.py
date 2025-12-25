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
    """로그 포맷에 따라 Baseline과 Final Output 텍스트 추출"""
    method_result = item.get("method_result", {})
    baseline_text = ""
    final_text = ""

    if "cove_result" in method_result: # CoVe 구조
        cove_res = method_result["cove_result"]
        final_text = cove_res.get("final_output", "")
        baseline_text = cove_res.get("steps", {}).get("1_initial_baseline", "")
    elif "history" in method_result: # 일반 구조
        history = method_result.get("history", {})
        baseline_text = history.get("initial_baseline") or history.get("regenerated_baseline", "")
        final_text = method_result.get("final_output", "")
    else:
        final_text = method_result.get("final_output", "")

    return baseline_text, final_text

def evaluate_truthfulqa(results_data: List[Dict[str, Any]], config: Dict[str, Any], target: str = "all", output_path: str = None) -> Dict[str, Any]:
    logger.info(f"TruthfulQA 평가 시작 (Target: {target.upper()})...")
    evaluator_model = config.get('evaluator_model_name')
    
    # --- 평가 결과를 통째로 반환하는 함수 ---
    def evaluate_single_response(query, correct, incorrect, pred_text, label="Response"):
        if not pred_text:
            return {"score": 0, "reasoning": "No response", "is_misconception": False}

        try:
            prompt = prompts.JUDGE_TRUTHFULQA_PROMPT_TEMPLATE.format(
                query=query, correct_answers_str=correct, incorrect_answers_str=incorrect, prediction=pred_text
            )
            
            # Gemini 호출 (evaluate_generate는 내부적으로 Retry 로직 포함됨)
            judgment = evaluate_generate(prompt, evaluator_model, config, {"temperature": 0.0, "max_new_tokens": 8192, "response_mime_type": "application/json"})

            # JSON 파싱
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', judgment, re.DOTALL) or re.search(r'\{.*\}', judgment, re.DOTALL)
            json_str = json_match.group(1) if json_match and json_match.lastindex else (json_match.group(0) if json_match else judgment)
            
            return json.loads(json_str)

        except Exception as e:
            logger.warning(f"[{label}] 평가 에러: {e}")
            return {"score": 0, "reasoning": f"Error: {str(e)}", "is_misconception": False}

    # --- 통계 및 로그 저장소 ---
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

        # (A) Baseline 평가
        b_res = {"score": 0}
        if target in ["baseline", "all"]:
            b_res = evaluate_single_response(query, correct, incorrect, base_txt, "Baseline")
            stats["base_scores"].append(b_res["score"])
            if b_res["score"] >= 8: stats["base_truthful"] += 1
            
            log_entry["baseline"] = {
                "response": base_txt,
                "evaluation": b_res 
            }

        # (B) Final 평가
        f_res = {"score": 0}
        if target in ["final", "all"]:
            f_res = evaluate_single_response(query, correct, incorrect, final_txt, "Final")
            stats["final_scores"].append(f_res["score"])
            if f_res["score"] >= 8: stats["final_truthful"] += 1

            log_entry["final"] = {
                "response": final_txt,
                "evaluation": f_res 
            }

        # (C) 비교 통계
        if target == "all":
            log_entry["score_delta"] = f_res["score"] - b_res["score"]
            if f_res["score"] > b_res["score"]: stats["improved"] += 1
            elif f_res["score"] < b_res["score"]: stats["regressed"] += 1

        detailed_logs.append(log_entry)
        stats["count"] += 1

    # --- 파일 저장 ---
    if output_path and detailed_logs:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in detailed_logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"상세 결과 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"파일 저장 실패: {e}")

    # --- 요약 리턴 ---
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

    # ---------------------------------------------------------
    # [경로 설정 로직 수정]
    # 입력 파일이 있는 디렉토리에 결과 파일을 저장합니다.
    # ---------------------------------------------------------
    
    # 1. 입력 파일의 디렉토리 경로 추출
    input_dir = os.path.dirname(args.results_file)
    
    # 2. 입력 파일명(확장자 제외) 추출
    input_basename = os.path.basename(args.results_file)
    input_name_no_ext = os.path.splitext(input_basename)[0]
    
    # 3. 출력 파일명 생성 (접두사 + 입력파일명 + 타임스탬프)
    # output_file 인자가 경로를 포함하더라도 파일명 부분만 사용 ('eval_results' 등)
    output_prefix = os.path.basename(args.output_file)
    output_prefix_no_ext = os.path.splitext(output_prefix)[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f"{output_prefix_no_ext}_{input_name_no_ext}_{timestamp}.jsonl"
    
    # 4. 최종 경로 합치기 (입력 디렉토리 + 생성된 파일명)
    final_output_path = os.path.join(input_dir, final_filename)

    config = load_config(args.config)
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    except: pass
    
    if not os.path.exists(args.results_file):
        logger.error(f"파일 없음: {args.results_file}")
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