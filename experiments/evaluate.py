# experiments/evaluate.py
import argparse
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Callable

# --- 프로젝트 루트 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import load_config, load_jsonl, save_jsonl, get_timestamp
try:
    # 1. TruthfulQA (예시)
    from eval_tools.truthfulqa.evaluate import evaluate as truthfulqa_official_eval
except ImportError:
    logging.warning("TruthfulQA 공식 평가 스크립트('eval_tools/truthfulqa/evaluate.py')를 찾을 수 없습니다.")
    truthfulqa_official_eval = None

try:
    # 2. FactScore (예시)
    from eval_tools.factscore.factscore import FactScorer
    # factscorer = FactScorer() # 필요시 전역 초기화
except ImportError:
    logging.warning("FactScore 공식 스크립트('eval_tools/factscore/factscore.py')를 찾을 수 없습니다.")
    FactScorer = None

try:
    from eval_tools.hallulens_eval.eval_precisewikiqa import evaluate_precisewikiqa as precisewikiqa_official_eval
except ImportError:
    logging.warning("HalluLens PreciseWikiQA 공식 평가 스크립트('eval_tools/hallulens_eval/eval_precisewikiqa.py')를 찾을 수 없습니다.")
    precisewikiqa_official_eval = None


# --- 벤치마크별 평가 함수 (모두 플레이스홀더화) ---

def evaluate_qa_benchmark(results_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """HalluLens PreciseWikiQA 결과를 공식 스크립트로 평가합니다."""
    logger.info("PreciseWikiQA 평가 시작 (공식 스크립트 연동)...")
    if precisewikiqa_official_eval is None:
        return {"error": "PreciseWikiQA 평가 스크립트가 로드되지 않았습니다."}
    
    # HalluLens 공식 평가 스크립트 사용법에 맞게 구현
    # 1. results_data에서 'final_output' (예측) 리스트 추출
    # 2. results_data에서 'answers' (정답) 리스트 추출
    # 3. metrics = precisewikiqa_official_eval(predictions, ground_truths)
    # 4. 결과 딕셔너리 반환 (EM, F1 등)
    
    logger.warning("PreciseWikiQA 공식 스크립트 연동 로직 구현 필요.")
    return {"status": "PreciseWikiQA evaluation (placeholder)"}

def evaluate_factscore(results_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """LongWiki (FactScore) 벤치마크 결과를 평가합니다."""
    logger.info("FactScore 평가 시작 (공식 스크립트 연동)...")
    if FactScorer is None:
        return {"error": "FactScore 라이브러리가 로드되지 않았습니다."}
    
    # TODO: FactScorer 라이브러리 사용법에 맞게 구현
    # ... (기존 플레이스홀더 내용과 동일) ...
    
    logger.warning("FactScore 공식 스크립트 연동 로직 구현 필요.")
    return {"status": "FactScore evaluation (placeholder)"}

def evaluate_truthfulqa(results_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """TruthfulQA 벤치마크 결과를 평가합니다."""
    logger.info("TruthfulQA 평가 시작 (공식 스크립트 연동)...")
    if truthfulqa_official_eval is None:
        return {"error": "TruthfulQA 평가 스크립트가 로드되지 않았습니다."}

    # TODO: TruthfulQA 공식 스크립트 사용법에 맞게 구현
    # ... (기존 플레이스홀더 내용과 동일) ...
    
    logger.warning("TruthfulQA 공식 스크립트 연동 로직 구현 필요.")
    return {"status": "TruthfulQA evaluation (placeholder)"}


# --- 메인 평가 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment results.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the .jsonl results file.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"), help="Path to config file.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Specify dataset (e.g., hallulens_precisewikiqa). If None, inferred from path.")
    args = parser.parse_args()

    # --- 설정 로드 ---
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"설정 파일 로드 실패 ({args.config}): {e}")
        return

    # --- 결과 파일 로드 ---
    results_data = load_jsonl(args.results_file)
    if not results_data:
         logging.error("결과 파일이 비어있음.")
         return
    logging.info(f"{args.results_file} 에서 {len(results_data)}개의 결과 로드.")
    
    # --- 데이터셋 이름 결정 ---
    dataset_name = args.dataset_name
    if not dataset_name:
        try:
            parts = args.results_file.replace("\\", "/").split('/')
            results_index = parts.index('results')
            dataset_name = parts[results_index + 2] # e.g., results/model/DATASET/file.jsonl
            logging.info(f"결과 파일 경로에서 데이터셋 이름 추론: {dataset_name}")
        except (ValueError, IndexError):
            logging.error("데이터셋 이름을 추론할 수 없습니다. --dataset_name 인자를 사용해주세요.")
            return

    # --- 벤치마크별 평가 함수 선택 및 실행 ---
    evaluation_func: Optional[Callable] = None
    name_lower = dataset_name.lower()
    
    if 'hallulens_longwiki' in name_lower:
        evaluation_func = evaluate_factscore
    elif 'hallulens_precisewikiqa' in name_lower:
        evaluation_func = evaluate_qa_benchmark 
    elif 'truthfulqa' in name_lower:
        evaluation_func = evaluate_truthfulqa
    else:
        logging.error(f"'{dataset_name}'에 대한 평가 함수를 찾을 수 없습니다.")
        return

    metrics = evaluation_func(results_data, config)

    # --- 결과 출력 ---
    print(f"\n--- Evaluation Results for {dataset_name} ({args.results_file}) ---")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                 print(f"{key}: {value:.4f}")
            else:
                 print(f"{key}: {value}")
    else:
        print(metrics)
    print("-----------------------------------------------------")

if __name__ == "__main__":
    main()

