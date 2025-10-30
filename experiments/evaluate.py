# experiments/evaluate.py
import argparse
import os
import sys
import logging
import re
import string
from collections import Counter
from typing import Dict, Any, List, Optional, Callable

# --- 프로젝트 루트 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import load_config, load_jsonl, save_jsonl, get_timestamp
# (FactScore 등 다른 평가 라이브러리 임포트)
# from factscore import FactScorer 
# from src.evaluation_truthfulqa import evaluate_truthfulqa # 예시

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 벤치마크별 평가 함수 ---

def normalize_answer(s: str) -> str:
    """답변 정규화 (소문자, 구두점/관사 제거, 공백 정리)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_qa_metrics_for_item(prediction: str, ground_truths: List[str]) -> Dict[str, float]:
    """단일 항목에 대해 EM과 F1 점수를 계산합니다 (SQuAD/HotpotQA 방식)."""
    if not ground_truths:
        return {'em': 0.0, 'f1': 0.0} # 정답이 없으면 0점

    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truths = [normalize_answer(gt) for gt in ground_truths]

    # EM (Exact Match)
    em_score = 0.0
    if normalized_prediction in normalized_ground_truths:
        em_score = 1.0

    # F1 Score
    f1_scores = []
    pred_tokens = normalized_prediction.split()
    for gt in normalized_ground_truths:
        gt_tokens = gt.split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1_scores.append(0.0)
            continue
            
        precision = 1.0 * num_same / len(pred_tokens) if pred_tokens else 0
        recall = 1.0 * num_same / len(gt_tokens) if gt_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    # 여러 정답 중 가장 높은 F1 점수를 해당 항목의 F1 점수로 사용
    f1_score = max(f1_scores) if f1_scores else 0.0

    return {'em': em_score, 'f1': f1_score}


def evaluate_qa_benchmark(results_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """MultiSpanQA 또는 PreciseWikiQA와 같은 QA 벤치마크 결과를 평가합니다."""
    logger.info("QA (EM/F1) 평가 시작...")
    
    em_sum = 0
    f1_sum = 0
    evaluated_count = 0
    
    for item in results_data:
        pred = item.get("method_result", {}).get("final_output")
        # data_loader에서 'answers' 키로 정답 리스트를 저장했다고 가정
        gold_list = item.get("answers") 
        
        if pred is None or gold_list is None:
             logger.warning(f"예측({pred is None}) 또는 정답({gold_list is None})이 없어 평가 건너<0xEB><0x9B><0x8D>니다. Query: {item.get('query')}")
             continue
             
        # 단일 항목 점수 계산
        metrics = calculate_qa_metrics_for_item(pred, gold_list)
        em_sum += metrics['em']
        f1_sum += metrics['f1']
        evaluated_count += 1

    em_avg = (em_sum / evaluated_count) * 100 if evaluated_count > 0 else 0
    f1_avg = (f1_sum / evaluated_count) * 100 if evaluated_count > 0 else 0
    
    logger.info(f"QA 평가 완료: EM={em_avg:.2f}%, F1={f1_avg:.2f}% ({evaluated_count}개 항목)")
    return {"exact_match_avg": em_avg, "f1_score_avg": f1_avg, "evaluated_count": evaluated_count}

def evaluate_factscore(results_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """LongWiki (FactScore) 벤치마크 결과를 평가합니다."""
    logger.info("FactScore 평가 시작 (플레이스홀더)...")
    # TODO: FactScore 라이브러리 임포트 및 평가 로직 구현
    # 1. results_data에서 'final_output' (생성된 텍스트) 리스트 추출
    # 2. results_data에서 'query' 또는 'topic' (주제) 리스트 추출
    # 3. FactScorer 객체 초기화
    # 4. scorer.get_score(topics=..., generations=...) 호출
    # 5. 결과 딕셔너리 반환
    logger.warning("FactScore 평가는 현재 구현되지 않았습니다. 공식 스크립트/라이브러리 연동 필요.")
    return {"status": "FactScore evaluation not implemented."}

def evaluate_truthfulqa(results_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """TruthfulQA 벤치마크 결과를 평가합니다."""
    logger.info("TruthfulQA 평가 시작 (플레이스홀더)...")
    # TODO: TruthfulQA 공식 평가 스크립트 임포트 또는 연동
    # 1. results_data에서 'final_output' (생성된 답변) 리스트 추출
    # 2. results_data에서 평가에 필요한 'correct_answers_truthfulqa' 등 필드 추출
    # 3. 공식 평가 함수 호출
    # 4. 결과 딕셔너리 반환
    logger.warning("TruthfulQA 평가는 현재 구현되지 않았습니다. 공식 스크립트 연동 필요.")
    return {"status": "TruthfulQA evaluation not implemented."}


# --- 메인 평가 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment results.")
    # ... (parser 인자 정의는 이전과 동일) ...
    parser.add_argument("--results_file", type=str, required=True, help="Path to the .jsonl results file.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"), help="Path to config file.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Specify dataset (e.g., multispanqa_dev). If None, inferred from path.")
    args = parser.parse_args()

    # --- 설정 로드 ---
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"설정 파일 로드 실패 ({args.config}): {e}")
        return

    # --- 결과 파일 로드 ---
    # ... (결과 파일 로드 로직 동일) ...
    results_data = load_jsonl(args.results_file)
    if not results_data:
         logging.error("결과 파일이 비어있음.")
         return
    logging.info(f"{args.results_file} 에서 {len(results_data)}개의 결과 로드.")
    
    # --- 데이터셋 이름 결정 ---
    dataset_name = args.dataset_name
    if not dataset_name:
        try:
            # ... (경로에서 데이터셋 이름 추론 로직) ...
            parts = args.results_file.replace("\\", "/").split('/')
            results_index = parts.index('results')
            dataset_name = parts[results_index + 2]
            logging.info(f"결과 파일 경로에서 데이터셋 이름 추론: {dataset_name}")
        except (ValueError, IndexError):
            logging.error("데이터셋 이름을 추론할 수 없습니다. --dataset_name 인자를 사용해주세요.")
            return

    # --- 벤치마크별 평가 함수 선택 및 실행 ---
    evaluation_func: Optional[Callable] = None
    name_lower = dataset_name.lower()
    
    if 'hallulens_longwiki' in name_lower:
        evaluation_func = evaluate_factscore
    elif 'hallulens_precisewikiqa' in name_lower: # PreciseWikiQA 추가
        evaluation_func = evaluate_qa_benchmark
    elif 'multispanqa' in name_lower: # MultiSpanQA도 동일한 함수 사용
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
