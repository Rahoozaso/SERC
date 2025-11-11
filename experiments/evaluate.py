# experiments/evaluate.py
import argparse
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Callable

# --- 프로젝트 루트 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- [수정] src 모듈 임포트 ---
from src.utils import load_config, load_jsonl
# [신규] 평가를 위해 model_wrappers의 generate 함수와 prompts 임포트
from src.model_wrappers import generate
from src import prompts # 평가용 프롬프트 임포트

# --- [유지] FActScore 공식 평가 도구 임포트 ---
try:
    from factscore.factscorer import FactScorer
    logging.info("FactScorer 라이브러리 임포트 성공.")
except ImportError:
    logging.warning("FactScore 라이브러리를 찾을 수 없습니다. (pip install factscore 필요)")
    FactScorer = None

# 로깅 설정
logger = logging.getLogger(__name__)

# --- 벤치마크별 평가 함수 ---

def evaluate_factscore(results_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Longform Biographies 결과를 FactScore 공식 라이브러리로 평가합니다."""
    logger.info("FactScore 평가 시작 (공식 라이브러리 연동)...")
    if FactScorer is None:
        return {"error": "FactScore 라이브러리가 로드되지 않았습니다."}

    # API 키 로드 (FActScore가 ChatGPT 평가기 사용 시)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return {"error": "FactScore용 OPENAI_API_KEY 환경 변수 필요."}
    
    try:
        fs = FactScorer(openai_key=openai_key, model_name="retrieval+ChatGPT")
    except Exception as e:
        logger.error(f"FactScorer 초기화 실패: {e}", exc_info=True)
        return {"error": f"FactScorer initialization failed: {e}"}

    # 데이터 추출 (topics, generations)
    topics = [item.get("topic") for item in results_data if item.get("topic") and item.get("method_result", {}).get("final_output")]
    generations = [item.get("method_result", {}).get("final_output") for item in results_data if item.get("topic") and item.get("method_result", {}).get("final_output")]
    
    if not generations:
        return {"error": "FactScore 평가를 위한 유효한 (topic, generation) 쌍이 없습니다."}

    logger.info(f"FactScore 계산 실행... (샘플 {len(generations)}개, OpenAI API 호출로 시간 소요)")
    try:
        out_scores = fs.get_score(topics=topics, generations=generations, gamma=10)
        metrics = {
            "factscore": out_scores.get("score"),
            "factscore_init_no_penalty": out_scores.get("init_score"),
            "respond_ratio": out_scores.get("respond_ratio"),
            "avg_num_facts": out_scores.get("num_facts_per_response"),
            "evaluated_count": len(generations)
        }
        logger.info(f"FactScore 계산 완료: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"FactScore.get_score() 실행 중 오류: {e}", exc_info=True)
        return {"error": f"FactScore calculation error: {e}"}

# --- [수정됨] PreciseWikiQA 평가 함수 ---
def evaluate_qa_benchmark(results_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """PreciseWikiQA 결과를 'LLM-as-Judge'로 평가합니다."""
    logger.info("PreciseWikiQA 평가 시작 (LLM-as-Judge)...")
    
    evaluator_model = config.get('evaluator_model_name')
    if not evaluator_model:
        return {"error": "Config에 'evaluator_model_name'이(가) 지정되지 않았습니다."}
    logger.info(f"평가자 모델: {evaluator_model}")
    
    correct_count = 0
    evaluated_count = 0
    
    from tqdm import tqdm
    for item in tqdm(results_data, desc="Evaluating PreciseWikiQA (LLM-as-Judge)"):
        pred = item.get("method_result", {}).get("final_output")
        gold_list = item.get("answers", []) # data_loader가 'answers' 키로 저장
        query = item.get("query", item.get("question"))

        if pred is None or not gold_list or not query:
             logger.warning(f"예측, 정답, 또는 쿼리가 없어 평가 건너<0xEB><0x9B><0x8D>니다.")
             continue
        
        # LLM 심판관 프롬프트 생성
        prompt = prompts.JUDGE_QA_PROMPT_TEMPLATE.format(
            query=query,
            ground_truths_str="; ".join(gold_list), # 정답 리스트를 문자열로
            prediction=pred
        )
        
        try:
            eval_params = {"temperature": 0.0, "max_new_tokens": 10} # [Correct]/[Incorrect] 응답용
            judgment = generate(prompt, evaluator_model, config, eval_params)
            
            if "[Correct]" in judgment:
                correct_count += 1
            elif "[Incorrect]" not in judgment:
                 logger.warning(f"평가자 응답 불명확: {judgment} (Incorrect로 처리)")
            
            evaluated_count += 1
        except Exception as e:
            logger.error(f"LLM-as-Judge 호출 중 오류: {e}")
            continue

    accuracy_avg = (correct_count / evaluated_count) * 100 if evaluated_count > 0 else 0
    
    logger.info(f"QA 평가 완료: Accuracy(LLM-Judge)={accuracy_avg:.2f}% ({evaluated_count}개 항목)")
    return {"accuracy_llm_judge": accuracy_avg, "evaluated_count": evaluated_count}

# --- [수정됨] TruthfulQA 평가 함수 ---
def evaluate_truthfulqa(results_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """TruthfulQA 벤치마크 결과를 'LLM-as-Judge'로 평가합니다."""
    logger.info("TruthfulQA 평가 시작 (LLM-as-Judge)...")

    evaluator_model = config.get('evaluator_model_name')
    if not evaluator_model:
        return {"error": "Config에 'evaluator_model_name'이(가) 지정되지 않았습니다."}
    logger.info(f"평가자 모델: {evaluator_model}")
    
    truthful_count = 0
    evaluated_count = 0
    
    from tqdm import tqdm
    for item in tqdm(results_data, desc="Evaluating TruthfulQA (LLM-as-Judge)"):
        pred = item.get("method_result", {}).get("final_output")
        query = item.get("query", item.get("question"))
        # data_loader가 저장한 원본 정답/오답 문자열
        correct_answers_str = item.get('correct_answers_truthfulqa')
        incorrect_answers_str = item.get('incorrect_answers_truthfulqa')

        if pred is None or not query or correct_answers_str is None or incorrect_answers_str is None:
             logger.warning(f"예측, 쿼리, 또는 정답/오답 문자열이 없어 평가 건너<0xEB><0x9B><0x8D>니다.")
             continue
        
        # LLM 심판관 프롬프트 생성
        prompt = prompts.JUDGE_TRUTHFULQA_PROMPT_TEMPLATE.format(
            query=query,
            correct_answers_str=correct_answers_str,
            incorrect_answers_str=incorrect_answers_str,
            prediction=pred
        )
        
        try:
            eval_params = {"temperature": 0.0, "max_new_tokens": 10} # [Truthful]/[Not Truthful] 응답용
            judgment = generate(prompt, evaluator_model, config, eval_params)
            
            if "[Truthful]" in judgment:
                truthful_count += 1
            elif "[Not Truthful]" not in judgment:
                 logger.warning(f"평가자 응답 불명확: {judgment} (Not Truthful로 처리)")

            evaluated_count += 1
        except Exception as e:
            logger.error(f"LLM-as-Judge 호출 중 오류: {e}")
            continue

    accuracy_avg = (truthful_count / evaluated_count) * 100 if evaluated_count > 0 else 0
    
    logger.info(f"TruthfulQA 평가 완료: Accuracy(LLM-Judge)={accuracy_avg:.2f}% ({evaluated_count}개 항목)")
    return {"accuracy_llm_judge": accuracy_avg, "evaluated_count": evaluated_count}


# --- 메인 평가 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment results.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the .jsonl results file.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"), help="Path to config file.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Specify dataset (e.g., hallulens_precisewikiqa). If None, inferred from path.")
    args = parser.parse_args()

    # --- 설정 로드 (평가자 모델 이름 포함) ---
    try:
        config = load_config(args.config)
        # .env 파일 로드 (API 키 등)
        from dotenv import load_dotenv
        # 프로젝트 루트의 .env 파일 로드
        load_dotenv(os.path.join(PROJECT_ROOT, '.env')) 
    except Exception as e:
        logger.error(f"설정 파일 로드 실패 ({args.config}): {e}")
        return

    # --- 결과 파일 로드 ---
    if not os.path.exists(args.results_file):
        logger.error(f"결과 파일을 찾을 수 없습니다: {args.results_file}")
        return
    results_data = load_jsonl(args.results_file)
    if not results_data:
         logger.error("결과 파일이 비어있음.")
         return
    logging.info(f"{args.results_file} 에서 {len(results_data)}개의 결과 로드.")
    
    # --- 데이터셋 이름 결정 ---
    dataset_name = args.dataset_name
    if not dataset_name:
        try:
            parts = args.results_file.replace("\\", "/").split('/')
            results_index = parts.index('results')
            dataset_name = parts[results_index + 2] 
            logging.info(f"결과 파일 경로에서 데이터셋 이름 추론: {dataset_name}")
        except (ValueError, IndexError):
            logging.error("데이터셋 이름을 추론할 수 없습니다. --dataset_name 인자를 사용해주세요.")
            return

    # --- [수정됨] 벤치마크별 평가 함수 선택 ---
    evaluation_func: Optional[Callable] = None
    name_lower = dataset_name.lower()
    
    if 'longform_bio' in name_lower: # FactScore (라이브러리)
        evaluation_func = evaluate_factscore
    elif 'hallulens_precisewikiqa' in name_lower: # PreciseWikiQA (LLM-as-Judge)
        evaluation_func = evaluate_qa_benchmark 
    elif 'truthfulqa' in name_lower: # TruthfulQA (LLM-as-Judge)
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
        print(metrics) # 오류 메시지 등
    print("-----------------------------------------------------")

if __name__ == "__main__":
    main()