import argparse
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Callable

# --- 프로젝트 루트 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- src 폴더 내 모듈 임포트 ---
# main_serc에서 필요한 헬퍼 함수들을 가져옵니다.
from src.main_serc import (
    prompt_baseline, prompt_extract_facts_from_sentence,
    prompt_get_verification_answer, prompt_validate_one_fact_against_evidence,
    prompt_find_sentence, prompt_generate_correct_fact, prompt_rewrite_sentence
)
# programmatic 헬퍼 함수 임포트
from src import programmatic_helpers as ph
# Dense 검증용 프롬프트 (src/prompts.py에 추가했다고 가정)
from src.prompts import GENERATE_QUESTION_FOR_ONE_FACT_TEMPLATE
# 유틸리티 함수 임포트
from src.utils import load_config, save_jsonl, get_timestamp
from src.data_loader import load_dataset
# model_wrappers 임포트 (generate 함수)
from src.model_wrappers import generate

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ablation용 모델 호출 헬퍼 ---

def prompt_generate_question_for_one_fact(fact_text: str, model_name: str, config: dict) -> str:
    """Ablation용: 사실 1개에 대한 검증 질문 생성"""
    prompt = GENERATE_QUESTION_FOR_ONE_FACT_TEMPLATE.format(fact_text=fact_text)
    return generate(prompt, model_name, config)


# --- [신규] SERC-Dense-Iterative 메인 함수 ---
def SERC_Dense_Iterative(query: str, model_name: str, config: Dict[str, Any],
                         t_max: Optional[int] = None
                         # (max_facts_per_group는 Dense 모드이므로 필요 없음)
                         ) -> Dict[str, Any]:

    # --- 하이퍼파라미터 설정 ---
    T_MAX = t_max if t_max is not None else config.get('default_t_max', 3)
    
    logging.info(f"--- SERC [Dense-Iterative] 실행 시작 --- Query: '{query[:50]}...'")
    logging.info(f"Model: {model_name}, T_max: {T_MAX}")

    # --- 결과 저장용 딕셔너리 ---
    history = {'query': query, 'model_name': model_name, 'params': {'t_max': T_MAX, 'method': 'dense-iterative'}, 'cycles': []}

    # --- 1. 초기 답변 생성 ---
    logging.info("--- [1단계] 초기 답변 생성 ---")
    current_baseline = prompt_baseline(query, model_name, config)
    history['initial_baseline'] = current_baseline
    logging.info(f"  초기 Baseline 생성 완료 (길이: {len(current_baseline)}자)")
    
    total_cycles_executed = 0
    syndrome: Dict[str, Dict[str, str]] = {} # 신드롬 변수 초기화

    # --- 4. 반복적 교정 루프 ---
    for t in range(1, T_MAX + 1):
        total_cycles_executed = t
        cycle_log: Dict[str, Any] = {'cycle': t, 'steps': {}, 'baseline_before_cycle': current_baseline}
        logging.info(f"\n--- [사이클 {t}/{T_MAX}] 교정 시작 ---")

        # --- 2. 사실 추출 ---
        logging.info("  [2단계] 사실 추출 시작...")
        sentences = ph.programmatic_split_into_sentences(current_baseline)
        facts: Dict[str, str] = {}
        fact_id_counter = 1
        raw_extractions = []
        for s_idx, s in enumerate(sentences):
            if not s: continue
            extracted_list_str = prompt_extract_facts_from_sentence(s, model_name, config)
            raw_extractions.append({'sentence': s, 'extracted_str': extracted_list_str})
            parsed_facts = ph.programmatic_parse_fact_list(extracted_list_str)
            for fact_text in parsed_facts:
                facts[f"f{fact_id_counter}"] = fact_text.strip()
                fact_id_counter += 1
        cycle_log['steps']['2_fact_extraction'] = {'raw': raw_extractions, 'parsed_facts': facts.copy()}

        if not facts:
            logging.info("  [2단계] 추출된 사실 없음. 루프를 종료합니다.")
            history['termination_reason'] = 'no_facts_extracted'
            break
        logging.info(f"  [2단계] 총 {len(facts)}개의 사실 추출 완료.")

        # --- 3. [DENSE] 신드롬 생성 ---
        logging.info(f"  [3단계-Dense] {len(facts)}개 사실 1:1 검증 시작...")
        syndrome = {} # 각 사이클마다 신드롬 초기화
        validation_details = []

        for fi, fi_text in facts.items():
            logging.debug(f"    - 사실 {fi} 검증: '{fi_text[:50]}...'")
            
            # (Model Call) 3a. 사실 1개짜리 질문 생성
            q = prompt_generate_question_for_one_fact(fi_text, model_name, config)
            if q.strip().lower() == "없음" or not q.strip():
                logging.warning(f"    [경고] {fi} 질문 생성 실패. 건너<0xEB><0x9B><0x8D>니다.")
                validation_details.append({'fact_id': fi, 'status': 'question_failed'})
                continue
                
            # (Model Call) 3b. 검증 답변 생성
            verified_answer = prompt_get_verification_answer(q, model_name, config)
            
            # (Model Call) 3c. 1:1 모순 검증
            is_contradictory = prompt_validate_one_fact_against_evidence(fi_text, verified_answer, model_name, config)
            
            validation_details.append({
                'fact_id': fi, 'fact_text': fi_text, 'question': q, 
                'verified_answer': verified_answer, 'result': is_contradictory
            })

            if is_contradictory == "[예]":
                logging.info(f"    [!!! 신드롬 탐지 !!!] {fi}: {fi_text}")
                syndrome[fi] = {"fact_text": fi_text, "evidence": verified_answer, "validation": "[예]"}

        cycle_log['steps']['3_syndrome_generation_dense'] = validation_details

        # --- 4c. 수렴 확인 ---
        if not syndrome:
            logging.info(f"\n  [4c단계] 신드롬 없음. 사이클 {t}에서 수렴.")
            history['termination_reason'] = f'converged_at_cycle_{t}'
            break # 반복 루프 종료
        else:
             logging.info(f"\n  [4c단계] 총 {len(syndrome)}개의 신드롬 탐지. 교정 시작.")

        # --- 5. 교정 ---
        logging.info("  [5단계] 분해된 교정 적용 시작...")
        facts_to_correct = syndrome
        final_response_snapshot = current_baseline 
        correction_log = []

        for fi, error_info in facts_to_correct.items():
            fi_text = error_info['fact_text']
            correction_item: Dict[str, Any] = {'fact_id': fi, 'original_fact': fi_text}
            logging.info(f"    - 오류 {fi} 교정 시도: '{fi_text[:100]}...'")

            # (Model Call) 5a. 탐색
            bad_sentence = prompt_find_sentence(final_response_snapshot, fi_text, model_name, config)
            bad_sentence = bad_sentence.strip() if bad_sentence else ""
            correction_item['found_sentence'] = bad_sentence
            if bad_sentence.lower() == "없음" or not bad_sentence:
                logging.warning(f"    [경고] 오류 {fi} 원본 문장 찾기 실패. 교정 건너<0xEB><0x9B><0x8D>니다.")
                correction_item['status'] = 'find_failed'
                correction_log.append(correction_item)
                continue

            # (Model Call) 5b. 사실 수정
            correct_fact_text = prompt_generate_correct_fact(fi_text, model_name, config)
            correction_item['corrected_fact'] = correct_fact_text
            if not correct_fact_text:
                logging.warning(f"    [경고] 오류 {fi} 수정된 팩트 생성 실패. 교정 건너<0xEB><0x9B><0x8D>니다.")
                correction_item['status'] = 'correct_fact_failed'
                correction_log.append(correction_item)
                continue
            
            # (Model Call) 5c. 문장 재작성
            good_sentence = prompt_rewrite_sentence(bad_sentence, correct_fact_text, model_name, config)
            correction_item['rewritten_sentence'] = good_sentence
            if not good_sentence:
                logging.warning(f"    [경고] 오류 {fi} 문장 재작성 실패. 교정 건너<0xEB><0x9B><0x8D>니다.")
                correction_item['status'] = 'rewrite_failed'
                correction_log.append(correction_item)
                continue
            
            # (Programmatic) 5d. 대체
            temp_snapshot = ph.programmatic_replace(final_response_snapshot, bad_sentence, good_sentence)
            if temp_snapshot == final_response_snapshot:
                 logging.warning(f"    [경고] 오류 {fi} 교정 위한 문장 대체 실패.")
                 correction_item['status'] = 'replace_failed'
            else:
                 final_response_snapshot = temp_snapshot
                 correction_item['status'] = 'corrected'
                 logging.info(f"    - 오류 {fi} 교정 적용 완료.")
            correction_log.append(correction_item)

        cycle_log['steps']['5_correction'] = correction_log
        current_baseline = final_response_snapshot # 다음 루프를 위해 갱신
        logging.info(f"  [5단계] 사이클 {t} 교정 적용 완료.")
        cycle_log['baseline_after_cycle'] = current_baseline
        history['cycles'].append(cycle_log) # 이번 사이클 기록 저장

    # --- 루프 종료 후 최종 결과 기록 ---
    history['final_baseline'] = current_baseline
    history['total_cycles_executed'] = total_cycles_executed
    if 'termination_reason' not in history:
        history['termination_reason'] = f'max_iterations_reached (T={T_MAX})'
    logging.info(f"--- SERC [Dense-Iterative] 실행 종료 (총 {total_cycles_executed} 사이클) ---")
    
    return history

# --- 단일 항목 처리 래퍼 ---
def run_single_item_wrapper(item: Dict[str, Any], model_name: str, config: Dict[str, Any], t_max: int) -> Dict[str, Any]:
    """SERC_Dense_Iterative를 호출하고 결과 형식을 맞추기 위한 래퍼 함수"""
    try:
        serc_history = SERC_Dense_Iterative(
            query=item.get('question', item.get('query')),
            model_name=model_name,
            config=config,
            t_max=t_max
        )
        method_result = {'serc_result': serc_history, 'final_output': serc_history.get('final_baseline', ''), 'status': 'success'}
    except Exception as e:
        logger.error(f"'{item.get('query')}' 처리 중 오류 발생 (Dense-Iterative): {e}", exc_info=False)
        method_result = {"error": f"Exception during processing: {e}", "status": "error"}

    output_item = {
        **item, 
        "method_result": method_result,
        "method_used": f"serc_dense_t{t_max}" # 메서드 이름에 t_max 포함
    }
    return output_item

# --- 메인 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Run SERC Ablation (Dense-Iterative) Experiment.")
    
    default_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to config file.")
    parser.add_argument("--model", type=str, required=True, help="Model name (defined in config).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (key in config data_paths).")
    parser.add_argument("--limit", type=int, default=None, help="Limit data points. Default: All")
    parser.add_argument("--output_dir", type=str, default="results/ablation_dense_iterative", help="Dir to save results.")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional output filename suffix.")
    
    # 이 스크립트는 T_max 값을 받아 반복 실행합니다.
    parser.add_argument("--t_max", type=int, default=None, help="Override default T_max (runs iteratively up to this value).")

    args = parser.parse_args()
    
    # T_max 값 결정 (인자 우선, 없으면 config 기본값)
    config_for_tmax = load_config(args.config) # T_max 기본값 로드를 위해 config 임시 로드
    T_MAX_TO_RUN = args.t_max if args.t_max is not None else config_for_tmax.get('default_t_max', 3)

    logging.info(f"--- Ablation (Dense-Iterative) 실험 시작 ---")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset} (Limit: {args.limit if args.limit is not None else 'All'})")
    logging.info(f"T_max: {T_MAX_TO_RUN}")

    # --- 설정 로드 ---
    config = config_for_tmax # 위에서 로드한 config 재사용

    # --- 데이터 로드 ---
    # ... (데이터 로드 로직은 이전과 동일) ...
    dataset_config_key = args.dataset
    relative_path = config.get('data_paths', {}).get(dataset_config_key)
    if not relative_path:
         logger.error(f"Config 파일({args.config})의 'data_paths'에서 '{dataset_config_key}' 키를 찾을 수 없습니다.")
         return
    dataset_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.exists(dataset_path):
        logger.error(f"데이터셋 경로를 찾을 수 없음: {dataset_path}")
        return
    try:
        data = load_dataset(dataset_config_key, dataset_path)
    except Exception as e:
        logger.error(f"데이터셋 로딩 중 오류 발생 ({dataset_path}). 종료합니다.", exc_info=True)
        return
    # 데이터 제한
    if args.limit and args.limit > 0:
        if args.limit < len(data): data = data[:args.limit]
        logging.info(f"데이터 {len(data)}개로 제한하여 실행.")
    else:
        logging.info(f"데이터셋 {len(data)}개 전체 사용.")
    if not data:
        logger.error("로드된 데이터가 없습니다. 실험을 중단합니다.")
        return
    # --- 모델 유효성 검사 ---
    if not any(m['name'] == args.model for m in config.get('models', [])):
         logger.error(f"오류: 모델 '{args.model}'이(가) 설정 파일에 정의되지 않았습니다.")
         return

    # --- 결과 저장 경로 설정 ---
    timestamp = get_timestamp()
    results_base_dir_abs = os.path.join(PROJECT_ROOT, args.output_dir)
    results_dir = os.path.join(results_base_dir_abs, args.model.replace('/', '_'), args.dataset)
    
    limit_str = f"_limit{args.limit}" if args.limit else ""
    suffix_str = f"_{args.output_suffix}" if args.output_suffix else ""
    output_filename = f"serc_dense_iterative_t{T_MAX_TO_RUN}{limit_str}{suffix_str}_{timestamp}.jsonl"
    output_path = os.path.join(results_dir, output_filename)
    logging.info(f"결과는 다음 경로에 저장됩니다: {output_path}")

    # --- 실험 실행 루프 ---
    results = []
    from tqdm import tqdm
    for item in tqdm(data, desc=f"Ablation (Dense-Iterative, T={T_MAX_TO_RUN})"):
        result_item = run_single_item_wrapper(item=item, model_name=args.model,
                                              config=config, t_max=T_MAX_TO_RUN)
        results.append(result_item)
    
    # --- 최종 결과 저장 ---
    try:
        save_jsonl(results, output_path)
        logging.info(f"\n--- Ablation (Dense-Iterative) 실험 완료. 총 {len(results)}개의 결과가 {output_path}에 저장되었습니다. ---")
    except Exception as e:
        logging.error(f"최종 결과 저장 실패: {e}")

if __name__ == "__main__":
    main()

