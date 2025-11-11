import argparse
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Callable
import pprint
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


try:
    from src.main_serc import (
        prompt_baseline, prompt_extract_facts_from_sentence,
        prompt_validate_one_fact_against_evidence,
        prompt_find_sentence, prompt_generate_correct_fact, prompt_rewrite_sentence
    )
    from src import programmatic_helpers as ph
    from src.utils import load_config, save_jsonl, get_timestamp
    from src.data_loader import load_dataset
    
    from src.prompts import (
        generate_sentence_group_question_prompt,
        VERIFICATION_ANSWER_TEMPLATE 
    )
    from src.model_wrappers import generate 
    
except ImportError:
    logging.error("ImportError: 'src' 폴더를 찾을 수 없습니다. PYTHONPATH를 확인하세요.")
    logging.error(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logging.error(f"sys.path: {sys.path}")
    sys.exit(1)


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _prompt_generate_question_for_sentence_group(fact_texts_list: List[str], model_name: str, config: dict) -> str:
    prompt = generate_sentence_group_question_prompt(fact_texts_list)
    question_params = {"temperature": 0.01, "max_new_tokens": 75}
    raw_response = generate(prompt, model_name, config, generation_params_override=question_params)
    clean_text = raw_response
    
    hallucination_tags = [
        "[SENTENCE]",    
        "[INSTRUCTION]",  
        "[ANSWER]",        
        "[REASON]",        
        "[VERIFICATION]", 
        "(Note:",        
        "The final answer is:"
    ]
    
    indices = []
    for tag in hallucination_tags:
        idx = clean_text.find(tag)
        if idx != -1:
            indices.append(idx)
    
    split_idx = min(indices) if indices else -1
    
    if split_idx != -1:
        clean_text = clean_text[:split_idx]

    question_mark_index = clean_text.rfind('?')
    if question_mark_index != -1:
        clean_text = clean_text[:question_mark_index + 1]
        
    return clean_text.strip().strip('"').strip("'")



def _prompt_get_verification_answer(question: str, model_name: str, config: dict) -> str:

    prompt = VERIFICATION_ANSWER_TEMPLATE.format(question=question)
    answer_params = {"temperature": 0.01, "max_new_tokens": 100}
    raw_response = generate(prompt, model_name, config, generation_params_override=answer_params)

    clean_text = raw_response

    hallucination_tags = [
        "[SENTENCE]",     # <-- 요청하신 태그
        "[INSTRUCTION]",  # <-- 요청하신 태그
        "[ANSWER]",       # (기존 '혼잣말' 태그)
        "[REASON]",       # (기존 '혼잣말' 태그)
        "[VERIFICATION]", # (기존 '혼잣말' 태그)
        "(Note:",        # (로그에서 발견된 태그)
        "The final answer is:" # (로그에서 발견된 태그)
    ]
    
    indices = []
    for tag in hallucination_tags:
        idx = clean_text.find(tag)
        if idx != -1:
            indices.append(idx)
    
    split_idx = min(indices) if indices else -1
    
    if split_idx != -1:
        clean_text = clean_text[:split_idx]
    clean_text = clean_text.split('\n')[0]

    return clean_text.strip().strip('"').strip("'")


def SERC_FactInSentence_Iterative(query: str, model_name: str, config: Dict[str, Any],
                                  t_max: Optional[int] = None,
                                  max_facts_per_group: Optional[int] = None
                                  ) -> Dict[str, Any]:

    # --- 하이퍼파라미터 설정 ---
    T_MAX = t_max if t_max is not None else config.get('default_t_max', 3)
    MAX_FACTS_PER_GROUP = max_facts_per_group if max_facts_per_group is not None else config.get('default_max_facts_per_group', 5)
    
    logging.info(f"--- SERC [Fact-in-Sentence] 실행 시작 --- Query: '{query[:50]}...'")
    logging.info(f"Model: {model_name}, T_max: {T_MAX}, Max_Facts_Per_Group: {MAX_FACTS_PER_GROUP}")

    history = {'query': query, 'model_name': model_name, 'params': {'t_max': T_MAX, 'method': 'fact-in-sentence', 'max_facts': MAX_FACTS_PER_GROUP}, 'cycles': []}

    # --- 1. 초기 답변 생성 ---
    logging.info("--- [1단계] 초기 답변 생성 ---")
    current_baseline = prompt_baseline(query, model_name, config)
    history['initial_baseline'] = current_baseline # 원본 저장
    logging.info(f"  초기 Baseline 생성 완료 (길이: {len(current_baseline)}자)")
    logging.info(f"  [1단계-전체 출력물] \n{current_baseline}")
    
    # [!!! USER REQUEST: 추가된 코드 !!!]
    # 1단계 응답의 마지막 문장이 불완전할 경우를 대비해 제거합니다.
    try:
        initial_sentences = ph.programmatic_split_into_sentences(current_baseline)
        if len(initial_sentences) > 1:
            # 최소 2문장 이상일 때만 마지막 문장을 검사
            last_sentence = initial_sentences[-1].strip()
            # 마침표, 물음표, 느낌표, 따옴표 등으로 끝나지 않으면 불완전한 문장으로 간주
            if last_sentence and not last_sentence.endswith(('.', '?', '!', '"', "'", "”", "’")):
                logging.warning(f"  [1단계-필터링] 마지막 문장이 불완전하여 제거합니다: '{last_sentence}'")
                # 마지막 문장을 제외하고 다시 합침
                filtered_sentences = initial_sentences[:-1]
                current_baseline = " ".join(filtered_sentences).strip()
                logging.info(f"  [1단계-필터링] 필터링된 Baseline (길이: {len(current_baseline)}자)")
            else:
                logging.info(f"  [1단계-필터링] 마지막 문장이 온전하여 그대로 사용합니다.")
        elif len(initial_sentences) == 1:
            logging.info(f"  [1단계-필터링] 문장이 1개이므로 필터링을 건너뜁니다.")
        else:
            logging.info(f"  [1단계-필터링] 문장이 없어 필터링을 건너뜁니다.")
    except Exception as e:
        logging.error(f"  [1단계-필터링] 마지막 문장 필터링 중 오류 발생: {e}", exc_info=True)
        # 오류 발생 시 current_baseline을 그대로 사용
    # [!!! 추가된 코드 종료 !!!]
    
    
    total_cycles_executed = 0

    # --- 4. 반복적 교정 루프 ---
    for t in range(1, T_MAX + 1):
        total_cycles_executed = t
        cycle_log: Dict[str, Any] = {'cycle': t, 'steps': {}, 'baseline_before_cycle': current_baseline}
        logging.info(f"\n--- [사이클 {t}/{T_MAX}] 교정 시작 ---")

        # --- 2. 사실 추출 (변수 노드 식별 및 이중 리스트 구조화) ---
        logging.info("  [2단계] 사실 추출 및 문장 그룹화 시작...")
        # `current_baseline`은 이제 1단계에서 필터링된 버전을 사용합니다.
        sentences = ph.programmatic_split_into_sentences(current_baseline)
        
        sentence_groups: List[Dict[str, Any]] = [] 
        all_facts: Dict[str, str] = {} 
        
        fact_id_counter = 1
        raw_extractions = []
        
        for s in sentences:
            if not s: continue
            
            # 1. 모델에서 원본 응답 받기
            raw_extracted_list_str = prompt_extract_facts_from_sentence(s, model_name, config)
            clean_text = raw_extracted_list_str
            
            # (2단계 파싱 로직)
            marker1 = "[SENTENCE]"
            idx1 = clean_text.find(marker1)
            marker2 = "[INSTRUCTION]"
            idx2 = clean_text.find(marker2)
            indices = [i for i in [idx1, idx2] if i != -1] # 2단계는 이 2개만
            split_idx = min(indices) if indices else -1
            
            if split_idx != -1:
                clean_text = clean_text[:split_idx] # 마커 앞부분까지만 사용
                
            clean_extracted_list_str = clean_text.strip()
            
            # 2. 로그에는 '원본' 응답을, 파서에는 '정제된' 응답을 전달
            raw_extractions.append({'sentence': s, 'extracted_str': raw_extracted_list_str}) # 로그용
            
            parsed_facts_list = ph.programmatic_parse_fact_list(clean_extracted_list_str) # <--- 'clean_' 변수 사용
            
            if parsed_facts_list:
                sentence_facts_map = {}
                for fact_text in parsed_facts_list:
                    fid = f"f{fact_id_counter}"
                    fact_text = fact_text.strip()
                    sentence_facts_map[fid] = fact_text
                    all_facts[fid] = fact_text # [전체 맵]에도 추가
                    fact_id_counter += 1
                
                # 문장과, 그 문장에 속한 사실 맵을 리스트에 추가
                if sentence_facts_map:
                    sentence_groups.append({
                        'sentence': s,
                        'facts': sentence_facts_map 
                    })

        cycle_log['steps']['2_fact_extraction'] = {'raw': raw_extractions, 'sentence_groups': sentence_groups, 'all_facts_map': all_facts.copy()}

        if not all_facts:
            logging.info("  [2단계] 추출된 사실 없음. 루프를 종료합니다.")
            history['termination_reason'] = 'no_facts_extracted'
            break
        logging.info(f"  [2단계] 총 {len(all_facts)}개 사실(변수 노드) / {len(sentence_groups)}개 문장(검사 노드) 식별.")
        
        # --- 3. [Fact-in-Sentence] 신드롬 생성 (수동 청크 적용) ---
        logging.info(f"  [3단계-FactInSentence] {len(sentence_groups)}개 문장 그룹 검증 시작...")
        
        # --- [사용자 검증 단계] ---
        print("\n" + "="*80)
        logging.info(f"--- [사용자 검증] ---")
        logging.info(f"사이클 {t}, 2단계(사실 추출) 완료. 총 {len(all_facts)}개의 사실이 추출되었습니다.")
        print("추출된 'sentence_groups' (이중 리스트)의 내용은 다음과 같습니다:")

        pprint.pprint(sentence_groups)

        print("="*80)

        user_input = ""
        while user_input not in ['y', 'n']:
            user_input = input(f"-> 총 {len(all_facts)}개의 사실로 3단계(검증)를 계속 진행하시겠습니까? (y/n): ").strip().lower()

        if user_input == 'n':
            logging.warning("사용자가 검증 단계에서 실행을 중단했습니다.")
            history['termination_reason'] = 'user_aborted_at_fact_verification'
            break 

        logging.info("사용자 확인 완료. 3단계(신드롬 생성)를 계속합니다...")
        # --- [검증 단계 끝] ---
        
        syndrome: Dict[str, Dict[str, str]] = {} 
        validation_details = []

        # 1. 문장 그룹(이중 리스트)을 순회
        for group in sentence_groups:
            sentence_text = group['sentence']
            facts_in_group = group['facts']

            if not facts_in_group:
                continue

            fact_items_list = list(facts_in_group.items()) 

            for i in range(0, len(fact_items_list), MAX_FACTS_PER_GROUP):
                chunk = fact_items_list[i : i + MAX_FACTS_PER_GROUP]
                
                fact_ids_chunk = [item[0] for item in chunk]
                fact_texts_chunk = [item[1] for item in chunk]

                logging.debug(f"    - 청크 검증: (문장: '{sentence_text[:30]}...', 사실 {i+1}~{i+len(chunk)})")

                # (Model Call) 3a. '그룹(문장)' 질문 생성
                q = _prompt_generate_question_for_sentence_group(
                    fact_texts_list=fact_texts_chunk, 
                    model_name=model_name, 
                    config=config
                )
                
                if q.strip().lower() == "none" or not q.strip():
                    logging.warning(f"    [경고] 그룹 질문 생성 실패. 건너뜁니다.")
                    validation_details.append({'group_context': 'N/A', 'status': 'question_failed'})
                    continue
                    
                # (Model Call) 3b. 검증 답변 생성
                verified_answer = _prompt_get_verification_answer(q, model_name, config)
                
                # (Model Call) 3c. 1:1 패리티 검사 (청크 내 N개 사실)
                for fid, ftext in chunk: 
                    is_contradictory = prompt_validate_one_fact_against_evidence(
                        ftext, verified_answer, model_name, config
                    )
                    
                    validation_details.append({
                        'fact_id': fid, 'fact_text': ftext,
                        'sentence': sentence_text, 'group_question': q, 
                        'verified_answer': verified_answer, 'result': is_contradictory
                    })

                    if is_contradictory == "[Yes]":
                        logging.info(f"    [!!! 신드롬 탐지 !!!] {fid}: {ftext}")
                        syndrome[fid] = {"fact_text": ftext, "evidence": verified_answer}

        cycle_log['steps']['3_syndrome_generation'] = validation_details

        # --- 4c. 수렴 확인 ---
        if not syndrome:
            logging.info(f"\n  [4c단계] 신드롬 없음. 사이클 {t}에서 수렴.")
            history['termination_reason'] = f'converged_at_cycle_{t}'
            break
        else:
             logging.info(f"\n  [4C단계] 총 {len(syndrome)}개의 오류 사실(신드롬) 탐지. 교정 시작.")

        # --- 5. 교정 ---
        logging.info("  [5단계] 분해된 교정 적용 시작...")
        facts_to_correct = syndrome
        final_response_snapshot = current_baseline 
        correction_log = []

        for fi, error_info in facts_to_correct.items():
            fi_text = error_info['fact_text']
            correction_item: Dict[str, Any] = {'fact_id': fi, 'original_fact': fi_text}
            logging.info(f"    - 오류 {fi} 교정 시도: '{fi_text[:100]}...'")

            # (Model Call) 5a. 탐색
            bad_sentence = prompt_find_sentence(final_response_snapshot, fi_text, model_name, config)
            bad_sentence = bad_sentence.strip() if bad_sentence else ""
            correction_item['found_sentence'] = bad_sentence
            if bad_sentence.lower() == "none" or not bad_sentence:
                logging.warning(f"    [경고] 오류 {fi} 원본 문장 찾기 실패. 교정 건너뜁니다.")
                correction_item['status'] = 'find_failed'
                correction_log.append(correction_item)
                continue

            # (Model Call) 5b. 사실 수정
            correct_fact_text = prompt_generate_correct_fact(fi_text, model_name, config)
            correction_item['corrected_fact'] = correct_fact_text
            if not correct_fact_text:
                logging.warning(f"    [경고] 오류 {fi} 수정된 팩트 생성 실패. 교정 건너뜁니다.")
                correction_item['status'] = 'correct_fact_failed'
                correction_log.append(correction_item)
                continue
            
            # (Model Call) 5c. 문장 재작성
            good_sentence = prompt_rewrite_sentence(bad_sentence, correct_fact_text, model_name, config)
            correction_item['rewritten_sentence'] = good_sentence
            if not good_sentence:
                logging.warning(f"    [경고] 오류 {fi} 문장 재작성 실패. 교정 건너뜁니다.")
                correction_item['status'] = 'rewrite_failed'
                correction_log.append(correction_item)
                continue
            
            # (Programmatic) 5d. 대체
            temp_snapshot = ph.programmatic_replace(final_response_snapshot, bad_sentence, good_sentence)
            if temp_snapshot == final_response_snapshot:
                 logging.warning(f"    [경고] 오류 {fi} 교정 위한 문장 대체 실패.")
                 correction_item['status'] = 'replace_failed'
            else:
                final_response_snapshot = temp_snapshot
                correction_item['status'] = 'corrected'
                logging.info(f"    - 오류 {fi} 교정 적용 완료.")
            correction_log.append(correction_item)

        cycle_log['steps']['5_correction'] = correction_log
        current_baseline = final_response_snapshot
        logging.info(f"  [5단계] 사이클 {t} 교정 적용 완료.")
        cycle_log['baseline_after_cycle'] = current_baseline
        history['cycles'].append(cycle_log)

    # --- 루프 종료 후 최종 결과 기록 ---
    history['final_baseline'] = current_baseline
    history['total_cycles_executed'] = total_cycles_executed
    if 'termination_reason' not in history:
        history['termination_reason'] = f'max_iterations_reached (T={T_MAX})'
    logging.info(f"--- SERC [Fact-in-Sentence] 실행 종료 (총 {total_cycles_executed} 사이클) ---")
    
    return history

# --- 단일 항목 처리 래퍼 ---
def run_single_item_wrapper(item: Dict[str, Any], model_name: str, config: Dict[str, Any], t_max: int) -> Dict[str, Any]:
    try:
        serc_history = SERC_FactInSentence_Iterative(
            query=item.get('question', item.get('query')),
            model_name=model_name,
            config=config,
            t_max=t_max,
            max_facts_per_group=config.get('default_max_facts_per_group', 5) # config에서 하이퍼파라미터 전달
        )
        method_result = {'serc_result': serc_history, 'final_output': serc_history.get('final_baseline', ''), 'status': 'success'}
    except Exception as e:
        logger.error(f"'{item.get('query')}' 처리 중 오류 발생 (Fact-in-Sentence): {e}", exc_info=False)
        method_result = {"error": f"Exception during processing: {e}", "status": "error"}

    output_item = {
        **item, 
        "method_result": method_result,
        "method_used": f"serc_fact_in_sentence_t{t_max}"
    }
    return output_item

# --- 메인 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Run SERC (Fact-in-Sentence) Experiment.")
    
    default_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to config file.")
    
    parser.add_argument("--model", type=str, required=True, help="Model name (defined in config).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (key in config data_paths).")
    parser.add_argument("--limit", type=int, default=None, help="Limit data points. Default: All")
    parser.add_argument("--output_dir", type=str, default="results/fact_in_sentence_iterative", help="Dir to save results.")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional output filename suffix.")
    
    parser.add_argument("--t_max", type=int, default=None, help="Override default T_max (runs iteratively up to this value).")

    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"설정 파일을 찾을 수 없습니다: {args.config}")
        return
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {e}")
        return
        
    T_MAX_TO_RUN = args.t_max if args.t_max is not None else config.get('default_t_max', 3)

    logging.info(f"--- SERC (Fact-in-Sentence) 실험 시작 ---")
    logging.info(f"Config: {args.config}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset} (Limit: {args.limit if args.limit is not None else 'All'})")
    logging.info(f"T_max: {T_MAX_TO_RUN}")

    dataset_config_key = args.dataset
    relative_path = config.get('data_paths', {}).get(dataset_config_key)
    if not relative_path:
         logger.error(f"Config 파일({args.config})의 'data_paths'에서 '{dataset_config_key}' 키를 찾을 수 없습니다.")
         return
    dataset_path = os.path.join(PROJECT_ROOT, relative_path)
    
    try:
        data = load_dataset(dataset_config_key, dataset_path)
    except FileNotFoundError:
        logger.error(f"데이터셋 경로를 찾을 수 없음: {dataset_path}")
        return
    except Exception as e:
        logger.error(f"데이터셋 로딩 중 오류 발생 ({dataset_path}). 종료합니다.", exc_info=True)
        return
    
    if args.limit and args.limit > 0:
        if args.limit < len(data): data = data[:args.limit]
        logging.info(f"데이터 {len(data)}개로 제한하여 실행.")
    else:
        logging.info(f"데이터셋 {len(data)}개 전체 사용.")
    if not data:
        logger.error("로드된 데이터가 없습니다. 실험을 중단합니다.")
        return
        
    if not any(m['name'] == args.model for m in config.get('models', [])):
         logger.error(f"오류: 모델 '{args.model}'이(가) 설정 파일 '{args.config}'에 정의되지 않았습니다.")
         return

    timestamp = get_timestamp()
    results_base_dir_abs = os.path.join(PROJECT_ROOT, args.output_dir)
    results_dir = os.path.join(results_base_dir_abs, args.model.replace('/', '_'), args.dataset)
    
    limit_str = f"_limit{args.limit}" if args.limit else ""
    suffix_str = f"_{args.output_suffix}" if args.output_suffix else ""
    output_filename = f"serc_fact_in_sentence_t{T_MAX_TO_RUN}{limit_str}{suffix_str}_{timestamp}.jsonl"
    output_path = os.path.join(results_dir, output_filename)
    logging.info(f"결과는 다음 경로에 저장됩니다: {output_path}")

    results = []
    from tqdm import tqdm
    for item in tqdm(data, desc=f"SERC (Fact-in-Sentence, T={T_MAX_TO_RUN})"):
        result_item = run_single_item_wrapper(item=item, model_name=args.model,
                                              config=config, t_max=T_MAX_TO_RUN)
        results.append(result_item)
    
    try:
        save_jsonl(results, output_path)
        logging.info(f"\n--- SERC (Fact-in-Sentence) 실험 완료. 총 {len(results)}개의 결과가 {output_path}에 저장되었습니다. ---")
    except Exception as e:
        logger.error(f"최종 결과 저장 실패: {e}", exc_info=True)

if __name__ == "__main__":
    main()