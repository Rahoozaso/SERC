# serc_framework/main_serc.py
import logging
from typing import Dict, List, Optional, Any

# 다른 모듈에서 함수 임포트
from . import prompts
from . import programmatic_helpers as ph # ph로 줄여서 사용
from .model_wrappers import generate

# 로깅 설정 (기본 설정)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 모델 호출 헬퍼 함수 정의 ---

def prompt_baseline(query: str, model_name: str, config: dict) -> str:
    prompt = prompts.BASELINE_PROMPT_TEMPLATE.format(query=query)
    return generate(prompt, model_name, config)

def prompt_extract_facts_from_sentence(sentence: str, model_name: str, config: dict) -> str:
    prompt = prompts.EXTRACT_FACTS_TEMPLATE.format(sentence=sentence)
    return generate(prompt, model_name, config)

def prompt_tag_one_fact(fact_text: str, model_name: str, config: dict) -> str:
    prompt = prompts.TAG_ONE_FACT_TEMPLATE.format(fact_text=fact_text)
    # 모델 응답에서 불필요한 공백/줄바꿈 제거
    tag_response = generate(prompt, model_name, config)
    return tag_response.strip() if tag_response else "기타"

def prompt_generate_question_for_group(tag: str, fact_texts_list: List[str], model_name: str, config: dict) -> str:
    prompt = prompts.generate_group_question_prompt(tag, fact_texts_list)
    return generate(prompt, model_name, config)

def prompt_get_verification_answer(question: str, model_name: str, config: dict) -> str:
    prompt = prompts.VERIFICATION_ANSWER_TEMPLATE.format(question=question)
    return generate(prompt, model_name, config)

def prompt_validate_one_fact_against_evidence(fact_text: str, evidence_text: str, model_name: str, config: dict) -> str:
    prompt = prompts.VALIDATE_EVIDENCE_TEMPLATE.format(fact_text=fact_text, evidence_text=evidence_text)
    response = generate(prompt, model_name, config)
    cleaned_response = response.strip().lower() if response else ""
    if any(kw in cleaned_response for kw in ['[예]', '예', 'yes']):
        return "[예]"
    elif any(kw in cleaned_response for kw in ['[아니오]', '아니오', '아니요', 'no']):
        return "[아니오]"
    else:
        logging.warning(f"Unexpected validation response: '{response}'. Defaulting to '[아니오]'.")
        return "[아니오]" 

def prompt_find_sentence(current_baseline: str, fact_text: str, model_name: str, config: dict) -> str:
    prompt = prompts.FIND_SENTENCE_TEMPLATE.format(current_baseline=current_baseline, fact_text=fact_text)
    return generate(prompt, model_name, config)

def prompt_generate_correct_fact(fact_text: str, model_name: str, config: dict) -> str:
    prompt = prompts.CORRECT_FACT_TEMPLATE.format(fact_text=fact_text)
    return generate(prompt, model_name, config)

def prompt_rewrite_sentence(bad_sentence: str, correct_fact_text: str, model_name: str, config: dict) -> str:
    prompt = prompts.REWRITE_SENTENCE_TEMPLATE.format(bad_sentence=bad_sentence, correct_fact_text=correct_fact_text)
    return generate(prompt, model_name, config)


# --- 4. 메인 실행 함수 (SERC 프레임워크) ---
def SERC(query: str, model_name: str, config: Dict[str, Any],
         t_max: Optional[int] = None,
         max_facts_per_group: Optional[int] = None) -> Dict[str, Any]:
    # --- 하이퍼파라미터 설정 ---
    T_MAX = t_max if t_max is not None else config.get('default_t_max', 3)
    MAX_FACTS_PER_GROUP = max_facts_per_group if max_facts_per_group is not None else config.get('default_max_facts_per_group', 5)

    logging.info(f"--- SERC 실행 시작 --- Query: '{query[:50]}...'")
    logging.info(f"Model: {model_name}, T_max: {T_MAX}, Max_Facts_Per_Group: {MAX_FACTS_PER_GROUP}")

    # --- 결과 저장용 딕셔너리 ---
    history = {'query': query, 'model_name': model_name, 'params': {'t_max': T_MAX, 'max_facts': MAX_FACTS_PER_GROUP}, 'cycles': []}

    # --- 1. 초기화 (초기 답변 생성) ---
    logging.info("--- [1단계] 초기 답변 생성 ---")
    current_baseline = prompt_baseline(query, model_name, config)
    history['initial_baseline'] = current_baseline
    logging.info(f"  초기 Baseline 생성 완료 (길이: {len(current_baseline)}자)")
    logging.debug(f"Initial Baseline:\n{current_baseline}")

    total_cycles_executed = 0
    # 반복적 교정 루프
    for t in range(1, T_MAX + 1):
        total_cycles_executed = t
        cycle_log: Dict[str, Any] = {'cycle': t, 'steps': {}, 'baseline_before_cycle': current_baseline}
        logging.info(f"\n--- [사이클 {t}/{T_MAX}] 교정 시작 ---")

        #  사실 추출 
        logging.info("  [2단계] 사실 추출 시작...")
        sentences = ph.programmatic_split_into_sentences(current_baseline)
        facts: Dict[str, str] = {}
        fact_id_counter = 1
        raw_extractions = []
        for s_idx, s in enumerate(sentences):
            if not s: continue
            logging.debug(f"    문장 {s_idx+1}에서 사실 추출 시도: '{s[:100]}...'")
            extracted_list_str = prompt_extract_facts_from_sentence(s, model_name, config)
            raw_extractions.append({'sentence': s, 'extracted_str': extracted_list_str})
            parsed_facts = ph.programmatic_parse_fact_list(extracted_list_str)
            if parsed_facts:
                logging.debug(f"      -> 추출된 사실 {len(parsed_facts)}개")
                for fact_text in parsed_facts:
                    facts[f"f{fact_id_counter}"] = fact_text.strip()
                    fact_id_counter += 1
            else:
                 logging.debug(f"      -> 추출된 사실 없음")
        cycle_log['steps']['2_fact_extraction'] = {'raw': raw_extractions, 'parsed_facts': facts.copy()}

        if not facts:
            logging.info("  [2단계] 추출된 사실 없음. 루프를 종료합니다.")
            history['termination_reason'] = 'no_facts_extracted'
            break
        logging.info(f"  [2단계] 총 {len(facts)}개의 사실 추출 완료.")

        # 신드롬 생성
        logging.info("  [3단계] 신드롬 생성 시작...")
        step3_log: Dict[str, Any] = {}

        # 1:1 태깅
        logging.info(f"    [3a] 사실 {len(facts)}개 1:1 태깅 수행 중...")
        fact_tags: Dict[str, str] = {}
        for fi, fi_text in facts.items():
            tag_response = prompt_tag_one_fact(fi_text, model_name, config)
            fact_tags[fi] = tag_response
            logging.debug(f"      {fi}: '{fi_text[:50]}...' -> Tag: '{tag_response}'")
        step3_log['3a_tags'] = fact_tags.copy()

        # 그룹화
        fact_groups_raw = ph.programmatic_group_facts_by_tag(fact_tags)
        logging.info(f"    [3b-1] 원본 {len(fact_groups_raw)}개 그룹 생성됨: {list(fact_groups_raw.keys())}")
        step3_log['3b1_raw_groups'] = fact_groups_raw.copy()

        # 그룹 분할
        fact_groups = ph.programmatic_chunk_groups(fact_groups_raw, MAX_FACTS_PER_GROUP)
        logging.info(f"    [3b-2] 그룹 분할 적용 (Max={MAX_FACTS_PER_GROUP}). 최종 {len(fact_groups)}개 그룹: {list(fact_groups.keys())}")
        step3_log['3b2_chunked_groups'] = fact_groups.copy()

        syndrome: Dict[str, Dict[str, str]] = {} # {fact_id: {"fact_text":..., "evidence": ..., "validation": "[예]"}}
        logging.info(f"    [3c/3d] {len(fact_groups)}개 그룹 검증 루프 시작...")
        validation_details = []

        # 그룹별 검증 및 1:1 검사
        for tag, fact_ids_list in fact_groups.items():
            group_log: Dict[str, Any] = {'tag': tag, 'fact_ids': fact_ids_list}
            logging.info(f"      - '{tag}' 그룹 검증 (사실 {len(fact_ids_list)}개)")
            fact_texts_list = [facts[fi] for fi in fact_ids_list]
            q = prompt_generate_question_for_group(tag, fact_texts_list, model_name, config)
            group_log['question'] = q
            if q.strip().lower() == "없음" or not q.strip():
                logging.warning(f"[경고] '{tag}' 그룹 질문 생성 실패. 검증 건너<0xEB><0x9B><0x8D>니다.")
                group_log['status'] = 'question_generation_failed'
                validation_details.append(group_log)
                continue
            logging.debug(f"검증 질문: '{q}'")
            verified_answer = prompt_get_verification_answer(q, model_name, config)
            group_log['verified_answer'] = verified_answer
            logging.debug(f"        검증 답변: '{verified_answer[:100]}...'")
            group_log['validations'] = []

            # 1:1 모순 검증
            for fi in fact_ids_list:
                fi_text = facts[fi]
                is_contradictory = prompt_validate_one_fact_against_evidence(fi_text, verified_answer, model_name, config)
                validation_item = {'fact_id': fi, 'fact_text': fi_text, 'result': is_contradictory}
                group_log['validations'].append(validation_item)
                logging.debug(f"          {fi} vs 증거 -> {is_contradictory}")

                if is_contradictory == "[예]":
                    logging.info(f"      [!!! 신드롬 탐지 !!!] {fi}: {fi_text}")
                    syndrome[fi] = {"fact_text": fi_text, "evidence": verified_answer, "validation": "[예]"}

            group_log['status'] = 'completed'
            validation_details.append(group_log)

        step3_log['3c_3d_details'] = validation_details
        cycle_log['steps']['3_syndrome_generation'] = step3_log

        #  수렴 확인 
        if not syndrome:
            logging.info(f"\n  [4c단계] 신드롬 없음. 사이클 {t}에서 수렴.")
            history['termination_reason'] = f'converged_at_cycle_{t}'
            break # 반복 루프 종료
        else:
             logging.info(f"\n  [4c단계] 총 {len(syndrome)}개의 신드롬 탐지. 교정 시작.")

        #  교정
        logging.info("  [5단계] 분해된 교정 적용 시작...")
        facts_to_correct = syndrome
        final_response_snapshot = current_baseline 
        correction_log = []

        for fi, error_info in facts_to_correct.items():
            fi_text = error_info['fact_text']
            correction_item: Dict[str, Any] = {'fact_id': fi, 'original_fact': fi_text}
            logging.info(f"    - 오류 {fi} 교정 시도: '{fi_text[:100]}...'")

            # 탐색
            bad_sentence = prompt_find_sentence(final_response_snapshot, fi_text, model_name, config)
            bad_sentence = bad_sentence.strip() if bad_sentence else ""
            correction_item['found_sentence'] = bad_sentence
            if bad_sentence.lower() == "없음" or not bad_sentence:
                logging.warning(f"[경고] 오류 {fi}의 원본 문장 찾기 실패. 교정 건너<0xEB><0x9B><0x8D>니다.")
                correction_item['status'] = 'find_failed'
                correction_log.append(correction_item)
                continue
            logging.debug(f"원본 문장 찾음: '{bad_sentence[:100]}...'")

            # 사실 수정
            correct_fact_text = prompt_generate_correct_fact(fi_text, model_name, config)
            correction_item['corrected_fact'] = correct_fact_text
            logging.debug(f"      수정된 팩트: '{correct_fact_text[:100]}...'")
            if not correct_fact_text:
                logging.warning(f"    [경고] 오류 {fi}의 수정된 팩트 생성 실패. 교정 건너<0xEB><0x9B><0x8D>니다.")
                correction_item['status'] = 'correct_fact_failed'
                correction_log.append(correction_item)
                continue


            # 문장 재작성
            good_sentence = prompt_rewrite_sentence(bad_sentence, correct_fact_text, model_name, config)
            correction_item['rewritten_sentence'] = good_sentence
            if not good_sentence:
                logging.warning(f"[경고] 오류 {fi}의 문장 재작성 실패. 교정 건너<0xEB><0x9B><0x8D>니다.")
                correction_item['status'] = 'rewrite_failed'
                correction_log.append(correction_item)
                continue
            logging.debug(f"재작성된 문장: '{good_sentence[:100]}...'")


            # 대체
            temp_snapshot = ph.programmatic_replace(final_response_snapshot, bad_sentence, good_sentence)
            if temp_snapshot == final_response_snapshot:
                 logging.warning(f"    [경고] 오류 {fi} 교정 위한 문장 대체 실패 (원본 문장 못 찾음?).")
                 correction_item['status'] = 'replace_failed'
            else:
                 final_response_snapshot = temp_snapshot
                 correction_item['status'] = 'corrected'
                 logging.info(f"    - 오류 {fi} 교정 적용 완료.")
            correction_log.append(correction_item)

        cycle_log['steps']['5_correction'] = correction_log
        current_baseline = final_response_snapshot 
        logging.info(f"  [5단계] 사이클 {t} 교정 적용 완료 (길이: {len(current_baseline)}자).")
        logging.debug(f"Baseline after cycle {t}:\n{current_baseline}")
        cycle_log['baseline_after_cycle'] = current_baseline
        history['cycles'].append(cycle_log)

    # --- 루프 종료 후 최종 결과 기록 ---
    if total_cycles_executed == T_MAX and syndrome: # T_max 도달 시 종료 이유
        history['termination_reason'] = f'max_iterations_reached_with_errors'
        logging.info(f"최대 반복 횟수({T_MAX}) 도달. 아직 신드롬이 남아있을 수 있음.")
    elif not syndrome: # 정상 수렴 시 기록됨
         logging.info(f"총 {total_cycles_executed} 사이클 실행 후 수렴.")
    else: # T_max 전에 사실 추출 실패 등 다른 이유로 종료된 경우
        if 'termination_reason' not in history:
             history['termination_reason'] = f'stopped_at_cycle_{total_cycles_executed}_unexpectedly'
             logging.warning(f"예상치 못한 이유로 사이클 {total_cycles_executed}에서 중단됨.")


    history['final_baseline'] = current_baseline
    history['total_cycles_executed'] = total_cycles_executed
    logging.info(f"--- SERC 실행 종료 --- 최종 Baseline 길이: {len(current_baseline)}자")

    return history