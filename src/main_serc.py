import argparse
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Callable
import pprint
import re 
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

try:
    from src import programmatic_helpers as ph
    from src.utils import load_config, save_jsonl, get_timestamp
    from src.data_loader import load_dataset
    
    from src import prompts 
    from src.prompts import (
        generate_sentence_group_question_prompt,
        VERIFICATION_ANSWER_TEMPLATE,     # [수정] RAG 버전 제거
        VALIDATE_EVIDENCE_TEMPLATE, 
        CORRECT_FACT_TEMPLATE,            # [수정] RAG 버전 제거
        RECOMPOSE_PROMPT_TEMPLATE,   
        BASELINE_PROMPT_TEMPLATE_PN,    
        EXTRACT_FACTS_TEMPLATE_PN,      
        FIND_SENTENCE_TEMPLATE,         
        REWRITE_SENTENCE_TEMPLATE,
        QUERY_ENTITY_EXTRACTOR_TEMPLATE   # [수정] main_subject 추출을 위해 유지
    )
    from src.model_wrappers import generate 
    
except ImportError:
    logging.error("--- ImportError Traceback (전체 오류 로그) ---")
    logging.error(traceback.format_exc())
    logging.error("ImportError: 'src' 폴더 내 모듈 임포트 실패. PYTHONPATH를 확인하세요.")
    logging.error(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logging.error(f"sys.path: {sys.path}")
    sys.exit(1)


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 헬퍼 함수들 ---

def prompt_baseline(query: str, model_name: str, config: dict) -> str:
    """Step 1: Generate Initial Response"""
    prompt = prompts.BASELINE_PROMPT_TEMPLATE_PN.format(query=query)
    return generate(prompt, model_name, config)

def prompt_extract_facts_from_sentence(sentence: str, model_name: str, config: dict, main_subject: str) -> str:
    """Step 2: Extract Facts from a Sentence"""
    prompt = prompts.EXTRACT_FACTS_TEMPLATE_PN.format(sentence=sentence, main_subject=main_subject)
    
    # 1. 모델 생성
    raw_response = generate(prompt, model_name, config)
    
    # 2. [ANSWER] 태그 기반 잘라내기 (후처리)
    clean_text = raw_response
    
    # 잘라낼 태그 목록 (대문자, 소문자, 변형 등)
    stop_tags = ["[ANSWER]", "[Answer]", "Answer:", "[answer]"]
    
    for tag in stop_tags:
        idx = clean_text.find(tag)
        if idx != -1:
            # 태그가 발견되면 그 앞부분만 취함
            clean_text = clean_text[:idx]
            # (선택사항) 디버깅용 로그
            # logging.debug(f"[Fact Extraction] '{tag}' 태그 감지. 뒷부분 제거함.")
            break # 가장 먼저 나오는 태그에서 자르고 루프 종료
            
    return clean_text.strip()

def _clean_model_output(raw_response: str) -> str:
    if not raw_response: return ""
    def _final_scrub(line: str) -> str:
        line = re.sub(r'#.*$', '', line).strip()
        line = re.sub(r'\[.*?\]$', '', line).strip()
        line = re.sub(r'END OF INSTRUCTION.*$', '', line, flags=re.IGNORECASE).strip()
        line = re.sub(r'Note:.*$', '', line, flags=re.IGNORECASE).strip()
        line = re.sub(r'//', '', line, flags=re.IGNORECASE).strip()
        return line.strip().strip('"').strip("'")
    answer_markers = [r'\[ANSWER\]', r'Answer:', r'\[FINAL ANSWER\]', r'\[Final Answer\]:']
    for marker_pattern in answer_markers:
        match = re.search(marker_pattern + r'(.*)', raw_response, re.DOTALL | re.IGNORECASE)
        if match:
            potential_answer_block = match.group(1).strip()
            for line in potential_answer_block.splitlines():
                clean_line = line.strip()
                if len(clean_line) > 5 and not clean_line.startswith(('#', '|', '`', '_', '?', '[')):
                    final_answer = _final_scrub(clean_line)
                    if final_answer:
                        logging.debug(f"[_clean_model_output] [ANSWER] 마커로 추출: '{final_answer}'")
                        return final_answer
    clean_text = raw_response
    patterns_to_remove = [ r'\[.*?\]', r'\(Note:.*?\)', r'\(This statement is TRUE\.\)', r'(Step \d+:|Note:|REASONING|JUSTIFICATION|EXPLANATION|\[REASON\]|\[RATING\])', r'^\s*#+.*$', r'```python.*$', r'```' ]
    for pattern in patterns_to_remove:
        clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE | re.MULTILINE)
    clean_text = re.sub(r'^[\s|?_*#-]*$', '', clean_text, flags=re.MULTILINE)
    lines = [line.strip() for line in clean_text.splitlines()]
    for line in lines:
        if len(line) > 5 and not line.startswith(('_', '?', '|', '#', '`')):
            final_answer = _final_scrub(line)
            if final_answer:
                logging.debug(f"[_clean_model_output] 쓰레기 필터링 후 첫 줄 추출: '{final_answer}'")
                return final_answer
    logging.warning(f"[_clean_model_output] 모델 출력이 쓰레기(garbage)라서 모두 필터링됨. 원본: '{raw_response[:100]}...'")
    return ""

def prompt_validate_one_fact_against_evidence(fact_text: str, evidence_text: str, model_name: str, config: dict) -> str:
    """ 2-값 논리([Yes]/[No])로 파싱 """
    prompt = prompts.VALIDATE_EVIDENCE_TEMPLATE.format(fact_text=fact_text, evidence_text=evidence_text)
    response = generate(prompt, model_name, config)
    cleaned_response = response.strip().lower() if response else ""
    
    # [Yes] = 모순 (신드롬 1)
    if cleaned_response.startswith("[yes]") or cleaned_response.startswith("yes"):
        return "[Yes]"
    
    # [No] = 모순 없음 (신드롬 0)
    elif cleaned_response.startswith("[no]") or cleaned_response.startswith("no"):
        return "[No]"
    else:
        logging.warning(f"Unexpected validation response: '{response}'. Defaulting to '[No]' (No Syndrome).")
        return "[No]" 

def prompt_find_sentence(current_baseline: str, fact_text: str, model_name: str, config: dict) -> str:
    prompt = prompts.FIND_SENTENCE_TEMPLATE.format(current_baseline=current_baseline, fact_text=fact_text)
    raw_response = generate(prompt, model_name, config)
    return _clean_model_output(raw_response)

def prompt_generate_correct_fact(fact_text: str, model_name: str, config: dict) -> str:
    """ [수정] RAG context 파라미터 제거, 비-RAG 프롬프트 사용 """
    prompt = prompts.CORRECT_FACT_TEMPLATE.format(fact_text=fact_text)
    raw_response = generate(prompt, model_name, config)
    return _clean_model_output(raw_response) 

def prompt_rewrite_sentence(bad_sentence: str, correct_fact_text: str, model_name: str, config: dict, main_subject: str) -> str:
    """ main_subject 전달 """
    prompt = prompts.REWRITE_SENTENCE_TEMPLATE.format(bad_sentence=bad_sentence, correct_fact_text=correct_fact_text, main_subject=main_subject)
    raw_response = generate(prompt, model_name, config)
    return _clean_model_output(raw_response) 

def _prompt_generate_question_for_sentence_group(fact_texts_list: List[str], model_name: str, config: dict) -> str:
    prompt = generate_sentence_group_question_prompt(fact_texts_list)
    question_params = {"temperature": 0.01, "max_new_tokens": 75}
    raw_response = generate(prompt, model_name, config, generation_params_override=question_params)
    clean_text = raw_response
    hallucination_tags = [ "[SENTENCE]", "[INSTRUCTION]", "[ANSWER]", "[REASON]", "[VERIFICATION]", "(Note:", "The final answer is:" ]
    indices = []
    for tag in hallucination_tags:
        idx = clean_text.find(tag)
        if idx != -1: indices.append(idx)
    split_idx = min(indices) if indices else -1
    if split_idx != -1: clean_text = clean_text[:split_idx]
    question_mark_index = clean_text.rfind('?')
    if question_mark_index != -1: clean_text = clean_text[:question_mark_index + 1]
    return clean_text.strip().strip('"').strip("'")


def _prompt_get_verification_answer(question: str, model_name: str, config: dict) -> str:
    """ [수정] RAG context 파라미터 제거, 비-RAG 프롬프트 사용 """
    prompt = VERIFICATION_ANSWER_TEMPLATE.format(question=question)
    answer_params = {"temperature": 0.01, "max_new_tokens": 100}
    raw_response = generate(prompt, model_name, config, generation_params_override=answer_params)
    clean_text = raw_response
    hallucination_tags = [ "[SENTENCE]", "[INSTRUCTION]", "[ANSWER]", "[REASON]", "[VERIFICATION]", "(Note:", "The final answer is:" ]
    indices = []
    for tag in hallucination_tags:
        idx = clean_text.find(tag)
        if idx != -1: indices.append(idx)
    split_idx = min(indices) if indices else -1
    if split_idx != -1: clean_text = clean_text[:split_idx]
    clean_text = clean_text.split('\n')[0]
    return clean_text.strip().strip('"').strip("'")

def prompt_recompose(query: str, final_facts_map: Dict[str, str], model_name: str, config: dict) -> str:
    """ [유지] Step 3.7: Final Recomposition (Junk 필터) """
    fact_texts = [f"- {text}" for text in final_facts_map.values() if text and len(text) > 5]
    if not fact_texts:
        logging.warning("[prompt_recompose] 재구성을 위한 유의미한 사실 목록이 없습니다.")
        return "N/A"
    fact_list_str = "\n".join(fact_texts)
    prompt = prompts.RECOMPOSE_PROMPT_TEMPLATE.format(query=query, fact_list_str=fact_list_str)
    raw_response = generate(prompt, model_name, config)
    return _clean_model_output(raw_response)

# --- [수정] 방화벽 헬퍼 (main_subject 추출용) ---

def _parse_entity_firewall_output(raw_response: str) -> str:
    # (RAG 버전의 파서를 그대로 사용)
    if not raw_response: 
        return ""
    raw_response = raw_response.strip()

    def _final_scrub(line: str) -> str:
        line = re.sub(r'#.*$', '', line).strip() 
        line = re.sub(r'//.*$', '', line).strip() 
        line = re.sub(r'\[.*?\]$', '', line).strip() 
        line = re.sub(r'\(Note:.*?\)$', '', line, flags=re.IGNORECASE).strip()
        line = re.sub(r'->.*$', '', line).strip()
        line = line.strip().strip('"').strip("'").strip('`').strip(':').strip() 
        return line

    lines = raw_response.splitlines()
    if not lines:
        logging.warning(f"[_parse_entity]파서: 빈 응답. 원본: '{raw_response[:100]}...'")
        return ""

    first_line = lines[0]
    paren_index = first_line.find(')')
    
    if paren_index != -1:
        answer = first_line[:paren_index + 1]
        clean_answer = _final_scrub(answer)
        
        junk_keywords = ('explanation:', 'answer key:', 'note:', 'reasoning:', 'step 1:', 
                         'identify the', 'the final answer is:', 'the single most dominant',
                         '[', '(', '{', '<')
                         
        if len(clean_answer) > 1 and not clean_answer.lower().startswith(junk_keywords):
            logging.debug(f"[_parse_entity_v43] '첫 줄 괄호' 기반 추출: '{clean_answer}'")
            return clean_answer

    clean_answer = _final_scrub(first_line)
    junk_keywords = ('explanation:', 'answer key:', 'note:', 'reasoning:', 'step 1:', 
                     'identify the', 'the final answer is:', 'the single most dominant',
                     '(', '{', '<')
                     
    if len(clean_answer) > 0 and not clean_answer.lower().startswith(junk_keywords):
         logging.debug(f"[_parse_entity_v43] '첫 줄 (괄호 없음)' Fallback 추출: '{clean_answer}'")
         return clean_answer

    logging.warning(f"[_parse_entity_v43] V43 파싱 실패. 원본: '{raw_response[:100]}...'")
    return ""

def prompt_extract_entity_desc(text: str, model_name: str, config: dict, is_query: bool = False) -> str:
    """ [수정] 쿼리에서만 개체 설명을 추출합니다. (is_query=False 로직 제거) """
    if is_query:
        prompt = prompts.QUERY_ENTITY_EXTRACTOR_TEMPLATE.format(query=text)
    else:
        # 비-RAG 모드에서는 baseline에서 개체를 추출할 필요가 없습니다.
        logging.warning("prompt_extract_entity_desc가 is_query=False로 호출되었으나 비-RAG 모드입니다.")
        return ""
    
    raw_response = generate(prompt, model_name, config)
    cleaned = _parse_entity_firewall_output(raw_response)
    return cleaned if cleaned.lower() != "none" else "" # "None" 문자열을 빈 문자열로 정규화


# --- [수정] RAG 관련 헬퍼 함수 모두 삭제 ---


def SERC_FactInSentence_Iterative(query: str, model_name: str, config: Dict[str, Any],
                                    t_max: Optional[int] = None,
                                    max_facts_per_group: Optional[int] = None
                                    ) -> Dict[str, Any]:

    # --- 하이퍼파라미터 설정 ---
    T_MAX = t_max if t_max is not None else config.get('default_t_max', 3)
    MAX_FACTS_PER_GROUP = max_facts_per_group if max_facts_per_group is not None else config.get('default_max_facts_per_group', 5)
    
    logging.info(f"--- SERC [Grounded Fact-in-Sentence] (Non-RAG) 실행 시작 --- Query: '{query[:50]}...'")
    logging.info(f"Model: {model_name}, T_max: {T_MAX}, Max_Facts_Per_Group: {MAX_FACTS_PER_GROUP}")

    history = {'query': query, 'model_name': model_name, 'params': {'t_max': T_MAX, 'method': 'fact-in-sentence-non-rag'}, 'cycles': []}

    # --- [삭제] RAG Retriever 초기화 삭제 ---

    # --- 1. 초기 답변 생성 ---
    logging.info("--- [1단계] 초기 답변 생성 ---")
    current_baseline = prompt_baseline(query, model_name, config)
    history['initial_baseline'] = current_baseline 
    logging.info(f"  초기 Baseline 생성 완료 (길이: {len(current_baseline)}자)")
    
    # ... (1단계 불완전한 문장 필터링 로직 ...)
    try:
        initial_sentences = ph.programmatic_split_into_sentences(current_baseline)
        if len(initial_sentences) > 1:
            last_sentence = initial_sentences[-1].strip()
            if last_sentence and not last_sentence.endswith(('.', '?', '!', '"', "'", "”", "’")):
                logging.warning(f"  [1단계-필터링] 마지막 문장이 불완전하여 제거합니다: '{last_sentence}'")
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

    
    # --- [수정] 1.5단계: 메인 주제(main_subject) 추출 ---
    logging.info("--- [1.5단계] 메인 주제 추출 시작 ---")
    
    main_subject = "" 
    
    try:
        # 1. 쿼리 분석
        query_desc = prompt_extract_entity_desc(query, model_name, config, is_query=True)
        logging.info(f"  [1.5a] 쿼리 개체(Query_Desc): '{query_desc}'")
        
        main_subject = query_desc.split('(', 1)[0].strip()
        
        if not main_subject: 
            logging.warning(f"  [1.5e] 메인 주제(main_subject)가 비어있음. 쿼리 자체를 Fallback으로 사용.")
            main_subject = query.split('(', 1)[0].strip()
            main_subject = main_subject.strip()

        logging.info(f" 메인 주제(main_subject) 확정: '{main_subject}'")

    except Exception as e:
        logging.error(f"  [1.5단계] 메인 주제 추출 중 오류 발생: {e}. 쿼리 자체를 Fallback으로 사용합니다.", exc_info=True)
        main_subject = query.split('(', 1)[0].strip()
        main_subject = main_subject.strip()

    
    total_cycles_executed = 0
    
    # --- 2. 반복적 교정 루프 ---
    for t in range(1, T_MAX + 1):
        total_cycles_executed = t
        cycle_log: Dict[str, Any] = {'cycle': t, 'steps': {}, 'baseline_before_cycle': current_baseline}
        logging.info(f"\n--- [사이클 {t}/{T_MAX}] 교정 시작 ---")

        # --- 2a. 사실 추출 (변수 노드 식별) ---
        logging.info("  [2단계] 사실 추출 및 문장 그룹화 시작...")
        sentences = ph.programmatic_split_into_sentences(current_baseline)
        
        sentence_groups: List[Dict[str, Any]] = [] 
        all_facts: Dict[str, str] = {} 
        fact_id_counter = 1
        raw_extractions = []
        
        for s in sentences:
            if not s: continue
            # [수정] main_subject 전달
            raw_extracted_list_str = prompt_extract_facts_from_sentence(s, model_name, config, main_subject=main_subject) 
            
            clean_text = raw_extracted_list_str
            marker1 = "[SENTENCE]"; idx1 = clean_text.find(marker1)
            marker2 = "[INSTRUCTION]"; idx2 = clean_text.find(marker2)
            indices = [i for i in [idx1, idx2] if i != -1]
            split_idx = min(indices) if indices else -1
            if split_idx != -1:
                clean_text = clean_text[:split_idx]
            clean_extracted_list_str = clean_text.strip()
            
            raw_extractions.append({'sentence': s, 'extracted_str': raw_extracted_list_str})
            parsed_facts_list = ph.programmatic_parse_fact_list(clean_extracted_list_str) 
            
            if parsed_facts_list:
                sentence_facts_map = {}
                for fact_text in parsed_facts_list:
                    fid = f"f{fact_id_counter}"
                    fact_text = fact_text.strip()
                    sentence_facts_map[fid] = fact_text
                    all_facts[fid] = fact_text 
                    fact_id_counter += 1
                
                if sentence_facts_map:
                    sentence_groups.append({ 'sentence': s, 'facts': sentence_facts_map })

        cycle_log['steps']['2_fact_extraction'] = {'raw': raw_extractions, 'sentence_groups': sentence_groups, 'all_facts_map': all_facts.copy()}

        if not all_facts:
            logging.info("  [2단계] 추출된 사실 없음. 루프를 종료합니다.")
            history['termination_reason'] = 'no_facts_extracted'
            break
        logging.info(f"  [2단계] 총 {len(all_facts)}개 사실(변수 노드) / {len(sentence_groups)}개 문장(검사 노드) 식별.")
        
        
        # --- 2b. [수정] 비-RAG 신드롬 생성 ---
        logging.info(f"  [3단계] {len(sentence_groups)}개 문장 그룹 검증 시작 (비-RAG)...")
        
        syndrome: Dict[str, Dict[str, Any]] = {} 
        validation_details = []

        for group in sentence_groups:
            sentence_text = group['sentence']
            facts_in_group = group['facts']
            if not facts_in_group: continue
            fact_items_list = list(facts_in_group.items()) 

            for i in range(0, len(fact_items_list), MAX_FACTS_PER_GROUP):
                chunk = fact_items_list[i : i + MAX_FACTS_PER_GROUP]
                fact_texts_chunk = [item[1] for item in chunk]
                logging.debug(f"    - 청크 검증: (문장: '{sentence_text[:30]}...', 사실 {i+1}~{i+len(chunk)})")

                # (3a. 질문 생성)
                q = _prompt_generate_question_for_sentence_group(
                    fact_texts_list=fact_texts_chunk, 
                    model_name=model_name, 
                    config=config
                )
                if q.strip().lower() == "none" or not q.strip(): continue
                    
                # [삭제] RAG 증거 검색 삭제
                
                # (3b. [수정] 내부 지식(Internal Knowledge) 기반 답변 생성)
                verified_answer = _prompt_get_verification_answer(
                    q, model_name, config
                )
                
                # (3c. 1:1 패리티 검사)
                for fid, ftext in chunk: 
                    validation_result = prompt_validate_one_fact_against_evidence(
                        ftext, verified_answer, model_name, config
                    )
                    
                    validation_details.append({
                        'fact_id': fid, 'fact_text': ftext,
                        'sentence': sentence_text, 'group_question': q, 
                        'verified_answer': verified_answer, 'result': validation_result
                        # [삭제] 'retrieved_docs' 삭제
                    })

                    if validation_result == "[Yes]": # [Yes] = 모순
                        logging.info(f"    [!!! 신드롬 탐지 !!!] {fid}: {ftext}")
                        syndrome[fid] = {
                            "fact_text": ftext, 
                            "evidence": verified_answer, # [수정] RAG 문서 대신 검증 답변을 증거로 사용
                            "original_sentence": sentence_text 
                        }

        cycle_log['steps']['3_syndrome_generation'] = validation_details

        # --- 2c. 수렴 확인 ---
        if not syndrome:
            logging.info(f"\n  [4c단계] 신드롬 없음. 사이클 {t}에서 수렴.")
            history['termination_reason'] = f'converged_at_cycle_{t}'
            break
        else:
             logging.info(f"\n  [4C단계] 총 {len(syndrome)}개의 오류 사실(신드롬) 탐지. 교정 시작.")

        # --- [수정] 2d. 비-RAG 교정 ---
        logging.info("  [5단계] 분해된 교정 적용 시작 (비-RAG)...")
        facts_to_correct = syndrome
        final_response_snapshot = current_baseline 
        correction_log = []

        for fi, error_info in facts_to_correct.items():
            fi_text = error_info['fact_text']
            correction_item: Dict[str, Any] = {'fact_id': fi, 'original_fact': fi_text}
            logging.info(f"    - 오류 {fi} 교정 시도: '{fi_text[:100]}...'")

            # (5a. 탐색)
            bad_sentence = error_info.get('original_sentence', '').strip()
            correction_item['found_sentence'] = bad_sentence
            
            if not bad_sentence:
                logging.warning(f"    [경고] 오류 {fi} 신드롬에 원본 문장(original_sentence) 없음. 교정 건너뜁니다.")
                correction_item['status'] = 'find_failed_no_sentence_in_syndrome'
                correction_log.append(correction_item)
                continue
            
            if bad_sentence not in final_response_snapshot:
                 logging.warning(f"    [경고] 오류 {fi}의 원본 문장이 현재 baseline에 없음. (이전 교정에서 덮어쓰인 듯 함). 교정 건너뜁니다.")
                 correction_item['status'] = 'find_failed_sentence_not_in_baseline'
                 correction_log.append(correction_item)
                 continue

            # (5b. [수정] 내부 지식 기반 사실 수정)
            correct_fact_text = prompt_generate_correct_fact(
                fi_text, model_name, config
            )
            correction_item['corrected_fact'] = correct_fact_text
            
            if not correct_fact_text or correct_fact_text.lower() == "[unknown]":
                logging.warning(f"    [경고] 내부 지식 기반 팩트 생성 실패 (Unknown 반환). 교정 건너뜁니다.")
                correction_item['status'] = 'correct_fact_failed_unknown'
                correction_log.append(correction_item)
                continue
            
            # (5c. 문장 재작성)
            good_sentence = prompt_rewrite_sentence(
                bad_sentence, correct_fact_text, model_name, config, main_subject=main_subject
            )
            correction_item['rewritten_sentence'] = good_sentence
            
            if not good_sentence:
                logging.warning(f"    [경고] 오류 {fi} 문장 재작성 실패 (Junk 생성). 교정 건너뜁니다.")
                correction_item['status'] = 'rewrite_failed'
                correction_log.append(correction_item)
                continue
            if bad_sentence == good_sentence:
                 logging.info(f"    - 오류 {fi} 교정 결과 원본과 동일. 변경 없음.")
                 correction_item['status'] = 'corrected_no_change'
                 correction_log.append(correction_item)
                 continue
            
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

    
    # --- [유지] 3.7. 최종 재구성 (Junk 필터링) ---
    logging.info(f"\n--- [3.7단계] 최종 재구성 시작 ---")
    dirty_final_baseline = current_baseline
    history['dirty_baseline_before_recomposition'] = current_baseline 
    logging.info(f"  [3.7a] 더러운 Baseline에서 최종 사실 목록 추출...")
    final_sentences = ph.programmatic_split_into_sentences(dirty_final_baseline)
    final_facts_map_for_recomposition: Dict[str, str] = {}
    fid_counter = 1
    
    for s in final_sentences:
        if not s: continue
        raw_extracted_list_str = prompt_extract_facts_from_sentence(s, model_name, config, main_subject=main_subject)
        
        # ... (이하 사실 추출 파싱 로직 원본 유지) ...
        clean_text = raw_extracted_list_str
        marker1 = "[SENTENCE]"; idx1 = clean_text.find(marker1)
        marker2 = "[INSTRUCTION]"; idx2 = clean_text.find(marker2)
        indices = [i for i in [idx1, idx2] if i != -1]
        split_idx = min(indices) if indices else -1
        if split_idx != -1:
            clean_text = clean_text[:split_idx]
        clean_extracted_list_str = clean_text.strip()
        parsed_facts_list = ph.programmatic_parse_fact_list(clean_extracted_list_str)
        
        for fact_text in parsed_facts_list:
            fid = f"reco_f{fid_counter}"
            fact_text = fact_text.strip()
            if fact_text: 
                final_facts_map_for_recomposition[fid] = fact_text
                fid_counter += 1

    if not final_facts_map_for_recomposition:
        logging.warning("  [3.7] 재구성을 위한 사실 맵이 없습니다. 교정된 Baseline을 그대로 반환합니다.")
        clean_final_baseline = dirty_final_baseline # Fallback
    else:
        logging.info(f"  [3.7b] 최종 추출된 {len(final_facts_map_for_recomposition)}개 사실 맵을 사용하여 재구성 시작...")
        
        clean_final_baseline = prompt_recompose(
            query=query,
            final_facts_map=final_facts_map_for_recomposition,
            model_name=model_name,
            config=config
        )
        
        if clean_final_baseline.strip().lower() == "n/a" or not clean_final_baseline.strip():
             logging.warning(f"  [3.7b] 재구성 실패 (N/A 반환). 교정본(Dirty)으로 대체합니다.")
             clean_final_baseline = dirty_final_baseline # Fallback
        else:
             logging.info("  [3.7b] 재구성 성공.")
    
    # --- 루프 종료 후 최종 결과 기록 ---
    history['final_baseline'] = clean_final_baseline 
    history['total_cycles_executed'] = total_cycles_executed
    if 'termination_reason' not in history:
        history['termination_reason'] = f'max_iterations_reached (T={T_MAX})'
    logging.info(f"--- SERC [Grounded Fact-in-Sentence] (Non-RAG) 실행 종료 (총 {total_cycles_executed} 사이클) ---")
    
    return history

# --- 단일 항목 처리 래퍼 ---
def run_single_item_wrapper(item: Dict[str, Any], model_name: str, config: Dict[str, Any], t_max: int) -> Dict[str, Any]:
    try:
        serc_history = SERC_FactInSentence_Iterative(
            query=item.get('question', item.get('query')),
            model_name=model_name,
            config=config,
            t_max=t_max,
            max_facts_per_group=config.get('default_max_facts_per_group', 5) 
        )
        method_result = {'serc_result': serc_history, 'final_output': serc_history.get('final_baseline', ''), 'status': 'success'}
    except Exception as e:
        logger.error(f"'{item.get('query')}' 처리 중 오류 발생 (Fact-in-Sentence): {e}", exc_info=False)
        method_result = {"error": f"Exception during processing: {e}", "status": "error"}

    output_item = {
        **item, 
        "method_result": method_result,
        "method_used": f"serc_fact_in_sentence_t{t_max}_non_rag" # [수정]
    }
    return output_item

# --- 메인 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Run SERC (Fact-in-Sentence) Experiment (Non-RAG).") # [수정]
    
    default_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to config file.")
    
    parser.add_argument("--model", type=str, required=True, help="Model name (defined in config).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (key in config data_paths).")
    parser.add_argument("--limit", type=int, default=None, help="Limit data points. Default: All")
    parser.add_argument("--output_dir", type=str, default="results/fact_in_sentence_iterative_non_rag", help="Dir to save results.") # [수정]
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

    logging.info(f"--- SERC (Fact-in-Sentence) [NON-RAG] 실험 시작 ---") # [수정]
    logging.info(f"Config: {args.config}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset} (Limit: {args.limit if args.limit is not None else 'All'})")
    logging.info(f"T_max: {T_MAX_TO_RUN}")

    # (데이터셋 로드 로직)
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
    output_filename = f"serc_fact_in_sentence_t{T_MAX_TO_RUN}_non_rag{limit_str}{suffix_str}_{timestamp}.jsonl" # [수정]
    output_path = os.path.join(results_dir, output_filename)
    logging.info(f"결과는 다음 경로에 저장됩니다: {output_path}")

    results = []
    from tqdm import tqdm
    for item in tqdm(data, desc=f"SERC (Fact-in-Sentence, T={T_MAX_TO_RUN}, Non-RAG)"): # [수정]
        result_item = run_single_item_wrapper(item=item, model_name=args.model,
                                               config=config, t_max=T_MAX_TO_RUN)
        results.append(result_item)
    
    try:
        save_jsonl(results, output_path)
        logging.info(f"\n--- SERC (Fact-in-Sentence) [NON-RAG] 실험 완료. 총 {len(results)}개의 결과가 {output_path}에 저장되었습니다. ---") # [수정]
    except Exception as e:
        logger.error(f"최종 결과 저장 실패: {e}", exc_info=True)

if __name__ == "__main__":
    main()