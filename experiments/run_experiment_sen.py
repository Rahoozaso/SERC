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
        VERIFICATION_ANSWER_TEMPLATE_RAG,
        VALIDATE_EVIDENCE_TEMPLATE, 
        CORRECT_FACT_TEMPLATE_RAG, 
        RECOMPOSE_PROMPT_TEMPLATE, 
        BASELINE_PROMPT_TEMPLATE_PN, 
        EXTRACT_FACTS_TEMPLATE_PN, 
        FIND_SENTENCE_TEMPLATE, 
        REWRITE_SENTENCE_TEMPLATE,
        QUERY_ENTITY_EXTRACTOR_TEMPLATE,
        BASELINE_ENTITY_EXTRACTOR_TEMPLATE,
        RAG_DOMINANT_ENTITY_TEMPLATE,
        ENTITY_CONSISTENCY_JUDGE_TEMPLATE,
        BASELINE_PROMPT_TEMPLATE_RAG_FIRST 
    )
    from src.model_wrappers import generate 
    
    # [신규] LangChain RAG Retriever 클래스 임포트
    from src.rag_retriever import RAGRetriever 
    
except ImportError:

    logging.error("--- ImportError Traceback (전체 오류 로그) ---")
    logging.error(traceback.format_exc()) # <-- 이것이 핵심입니다.
    logging.error("ImportError: 'src' 폴더 내 모듈 임포트 실패. PYTHONPATH를 확인하세요.")
    logging.error(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logging.error(f"sys.path: {sys.path}")
    sys.exit(1)


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [신규] 헬퍼 함수들을 이 파일로 이동/수정 ---

def prompt_baseline(query: str, model_name: str, config: dict) -> str:
    """Step 1: Generate Initial Response"""
    prompt = prompts.BASELINE_PROMPT_TEMPLATE_PN.format(query=query)
    return generate(prompt, model_name, config)

def prompt_extract_facts_from_sentence(sentence: str, model_name: str, config: dict, main_subject: str) -> str:
    """Step 2: Extract Facts from a Sentence"""
    prompt = prompts.EXTRACT_FACTS_TEMPLATE_PN.format(sentence=sentence, main_subject=main_subject)
    return generate(prompt, model_name, config)

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
    """ [수정] 2-값 논리([Yes]/[No])로 파싱 """
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

def prompt_generate_correct_fact(fact_text: str, model_name: str, config: dict, context: str) -> str:
    """ [유지] RAG 문맥(context)을 받음 """
    prompt = prompts.CORRECT_FACT_TEMPLATE_RAG.format(fact_text=fact_text, context=context)
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


def _prompt_get_verification_answer(question: str, model_name: str, config: dict, context: str) -> str:
    prompt = VERIFICATION_ANSWER_TEMPLATE_RAG.format(question=question, context=context)
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

# --- 개체 검증 방화벽 헬퍼 함수 ---
def _parse_entity_firewall_output(raw_response: str) -> str:

    if not raw_response: 
        return ""
    raw_response = raw_response.strip()

    def _final_scrub(line: str) -> str:
        """최종 라인 정리: 주석 및 앞뒤 공백/특수문자 제거"""
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
            logging.debug(f"[_parse_entity] '첫 줄 괄호' 기반 추출: '{clean_answer}'")
            return clean_answer

    clean_answer = _final_scrub(first_line)
    junk_keywords = ('explanation:', 'answer key:', 'note:', 'reasoning:', 'step 1:', 
                     'identify the', 'the final answer is:', 'the single most dominant',
                     '(', '{', '<')
                     
    if len(clean_answer) > 0 and not clean_answer.lower().startswith(junk_keywords):
         logging.debug(f"[_parse_entity] '첫 줄 (괄호 없음)' Fallback 추출: '{clean_answer}'")
         return clean_answer

    logging.warning(f"[_parse_entity] 파싱 실패. 원본: '{raw_response[:100]}...'")
    return ""

def prompt_extract_entity_desc(text: str, model_name: str, config: dict, is_query: bool = False) -> str:
    """ 쿼리 또는 베이스라인에서 개체 설명을 추출합니다."""
    if is_query:
        prompt = prompts.QUERY_ENTITY_EXTRACTOR_TEMPLATE.format(query=text)
    else:
        prompt = prompts.BASELINE_ENTITY_EXTRACTOR_TEMPLATE.format(baseline_text=text)
    
    raw_response = generate(prompt, model_name, config)
    cleaned = _parse_entity_firewall_output(raw_response)
    # [수정] (None)을 포함한 빈 문자열 반환을 위해 cleaned.lower() != "none" 제거
    return cleaned 

def prompt_extract_rag_desc(query: str, context: str, model_name: str, config: dict) -> str:
    """ RAG 문맥에서 지배적인 개체 설명을 추출합니다."""
    prompt = prompts.RAG_DOMINANT_ENTITY_TEMPLATE.format(query=query, context=context)
    return _parse_entity_firewall_output(generate(prompt, model_name, config))

def prompt_judge_entity_consistency(desc_a: str, desc_b: str, model_name: str, config: dict) -> bool:
    """ 
    두 개체 설명이 일치하는지 판단합니다.
    [Yes] (일치) -> True 반환
    [No] (불일치/충돌) -> False 반환
    """
    prompt = prompts.ENTITY_CONSISTENCY_JUDGE_TEMPLATE.format(desc_a=desc_a, desc_b=desc_b)
    raw_response = generate(prompt, model_name, config)
    
    cleaned = raw_response.strip().lower()
    
    # [수정] re.DOTALL 플래그를 추가하여 [judgment]\n[yes] 같은 줄바꿈을 처리합니다.
    match = re.search(r"\[judgment\][\s\n]*(\[yes\]|yes|\[no\]|no)", cleaned, re.IGNORECASE | re.DOTALL)
    
    if match:
        result = match.group(1).lower().strip() # .strip() 추가
        logging.info(f"  [1.5d] CoT 판단: '{result}'")
        
        if result == "[yes]" or result == "yes":
            return True  # 일치함 (Consistent)
        elif result == "[no]" or result == "no":
            return False # 불일치/충돌 (Contradictory)
        else:
            logging.warning(f"  [1.5d] CoT 판단이 정규식과 다름: '{result}'. 충돌로 간주.")
            return False # Default to False (충돌)
            
    else:
        # [수정] Fallback 로직도 .lower()된 cleaned를 기준으로 합니다.
        logging.warning(f"  [1.5d] CoT 파싱 실패 (Regex No Match). startswith로 Fallback: {cleaned[:50]}...")
        if cleaned.startswith("[yes]") or cleaned.startswith("yes"):
            return True
        if cleaned.startswith("[no]") or cleaned.startswith("no"):
            return False
        
        # [수정] Fallback에서도 [judgment] 태그 이후를 확인하도록 시도
        judgment_marker = "[judgment]"
        marker_pos = cleaned.find(judgment_marker)
        if marker_pos != -1:
            text_after_marker = cleaned[marker_pos + len(judgment_marker):].strip()
            logging.debug(f"  [1.5d] Fallback: [judgment] 이후 텍스트 '{text_after_marker[:20]}...'")
            if text_after_marker.startswith("[yes]") or text_after_marker.startswith("yes"):
                return True
            if text_after_marker.startswith("[no]") or text_after_marker.startswith("no"):
                return False

        logging.error(f"  [1.5d]판단 응답이 [Yes]/[No]가 아님. 충돌로 간주: {cleaned[:50]}...")
        return False # 최종 Fallback: 충돌로 간주 (False)

def prompt_regenerate_baseline_rag(query: str, context: str, model_name: str, config: dict) -> str:
    prompt = prompts.BASELINE_PROMPT_TEMPLATE_RAG_FIRST.format(context=context, query=query)
    return generate(prompt, model_name, config)

# --- 헬퍼 함수 정의 끝 ---


def SERC_FactInSentence_Iterative(query: str, model_name: str, config: Dict[str, Any],
                                    t_max: Optional[int] = None,
                                    max_facts_per_group: Optional[int] = None
                                    ) -> Dict[str, Any]:

    # --- 하이퍼파라미터 설정 ---
    T_MAX = t_max if t_max is not None else config.get('default_t_max', 3)
    MAX_FACTS_PER_GROUP = max_facts_per_group if max_facts_per_group is not None else config.get('default_max_facts_per_group', 5)
    
    logging.info(f"--- SERC [Grounded Fact-in-Sentence] 실행 시작 --- Query: '{query[:50]}...'")
    logging.info(f"Model: {model_name}, T_max: {T_MAX}, Max_Facts_Per_Group: {MAX_FACTS_PER_GROUP}")

    history = {'query': query, 'model_name': model_name, 'params': {'t_max': T_MAX, 'method': 'fact-in-sentence-rag'}, 'cycles': []}

    # --- RAG Retriever 초기화 ---
    try:
        retriever = RAGRetriever(config=config)
    except Exception as e:
        logging.error(f"RAG Retriever 초기화 실패: {e}. Grounded SERC를 실행할 수 없습니다.", exc_info=True)
        raise e

    # --- 1. 초기 답변 생성 ---
    logging.info("--- [1단계] 초기 답변 생성 ---")
    current_baseline = prompt_baseline(query, model_name, config)
    history['initial_baseline'] = current_baseline 
    logging.info(f"  초기 Baseline 생성 완료 (길이: {len(current_baseline)}자)")
    
    # ... (1단계 불완전한 문장 필터링 로직 ...)
    try:
        initial_sentences = ph.programmatic_split_into_sentences(current_baseline)
        if len(initial_sentences) > 1:
            last_sentence = initial_sentences[-1].strip()
            if last_sentence and not last_sentence.endswith(('.', '?', '!', '"', "'", "”", "’")):
                logging.warning(f"  [1단계-필터링] 마지막 문장이 불완전하여 제거합니다: '{last_sentence}'")
                filtered_sentences = initial_sentences[:-1]
                current_baseline = " ".join(filtered_sentences).strip()
                logging.info(f"  [1단계-필터링] 필터링된 Baseline (길이: {len(current_baseline)}자)")
            else:
                logging.info(f"  [1단계-필터링] 마지막 문장이 온전하여 그대로 사용합니다.")
        elif len(initial_sentences) == 1:
             logging.info(f"  [1단계-필터링] 문장이 1개이므로 필터링을 건너뜁니다.")
        else:
             logging.info(f"  [1단계-필터링] 문장이 없어 필터링을 건너뜁니다.")
    except Exception as e:
        logging.error(f"  [1단계-필터링] 마지막 문장 필터링 중 오류 발생: {e}", exc_info=True)

    
    # --- 1.5단계: 개체 검증 방화벽  ---
    logging.info("--- [1.5단계] 개체 검증 방화벽 시작 ---")
    
    main_subject = "" 
    rag_context_override = None 

    try:
        # 1. 쿼리 분석
        query_desc = prompt_extract_entity_desc(query, model_name, config, is_query=True)
        logging.info(f"  [1.5a] 쿼리 개체(Query_Desc): '{query_desc}'")

        # 2. 모델 분석
        model_desc = prompt_extract_entity_desc(current_baseline, model_name, config, is_query=False)
        logging.info(f"  [1.5b] 모델 개체(Model_Desc): '{model_desc}'")
        
        # 3. RAG 분석
        rag_context = retriever.retrieve(query) # RAG는 쿼리만으로 검색
        rag_desc = prompt_extract_rag_desc(query, rag_context, model_name, config)
        logging.info(f"  [1.5c] RAG 개체(RAG_Desc): '{rag_desc}'")

        # 4. 계층적 교정 결정
        # [수정] 쿼리 개체에서 (none) 확인 시 소문자 변환 및 공백 제거
        is_query_ambiguous = (not query_desc) or query_desc.lower().strip() == "(none)"

        if not is_query_ambiguous:
            # [Case C] 쿼리가 명시적일 때 (e.g., "Paris (Capital)")
            logging.info(f"  [1.5d] [Case C] 쿼리가 명시적({query_desc}). RAG({rag_desc})와 일관성 검사.")
            if not prompt_judge_entity_consistency(query_desc, rag_desc, model_name, config):
                logging.warning(f"  [1.5d] [Case C] 쿼리({query_desc})와 RAG({rag_desc}) 충돌. 쿼리를 존중(A안)합니다.")
                rag_context_override = f"Context: This query is specifically about {query_desc}."
            else:
                logging.info(f"  [1.5d] [Case C] 쿼리와 RAG 일치. 정상 진행.")
            main_subject = model_desc.split('(', 1)[0].strip()

        else:
            # [Case A/B] 쿼리가 모호할 때 (e.g., "Suthida (None)" or "")
            if not query_desc:
                logging.info(f"  [1.5d] [Case A/B] 쿼리가 모호함 (추출된 개체 없음). RAG 기반으로 진행.")
            else:
                logging.info(f"  [1.5d] [Case A/B] 쿼리가 모호함 ({query_desc} 반환). RAG 기반으로 진행.")

            if model_desc and rag_desc and not prompt_judge_entity_consistency(model_desc, rag_desc, model_name, config):
                logging.warning(f"  [1.5d] [Case B] 모델({model_desc})과 RAG({rag_desc}) 충돌. '적극적 개체 교정(B안)' 실행.")
                logging.info(f"  [1.5d] [Case B] 메인 주제를 RAG 개체({rag_desc})로 설정.")
                current_baseline = prompt_regenerate_baseline_rag(query, rag_context, model_name, config)
                logging.info(f"  [1.6] 베이스라인 재생성 완료 (길이: {len(current_baseline)}자)")
                main_subject = rag_desc.split('(', 1)[0].strip() # 메인 주제를 RAG로 설정
            else:
                if model_desc and rag_desc:
                    logging.info(f"  [1.5d] [Case A] 모델({model_desc})과 RAG({rag_desc}) 일치.")
                else:
                    logging.info(f"  [1.5d] [Case A] 모델/RAG 정보 부족. 비교 없이 진행.")
                logging.info(f"  [1.5d] [Case A] 메인 주제를 기존 모델 개체({model_desc})로 유지.")
                main_subject = model_desc.split('(', 1)[0].strip() # 메인 주제를 모델로 유지

        if not main_subject: 
            logging.warning(f"  [1.5e] 메인 주제(main_subject)가 비어있음 (e.g., Case A에서 model_desc가 비었음). Fallback 실행.") 
            main_subject = rag_desc.split('(', 1)[0].strip() if rag_desc else query.split('(', 1)[0].strip()
            main_subject = main_subject.strip()

        logging.info(f" 메인 주제(main_subject) 확정: '{main_subject}'")

    except Exception as e:
        logging.error(f"  [1.5단계] 개체 검증 방화벽 실행 중 오류 발생: {e}. 방화벽을 건너뛰고 진행합니다.", exc_info=True)
        main_subject = query.split('(', 1)[0].strip()
        main_subject = re.sub(r"Tell me a bio of", "", main_subject, flags=re.IGNORECASE).strip() # [수정] 쿼리 정제

    total_cycles_executed = 0
    
    # --- 2. 반복적 교정 루프 ---
    for t in range(1, T_MAX + 1):
        total_cycles_executed = t
        cycle_log: Dict[str, Any] = {'cycle': t, 'steps': {}, 'baseline_before_cycle': current_baseline}
        logging.info(f"\n--- [사이클 {t}/{T_MAX}] 교정 시작 ---")

        # --- 2a. 사실 추출 (변수 노드 식별) ---
        logging.info("  [2단계] 사실 추출 및 문장 그룹화 시작...")
        sentences = ph.programmatic_split_into_sentences(current_baseline)
        
        sentence_groups: List[Dict[str, Any]] = [] 
        all_facts: Dict[str, str] = {} 
        fact_id_counter = 1
        raw_extractions = []
        
        for s in sentences:
            if not s: continue
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
            logging.info("  [2단계] 추출된 사실 없음. 루프를 종료합니다.")
            history['termination_reason'] = 'no_facts_extracted'
            break
        logging.info(f"  [2단계] 총 {len(all_facts)}개 사실(변수 노드) / {len(sentence_groups)}개 문장(검사 노드) 식별.")
        
        
        # --- 2b. RAG 기반 신드롬 생성 ---
        logging.info(f"  [3단계-Grounded] {len(sentence_groups)}개 문장 그룹 검증 시작...")
        
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
                logging.debug(f"    - 청크 검증: (문장: '{sentence_text[:30]}...', 사실 {i+1}~{i+len(chunk)})")

                # (3a. 질문 생성)
                q = _prompt_generate_question_for_sentence_group(
                    fact_texts_list=fact_texts_chunk, 
                    model_name=model_name, 
                    config=config
                )
                if q.strip().lower() == "none" or not q.strip(): continue
                    
                # [수정] 3b-1. RAG 증거 검색
                if rag_context_override:
                    contextual_q = f"{q} ({rag_context_override})"
                    logging.debug(f"  문맥화된 RAG 쿼리: {contextual_q}")
                    retrieved_docs = retriever.retrieve(contextual_q)
                else:
                    retrieved_docs = retriever.retrieve(q) 
                
                # (3b-2. Grounded 답변 생성)
                verified_answer = _prompt_get_verification_answer(
                    q, model_name, config, context=retrieved_docs
                )
                
                # (3c. 1:1 패리티 검사)
                for fid, ftext in chunk: 
                    validation_result = prompt_validate_one_fact_against_evidence(
                        ftext, verified_answer, model_name, config
                    )
                    
                    validation_details.append({
                        'fact_id': fid, 'fact_text': ftext,
                        'sentence': sentence_text, 'group_question': q, 
                        'verified_answer': verified_answer, 'result': validation_result,
                        'retrieved_docs': retrieved_docs 
                    })

                    if validation_result == "[Yes]": # [Yes] = 모순
                        logging.info(f"    [!!! 신드롬 탐지 !!!] {fid}: {ftext}")
                        syndrome[fid] = {
                            "fact_text": ftext, 
                            "evidence_docs": retrieved_docs, 
                            "original_sentence": sentence_text 
                        }

        cycle_log['steps']['3_syndrome_generation'] = validation_details

        # --- 2c. 수렴 확인 ---
        if not syndrome:
            logging.info(f"\n  [4c단계] 신드롬 없음. 사이클 {t}에서 수렴.")
            history['termination_reason'] = f'converged_at_cycle_{t}'
            break
        else:
             logging.info(f"\n  [4C단계] 총 {len(syndrome)}개의 오류 사실(신드롬) 탐지. 교정 시작.")

        # --- [수정] 2d. RAG 기반 교정 (원본 V9 로직, Locate만 수정) ---
        logging.info("  [5단계-Grounded] 분해된 교정 적용 시작...")
        facts_to_correct = syndrome
        final_response_snapshot = current_baseline 
        correction_log = []

        for fi, error_info in facts_to_correct.items():
            fi_text = error_info['fact_text']
            correction_item: Dict[str, Any] = {'fact_id': fi, 'original_fact': fi_text}
            logging.info(f"    - 오류 {fi} 교정 시도: '{fi_text[:100]}...'")

            # [수정] 5a. 탐색 (Locate) - prompt_find_sentence 호출 제거
            bad_sentence = error_info.get('original_sentence', '').strip()
            correction_item['found_sentence'] = bad_sentence
            
            if not bad_sentence:
                logging.warning(f"    [경고] 오류 {fi} 신드롬에 원본 문장(original_sentence) 없음. 교정 건너뜁니다.")
                correction_item['status'] = 'find_failed_no_sentence_in_syndrome'
                correction_log.append(correction_item)
                continue
            
            if bad_sentence not in final_response_snapshot:
                 logging.warning(f"    [경고] 오류 {fi}의 원본 문장이 현재 baseline에 없음. (이전 교정에서 덮어쓰인 듯 함). 교정 건너뜁니다.")
                 correction_item['status'] = 'find_failed_sentence_not_in_baseline'
                 correction_log.append(correction_item)
                 continue

            # 5b. Grounded 사실 수정 (Correct Belief)
            context_docs = error_info.get("evidence_docs", "[No context provided]")
            correct_fact_text = prompt_generate_correct_fact(
                fi_text, model_name, config, context=context_docs
            )
            correction_item['corrected_fact'] = correct_fact_text
            
            if not correct_fact_text or correct_fact_text.lower() == "[unknown]":
                logging.warning(f"    [경고] RAG 기반 팩트 생성 실패 (Unknown 반환). 교정 건너뜁니다.")
                correction_item['status'] = 'correct_fact_failed_unknown'
                correction_log.append(correction_item)
                continue
            
            # 5c. 문장 재작성 (Rewrite / Propagate Belief)
            good_sentence = prompt_rewrite_sentence(
                bad_sentence, correct_fact_text, model_name, config, main_subject=main_subject
            )
            correction_item['rewritten_sentence'] = good_sentence
            
            if not good_sentence:
                logging.warning(f"    [경고] 오류 {fi} 문장 재작성 실패 (Junk 생성). 교정 건너뜁니다.")
                correction_item['status'] = 'rewrite_failed'
                correction_log.append(correction_item)
                continue
            if bad_sentence == good_sentence:
                 logging.info(f"    - 오류 {fi} 교정 결과 원본과 동일. 변경 없음.")
                 correction_item['status'] = 'corrected_no_change'
                 correction_log.append(correction_item)
                 continue
            
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
        current_baseline = final_response_snapshot
        logging.info(f"  [5단계] 사이클 {t} 교정 적용 완료.")
        cycle_log['baseline_after_cycle'] = current_baseline
        history['cycles'].append(cycle_log)

    
    # --- [유지] 3.7. 최종 재구성 (Junk 필터링) ---
    logging.info(f"\n--- [3.7단계] 최종 재구성 시작 ---")
    dirty_final_baseline = current_baseline
    history['dirty_baseline_before_recomposition'] = current_baseline 
    logging.info(f"  [3.7a] 더러운 Baseline에서 최종 사실 목록 추출...")
    final_sentences = ph.programmatic_split_into_sentences(dirty_final_baseline)
    final_facts_map_for_recomposition: Dict[str, str] = {}
    fid_counter = 1
    
    for s in final_sentences:
        if not s: continue
        raw_extracted_list_str = prompt_extract_facts_from_sentence(s, model_name, config, main_subject=main_subject)
        
    
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
        logging.warning("  [3.7] 재구성을 위한 사실 맵이 없습니다. 교정된 Baseline을 그대로 반환합니다.")
        clean_final_baseline = dirty_final_baseline # Fallback
    else:
        logging.info(f"  [3.7b] 최종 추출된 {len(final_facts_map_for_recomposition)}개 사실 맵을 사용하여 재구성 시작...")
        
        clean_final_baseline = prompt_recompose(
            query=query,
            final_facts_map=final_facts_map_for_recomposition,
            model_name=model_name,
            config=config
        )
        
        if clean_final_baseline.strip().lower() == "n/a" or not clean_final_baseline.strip():
             logging.warning(f"  [3.7b] 재구성 실패 (N/A 반환). 교정본(Dirty)으로 대체합니다.")
             clean_final_baseline = dirty_final_baseline # Fallback
        else:
             logging.info("  [3.7b] 재구성 성공.")
    
    # --- 루프 종료 후 최종 결과 기록 ---
    history['final_baseline'] = clean_final_baseline 
    history['total_cycles_executed'] = total_cycles_executed
    if 'termination_reason' not in history:
        history['termination_reason'] = f'max_iterations_reached (T={T_MAX})'
    logging.info(f"--- SERC [Grounded Fact-in-Sentence] 실행 종료 (총 {total_cycles_executed} 사이클) ---")
    
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
        "method_used": f"serc_fact_in_sentence_t{t_max}_rag" 
    }
    return output_item

# --- 메인 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Run SERC (Fact-in-Sentence) Experiment.")
    
    default_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to config file.")
    
    parser.add_argument("--model", type=str, required=True, help="Model name (defined in config).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (key in config data_paths).")
    
    # [수정] --limit을 --start와 --end로 변경
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive) of the dataset to process.")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) of the dataset to process. Default: all.")
    
    parser.add_argument("--output_dir", type=str, default="results/fact_in_sentence_iterative_rag", help="Dir to save results.") 
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

    logging.info(f"--- SERC (Fact-in-Sentence) [RAG-HYBRID] 실험 시작 ---") 
    logging.info(f"Config: {args.config}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset} (Slice: {args.start} to {args.end if args.end is not None else 'end'})")
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
    total_data_count = len(data)
    start_idx = args.start
    end_idx = args.end if args.end is not None else total_data_count
    
    # Python 슬라이싱은 범위를 벗어나도 안전하게 처리함
    data_slice = data[start_idx:end_idx]
    
    if len(data_slice) < total_data_count:
         logging.info(f"데이터 {len(data_slice)}개로 슬라이싱하여 실행 (인덱스 {start_idx}부터 {end_idx}까지).")
    else:
         logging.info(f"데이터셋 {total_data_count}개 전체 사용.")
    
    if not data_slice: # [수정] data -> data_slice
        logger.error("슬라이싱된 데이터가 없습니다. (start/end 인덱스 확인). 실험을 중단합니다.")
        return
        
    if not any(m['name'] == args.model for m in config.get('models', [])):
         logger.error(f"오류: 모델 '{args.model}'이(가) 설정 파일 '{args.config}'에 정의되지 않았습니다.")
         return

    timestamp = get_timestamp()
    results_base_dir_abs = os.path.join(PROJECT_ROOT, args.output_dir)
    results_dir = os.path.join(results_base_dir_abs, args.model.replace('/', '_'), args.dataset)
    
    # [수정] 출력 파일명에 슬라이스 정보 반영
    slice_str = ""
    if args.start != 0 or args.end is not None:
        start_str = args.start
        end_str = "end" if args.end is None else args.end
        slice_str = f"_slice{start_str}-{end_str}"
        
    suffix_str = f"_{args.output_suffix}" if args.output_suffix else ""
    output_filename = f"serc_fact_in_sentence_t{T_MAX_TO_RUN}_rag{slice_str}{suffix_str}_{timestamp}.jsonl" 
    output_path = os.path.join(results_dir, output_filename)
    logging.info(f"결과는 다음 경로에 저장됩니다: {output_path}")

    results = []
    from tqdm import tqdm
    # [수정] data -> data_slice
    for item in tqdm(data_slice, desc=f"SERC (Fact-in-Sentence, T={T_MAX_TO_RUN}, RAG)"): 
        result_item = run_single_item_wrapper(item=item, model_name=args.model,
                                               config=config, t_max=T_MAX_TO_RUN)
        results.append(result_item)
    
    try:
        save_jsonl(results, output_path)
        logging.info(f"\n--- SERC (Fact-in-Sentence) [RAG-HYBRID] 실험 완료. 총 {len(results)}개의 결과가 {output_path}에 저장되었습니다. ---") 
    except Exception as e:
        logger.error(f"최종 결과 저장 실패: {e}", exc_info=True)

if __name__ == "__main__":
    main()