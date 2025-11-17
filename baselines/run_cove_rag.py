import argparse
import os
import sys
import logging
import re
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import traceback

# --- [1] 프로젝트 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
# src 폴더도 경로에 추가 (다른 모듈 임포트를 위해)
sys.path.append(os.path.join(PROJECT_ROOT, "src")) 

try:
    from src.model_wrappers import generate
    from src import prompts
    from src.utils import load_config, save_jsonl, get_timestamp
    from src.data_loader import load_dataset

    # [수정] RAG 모듈 및 RAG용 헬퍼 임포트
    from src.rag_retriever import RAGRetriever
    from src.prompts import VERIFICATION_ANSWER_TEMPLATE_RAG
    # SERC의 Baseline 생성 함수 재사용
    from src.main_serc import prompt_baseline 

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

# --- [2] CoVe-RAG 헬퍼 함수 ---

def _parse_cove_questions(text: str) -> List[str]:
    """CoVe 계획 단계에서 생성된 질문 목록 문자열을 파싱합니다."""
    lines = text.strip().splitlines()
    questions = [re.sub(r"^\s*(\d+\.|Q\d:)\s*", "", line).strip() for line in lines]
    return [q for q in questions if q and q != ""] # 빈 줄 제거

def _format_qa_evidence(qa_list: List[Dict[str, str]]) -> str:
    """검증 Q&A 리스트를 최종 프롬프트에 넣을 문자열로 포맷팅합니다."""
    if not qa_list:
        return "검증 결과 없음."
    
    formatted_str = ""
    for i, qa in enumerate(qa_list, 1):
        formatted_str += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"
    return formatted_str.strip()

def _cove_get_rag_answer(question: str, context: str, model_name: str, config: dict) -> str:
    """CoVe 3단계: RAG로 검증 답변 생성"""
    prompt = prompts.VERIFICATION_ANSWER_TEMPLATE_RAG.format(question=question, context=context)
    answer_params = {"temperature": 0.01, "max_new_tokens": 100}
    raw_response = generate(prompt, model_name, config, generation_params_override=answer_params)
    
    # main_serc.py의 _prompt_get_verification_answer에서 가져온 정리 로직
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

def _clean_model_output(raw_response: str) -> str:
    """SERC의 _clean_model_output 재사용 (CoVe 4단계 정제용)"""
    if not raw_response: return ""
    def _final_scrub(line: str) -> str:
        line = re.sub(r'#.*$', '', line).strip()
        line = re.sub(r'\[.*?\]$', '', line).strip()
        line = re.sub(r'END OF INSTRUCTION.*$', '', line, flags=re.IGNORECASE).strip()
        line = re.sub(r'Note:.*$', '', line, flags=re.IGNORECASE).strip()
        line = re.sub(r'//', '', line, flags=re.IGNORECASE).strip()
        return line.strip().strip('"').strip("'")
    
    # [수정] CoVe 4단계 프롬프트에 맞춰 '[FINAL REVISED RESPONSE]' 마커 추가
    answer_markers = [r'\[FINAL REVISED RESPONSE\]', r'\[ANSWER\]', r'Answer:', r'\[FINAL ANSWER\]', r'\[Final Answer\]:']
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

# --- [3] CoVe-RAG 메인 실행 함수 ---

def run_cove_rag(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    CoVe (Chain-of-Verification)의 4단계를 실행합니다.
    [수정] 3단계 검증을 RAG를 사용하여 실행합니다.
    """
    
    cove_history = {'query': query, 'model_name': model_name, 'params': {'method': 'cove-rag'}, 'steps': {}}
    
    # [신규] RAG Retriever 초기화
    try:
        retriever = RAGRetriever(config=config)
    except Exception as e:
        logging.error(f"RAG Retriever (CoVe) 초기화 실패: {e}", exc_info=True)
        cove_history['error'] = f"RAG Retriever (CoVe) 초기화 실패: {e}"
        cove_history['final_output'] = "Error during CoVe initialization."
        return cove_history

    try:
        # --- 1단계: 초기 답변 생성 ---
        logging.info("  [CoVe-RAG 1/4] 초기 답변 생성 중...")
        # SERC의 prompt_baseline 재사용 (내부 지식으로 생성)
        initial_baseline = prompt_baseline(query, model_name, config)
        cove_history['steps']['1_initial_baseline'] = initial_baseline
        logging.info(f"    CoVe Baseline: {initial_baseline[:100]}...")

        # --- 2단계: 검증 계획 수립 ---
        logging.info("  [CoVe-RAG 2/4] 검증 계획 수립 중...")
        plan_prompt = prompts.COVE_PLAN_PROMPT_TEMPLATE.format(
            query=query,
            baseline_response=initial_baseline
        )
        plan_response = generate(plan_prompt, model_name, config)
        verification_questions = _parse_cove_questions(plan_response)
        cove_history['steps']['2_verification_plan'] = {
            'raw_response': plan_response,
            'parsed_questions': verification_questions
        }
        logging.info(f"    CoVe Plan: {len(verification_questions)}개 질문 생성됨.")

        # --- 3단계: 검증 실행 (RAG 사용) ---
        logging.info(f"  [CoVe-RAG 3/4] {len(verification_questions)}개 질문 RAG 검증 실행 중...")
        verification_results = []
        for q in verification_questions:
            # [수정] RAG 검색 수행
            retrieved_docs = retriever.retrieve(q)
            # [수정] RAG 기반 답변 생성 헬퍼 호출
            answer = _cove_get_rag_answer(q, retrieved_docs, model_name, config)
            
            verification_results.append({'question': q, 'answer': answer, 'retrieved_docs': retrieved_docs})
            logging.debug(f"      Q: {q}\n        A: {answer[:100]}...")
        
        cove_history['steps']['3_verification_results'] = verification_results
        logging.info(f"    CoVe Execution: {len(verification_results)}개 답변 완료.")

        # --- 4단계: 최종 답변 생성 (수정) ---
        logging.info("  [CoVe-RAG 4/4] 최종 답변 생성 중...")
        evidence_str = _format_qa_evidence(verification_results)
        revise_prompt = prompts.COVE_REVISE_PROMPT_TEMPLATE.format(
            query=query,
            baseline_response=initial_baseline,
            verification_evidence=evidence_str
        )
        final_output_raw = generate(revise_prompt, model_name, config)
        
        # [수정] _clean_model_output 적용 (SERC와 동일하게)
        final_output_cleaned = _clean_model_output(final_output_raw) 
        
        cove_history['steps']['4_final_output_raw'] = final_output_raw
        cove_history['final_output'] = final_output_cleaned # 최종 결과 키
        logging.info(f"    CoVe Final Output: {final_output_cleaned[:100]}...")

    except Exception as e:
        logging.error(f"CoVe-RAG 실행 중 오류 발생: {e}", exc_info=True)
        cove_history['error'] = str(e)
        cove_history['final_output'] = f"Error during CoVe-RAG: {e}"

    return cove_history

# --- 단일 항목 처리 래퍼 ---
def run_single_item_wrapper(item: Dict[str, Any], model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        query = item.get('question', item.get('query'))
        cove_history = run_cove_rag(
            query=query,
            model_name=model_name,
            config=config
        )
        method_result = {'cove_result': cove_history, 'final_output': cove_history.get('final_output', ''), 'status': 'success'}
    except Exception as e:
        logger.error(f"'{query}' 처리 중 오류 발생 (CoVe-RAG): {e}", exc_info=False)
        method_result = {"error": f"Exception during processing: {e}", "status": "error"}

    output_item = {
        **item, 
        "method_result": method_result,
        "method_used": "cove_rag" # [수정]
    }
    return output_item

# --- 메인 함수 (이 파일을 직접 실행할 경우) ---
def main():
    parser = argparse.ArgumentParser(description="Run CoVe (Chain-of-Verification) Experiment with RAG.") # [수정]
    
    default_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to config file.")
    
    parser.add_argument("--model", type=str, required=True, help="Model name (defined in config).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (key in config data_paths).")
    parser.add_argument("--limit", type=int, default=None, help="Limit data points. Default: All")
    parser.add_argument("--output_dir", type=str, default="results/cove_rag", help="Dir to save results.") # [수정]
    parser.add_argument("--output_suffix", type=str, default="", help="Optional output filename suffix.")

    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"설정 파일을 찾을 수 없습니다: {args.config}")
        return
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {e}")
        return
        
    logging.info(f"--- CoVe [RAG] 실험 시작 ---") # [수정]
    logging.info(f"Config: {args.config}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset} (Limit: {args.limit if args.limit is not None else 'All'})")

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
    output_filename = f"cove_rag{limit_str}{suffix_str}_{timestamp}.jsonl" # [수정]
    output_path = os.path.join(results_dir, output_filename)
    logging.info(f"결과는 다음 경로에 저장됩니다: {output_path}")

    results = []
    for item in tqdm(data, desc=f"CoVe (RAG)"): # [수정]
        result_item = run_single_item_wrapper(item=item, model_name=args.model, config=config)
        results.append(result_item)
    
    try:
        save_jsonl(results, output_path)
        logging.info(f"\n--- CoVe [RAG] 실험 완료. 총 {len(results)}개의 결과가 {output_path}에 저장되었습니다. ---") # [수정]
    except Exception as e:
        logger.error(f"최종 결과 저장 실패: {e}", exc_info=True)

if __name__ == "__main__":
    main()