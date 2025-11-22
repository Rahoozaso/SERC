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
# src 폴더도 경로에 추가
sys.path.append(os.path.join(PROJECT_ROOT, "src")) 

try:
    from src.model_wrappers import generate
    from src import prompts
    from src.utils import load_config, save_jsonl, get_timestamp, token_tracker
    from src.data_loader import load_dataset
    from src.rag_retriever import RAGRetriever
    from src.prompts import VERIFICATION_ANSWER_TEMPLATE_RAG
    from src.main_serc import prompt_baseline 

except ImportError:
    logging.error("--- ImportError Traceback (전체 오류 로그) ---")
    logging.error(traceback.format_exc())
    logging.error("ImportError: 'src' 폴더 내 모듈 임포트 실패. PYTHONPATH를 확인하세요.")
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
    prompt = prompts.VERIFICATION_ANSWER_TEMPLATE_RAG.format( context=context, query=question)
    answer_params = {"temperature": 0.01, "max_new_tokens": 100}
    raw_response = generate(prompt, model_name, config, generation_params_override=answer_params)
    
    # 정제 로직
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
    
    # [FINAL REVISED RESPONSE] 마커 우선 탐색
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
                return final_answer
                
    return ""

# --- [3] CoVe-RAG 메인 실행 함수 ---

def run_cove_rag(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    CoVe (Chain-of-Verification)의 4단계를 실행합니다.
    """
    cove_history = {'query': query, 'model_name': model_name, 'params': {'method': 'cove-rag'}, 'steps': {}}
    
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
            retrieved_docs = retriever.retrieve(q)
            answer = _cove_get_rag_answer(q, retrieved_docs, model_name, config)
            verification_results.append({'question': q, 'answer': answer, 'retrieved_docs': retrieved_docs})
            logging.debug(f"      Q: {q}\n        A: {answer[:100]}...")
        
        cove_history['steps']['3_verification_results'] = verification_results
        logging.info(f"    CoVe Execution: {len(verification_results)}개 답변 완료.")

        # --- 4단계: 최종 답변 생성 ---
        logging.info("  [CoVe-RAG 4/4] 최종 답변 생성 중...")
        evidence_str = _format_qa_evidence(verification_results)
        revise_prompt = prompts.COVE_REVISE_PROMPT_TEMPLATE.format(
            query=query,
            baseline_response=initial_baseline,
            verification_evidence=evidence_str
        )
        final_output_raw = generate(revise_prompt, model_name, config)
        final_output_cleaned = _clean_model_output(final_output_raw) 
        
        cove_history['steps']['4_final_output_raw'] = final_output_raw
        cove_history['final_output'] = final_output_cleaned
        logging.info(f"    CoVe Final Output: {final_output_cleaned[:100]}...")

    except Exception as e:
        logging.error(f"CoVe-RAG 실행 중 오류 발생: {e}", exc_info=True)
        cove_history['error'] = str(e)
        cove_history['final_output'] = f"Error during CoVe-RAG: {e}"

    return cove_history

# --- 단일 항목 처리 래퍼 ---
def run_single_item_wrapper(item: Dict[str, Any], model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    token_tracker.reset()
    try:
        query = item.get('question', item.get('query'))
        cove_history = run_cove_rag(
            query=query,
            model_name=model_name,
            config=config
        )
        usage = token_tracker.get_usage()
        method_result = {
            'cove_result': cove_history, 
            'final_output': cove_history.get('final_output', ''), 
            'status': 'success',
            'token_usage': usage
        }
    except Exception as e:
        logger.error(f"'{query}' 처리 중 오류 발생 (CoVe-RAG): {e}", exc_info=False)
        method_result = {
            "error": f"Exception during processing: {e}", 
            "status": "error",
            "token_usage": token_tracker.get_usage()
        }
    output_item = {
        **item, 
        "method_result": method_result,
        "method_used": "cove_rag"
    }
    return output_item

# --- 메인 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Run CoVe (Chain-of-Verification) Experiment with RAG.")
    
    default_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to config file.")
    parser.add_argument("--model", type=str, required=True, help="Model name (defined in config).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (key in config data_paths).")
    parser.add_argument("--start", type=int, default=0, help="Start index of the dataset.")
    parser.add_argument("--end", type=int, default=None, help="End index of the dataset.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save results every N items.")
    parser.add_argument("--output_dir", type=str, default="results/cove_rag", help="Dir to save results.")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional output filename suffix.")

    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {e}")
        return
        
    logging.info(f"--- CoVe [RAG] 실험 시작 ---")
    logging.info(f"Config: {args.config}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset} (Range: {args.start} ~ {args.end})")

    # 데이터셋 로드
    dataset_config_key = args.dataset
    relative_path = config.get('data_paths', {}).get(dataset_config_key)
    if not relative_path:
         logger.error(f"Config 파일에서 '{dataset_config_key}' 키를 찾을 수 없습니다.")
         return
    dataset_path = os.path.join(PROJECT_ROOT, relative_path)
    
    try:
        data = load_dataset(dataset_config_key, dataset_path)
    except Exception as e:
        logger.error(f"데이터셋 로딩 중 오류 발생 ({dataset_path}). 종료합니다.", exc_info=True)
        return
    if not data:
        logger.error("로드된 데이터가 없습니다.")
        return

    total_data_len = len(data)
    end_idx = args.end if args.end is not None else total_data_len
    if args.start < 0: args.start = 0
    if end_idx > total_data_len: end_idx = total_data_len
    
    data = data[args.start : end_idx]
    logging.info(f"데이터 슬라이싱 완료: 총 {len(data)}개 항목 처리 예정.")

    timestamp = get_timestamp()
    results_base_dir_abs = os.path.join(PROJECT_ROOT, args.output_dir)
    results_dir = os.path.join(results_base_dir_abs, args.model.replace('/', '_'), args.dataset)
    os.makedirs(results_dir, exist_ok=True) 
    
    suffix_str = f"_{args.output_suffix}" if args.output_suffix else ""
    output_filename = f"cove_rag_{args.start}-{end_idx}{suffix_str}_{timestamp}.jsonl"
    output_path = os.path.join(results_dir, output_filename)
    logging.info(f"결과는 다음 경로에 저장됩니다: {output_path}")

    results = []
    for i, item in enumerate(tqdm(data, desc=f"CoVe (RAG)")):
        result_item = run_single_item_wrapper(item=item, model_name=args.model, config=config)
        results.append(result_item)
        
        if (i + 1) % args.save_interval == 0:
            save_jsonl(results, output_path)
    
    save_jsonl(results, output_path)
    logging.info(f"\n--- CoVe [RAG] 실험 완료. 총 {len(results)}개의 결과가 {output_path}에 저장되었습니다. ---")

if __name__ == "__main__":
    main()