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
sys.path.append(os.path.join(PROJECT_ROOT, "src")) 

try:
    from src.model_wrappers import generate
    from src import prompts
    from src.utils import load_config, save_jsonl, get_timestamp, token_tracker
    from src.data_loader import load_dataset
    from src.rag_retriever import RAGRetriever
    from src.main_serc import prompt_baseline 

except ImportError:
    logging.error("--- ImportError Traceback ---")
    logging.error(traceback.format_exc())
    sys.exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RE_EX_STEP1_TEMPLATE = """Your task is to decompose the text into simple sub-questions for checking factual accuracy of the text. 
Make sure to clear up any references.

Topic: {query}
Text: {initial_response}

Sub-Questions:"""

RE_EX_STEP2_TEMPLATE = """You will receive an initial response along with a prompt. Your goal is to refine and enhance this response, ensuring its factual accuracy. 
Check for any factually inaccurate information in the initial response.
Use the provided sub-questions and corresponding answers as key resources in this process.

Sub-questions and Answers:
{evidence}

Prompt: {query}
Initial Response: {initial_response}

Please explain the factual errors in the initial response.
If there are no factual errors, respond with "None".
If there are factual errors, explain each factual error.

Factual Errors:"""

RE_EX_STEP3_TEMPLATE = """You will receive an initial response along with a prompt. Your goal is to refine and enhance this response, ensuring its factual accuracy.
You will receive a list of factual errors in the initial response from the previous step. Use this explanation of each factual error as a key resource in this process.

Factual Errors:
{explanation}

Prompt: {query}
Initial Response: {initial_response}

Revised Response:"""

# --- [3] RE-EX 헬퍼 함수 ---

def _parse_re_ex_questions(text: str) -> List[str]:
    """RE-EX Step 1에서 생성된 질문 목록을 파싱합니다."""
    lines = text.strip().splitlines()
    questions = []
    for line in lines:
        # 숫자 리스트 제거 (1. 질문 -> 질문) 및 물음표 확인
        cleaned = re.sub(r"^\s*(\d+\.|Q\d:|Sub-question \d+:)\s*", "", line).strip()
        if cleaned and '?' in cleaned:
            questions.append(cleaned)
    return questions

def _format_evidence(qa_list: List[Dict[str, str]]) -> str:
    """검색된 증거(Evidence)를 프롬프트 포맷으로 변환합니다."""
    if not qa_list:
        return "No evidence found."
    
    formatted_str = ""
    for i, item in enumerate(qa_list, 1):
        # RAGRetriever가 반환한 검색 결과(Context)를 Answer로 사용
        formatted_str += f"[Sub-question {i}]: {item['question']}\n[Sub-answer {i}]: {item['evidence']}\n\n"
    return formatted_str.strip()

def _clean_re_ex_output(raw_response: str) -> str:
    """최종 답변 정제 (CoVe 정제 로직 재사용)"""
    if not raw_response: return ""
    
    # "Revised Response:" 이후의 텍스트만 추출 시도
    split_patterns = [r"Revised Response:", r"\[Revised Response\]", r"Output:"]
    for pattern in split_patterns:
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if match:
            raw_response = raw_response[match.end():].strip()
            break
            
    # 기존 정제 로직 (불필요한 태그 제거)
    clean_text = re.sub(r'\(Note:.*?\)', '', raw_response, flags=re.IGNORECASE)
    clean_text = clean_text.strip().strip('"').strip("'")
    return clean_text

# --- [4] RE-EX 메인 실행 함수 ---

def run_re_ex(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    RE-EX (Revising Explanations) 3단계 파이프라인 실행
    Step 1: Generate Questions & Retrieve Evidence
    Step 2: Explain Factual Errors
    Step 3: Revise Response
    """
    history = {'query': query, 'model_name': model_name, 'params': {'method': 're-ex'}, 'steps': {}}
    
    try:
        retriever = RAGRetriever(config=config)
    except Exception as e:
        logging.error(f"RAG Retriever 초기화 실패: {e}", exc_info=True)
        history['error'] = f"Retriever Init Error: {e}"
        return history

    try:
        # --- 0단계: 초기 답변 생성 (Baseline) ---
        logging.info("  [RE-EX 0/3] 초기 답변 생성 중...")
        initial_response = prompt_baseline(query, model_name, config)
        history['steps']['0_initial_response'] = initial_response
        logging.info(f"    Baseline Length: {len(initial_response)} chars")

        # --- 1단계: 질문 생성 및 증거 검색 (Generate & Retrieve) ---
        logging.info("  [RE-EX 1/3] 질문 생성 및 증거 검색 중...")
        
        # 1-1. 질문 생성 (Batch)
        step1_prompt = RE_EX_STEP1_TEMPLATE.format(query=query, initial_response=initial_response)
        questions_raw = generate(step1_prompt, model_name, config)
        questions = _parse_re_ex_questions(questions_raw)
        
        history['steps']['1_questions_raw'] = questions_raw
        history['steps']['1_parsed_questions'] = questions
        logging.info(f"    Generated {len(questions)} sub-questions.")

        # 1-2. 증거 검색 (Retrieval)
        evidence_list = []
        for q in questions:
            # RAGRetriever를 사용하여 문서 검색 (검색된 문맥 자체를 Answer로 간주)
            retrieved_docs = retriever.retrieve(q)
            evidence_list.append({'question': q, 'evidence': retrieved_docs})
        
        history['steps']['1_evidence_list'] = evidence_list

        # --- 2단계: 오류 설명 (Factual Error Explanation) ---
        logging.info("  [RE-EX 2/3] 오류 설명 생성 중...")
        evidence_str = _format_evidence(evidence_list)
        
        step2_prompt = RE_EX_STEP2_TEMPLATE.format(
            evidence=evidence_str,
            query=query,
            initial_response=initial_response
        )
        explanation = generate(step2_prompt, model_name, config)
        history['steps']['2_explanation'] = explanation
        logging.info(f"    Explanation: {explanation[:100]}...")

        # --- 3단계: 최종 수정 (Revision) ---
        if "None" in explanation or "no factual errors" in explanation.lower():
            logging.info("    오류 없음 감지 -> 원본 유지")
            final_output = initial_response
            history['steps']['3_revision'] = "Skipped (No errors)"
        else:
            logging.info("  [RE-EX 3/3] 최종 답변 수정 중...")
            step3_prompt = RE_EX_STEP3_TEMPLATE.format(
                explanation=explanation,
                query=query,
                initial_response=initial_response
            )
            final_output_raw = generate(step3_prompt, model_name, config)
            final_output = _clean_re_ex_output(final_output_raw)
            history['steps']['3_revision_raw'] = final_output_raw

        history['final_output'] = final_output
        logging.info(f"    RE-EX Final Output: {final_output[:100]}...")

    except Exception as e:
        logging.error(f"RE-EX 실행 중 오류 발생: {e}", exc_info=True)
        history['error'] = str(e)
        history['final_output'] = f"Error: {e}"

    return history

# --- 단일 항목 처리 래퍼 ---
def run_single_item_wrapper(item: Dict[str, Any], model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    token_tracker.reset()
    try:
        query = item.get('question', item.get('query'))
        re_ex_history = run_re_ex(
            query=query,
            model_name=model_name,
            config=config
        )
        usage = token_tracker.get_usage()
        method_result = {
            're_ex_result': re_ex_history, 
            'final_output': re_ex_history.get('final_output', ''), 
            'status': 'success',
            'token_usage': usage
        }
    except Exception as e:
        logger.error(f"'{query}' 처리 중 오류 발생 (RE-EX): {e}", exc_info=False)
        method_result = {
            "error": f"Exception: {e}", 
            "status": "error",
            "token_usage": token_tracker.get_usage()
        }
    output_item = {
        **item, 
        "method_result": method_result,
        "method_used": "re_ex"
    }
    return output_item

# --- 메인 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Run RE-EX (Revising Explanations) Experiment.")
    
    default_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to config file.")
    parser.add_argument("--model", type=str, required=True, help="Model name (defined in config).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--end", type=int, default=None, help="End index.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval.")
    parser.add_argument("--output_dir", type=str, default="results/re_ex", help="Output directory.")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix.")

    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Config 로드 실패: {e}")
        return
        
    logging.info(f"--- RE-EX 실험 시작 ---")
    logging.info(f"Dataset: {args.dataset} | Model: {args.model}")

    # 데이터셋 로드
    dataset_path = os.path.join(PROJECT_ROOT, config['data_paths'][args.dataset])
    data = load_dataset(args.dataset, dataset_path)
    
    if not data: return

    # 슬라이싱
    end_idx = args.end if args.end is not None else len(data)
    data = data[args.start : end_idx]

    # 경로 설정
    timestamp = get_timestamp()
    results_dir = os.path.join(PROJECT_ROOT, args.output_dir, args.model.replace('/', '_'), args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    
    output_filename = f"re_ex_{args.start}-{end_idx}_{args.output_suffix}_{timestamp}.jsonl"
    output_path = os.path.join(results_dir, output_filename)

    results = []
    for i, item in enumerate(tqdm(data, desc="RE-EX")):
        result_item = run_single_item_wrapper(item, args.model, config)
        results.append(result_item)
        
        if (i + 1) % args.save_interval == 0:
            save_jsonl(results, output_path)
    
    save_jsonl(results, output_path)
    logging.info(f"실험 완료. 결과 저장: {output_path}")

if __name__ == "__main__":
    main()