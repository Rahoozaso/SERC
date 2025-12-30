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
    # 로컬 테스트를 위해 src가 없을 경우를 대비한 Mock 객체 (실제 구동 시 제거 가능)
    # sys.exit(1) 

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [2] RARR Prompts (Based on Appendix D of the paper) ---

# [cite: 881] Figure 13: Query Generation Prompt
RARR_QUERY_GEN_TEMPLATE = """[web] I will check things you said and ask questions.

(1) You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It's called the nasal cycle.
To verify it,
a) I googled: Does your nose switch between nostrils?
b) I googled: How often does your nostrils switch?
c) I googled: What is nasal cycle?

(2) You said: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford's psychology building.
To verify it,
a) I googled: Where was Stanford Prison Experiment was conducted?

(3) You said: {text}
To verify it,"""

# [cite: 948] Figure 14: Agreement Model Prompt
RARR_AGREEMENT_TEMPLATE = """[web] I will check some things you said.

(1) You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It's called the nasal cycle.
I checked: How often do your nostrils switch?
I found this article: Although we don't usually notice it, during the nasal cycle one nostril becomes congested... On average, the congestion pattern switches about every 2 hours...
Your nose's switching time is about every 2 hours, not 45 minutes.
This disagrees with what you said.

(2) You said: The Little House books were written by Laura Ingalls Wilder. The books were published by HarperCollins.
I checked: Who published the Little House books?
I found this article: ...Written by Laura Ingalls Wilder and published by HarperCollins, these beloved books remain a favorite to this day.
The Little House books were published by HarperCollins.
This agrees with what you said.

(3) You said: {text}
I checked: {query}
I found this article: {evidence}
"""

# [cite: 1048] Figure 15: Edit Model Prompt
RARR_EDIT_TEMPLATE = """[web] I will fix some things you said.

(1) You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It's called the nasal cycle.
I checked: How often do your nostrils switch?
I found this article: ...On average, the congestion pattern switches about every 2 hours...
This suggests 45 minutes switch time in your statement is wrong.
My fix: Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It's called the nasal cycle.

(2) You said: {text}
I checked: {query}
I found this article: {evidence}
This suggests"""

# --- [3] RARR Helper Functions ---

def _parse_rarr_questions(text: str) -> List[str]:
    """RARR Step 1: 생성된 검색 쿼리 파싱"""
    lines = text.strip().splitlines()
    questions = []
    for line in lines:
        # a) I googled: ... 패턴 매칭
        match = re.search(r"^[a-z]\)\s*I googled:\s*(.*)", line, re.IGNORECASE)
        if match:
            questions.append(match.group(1).strip())
    # 매칭 실패 시 fallback (단순 라인 분리)
    if not questions:
        questions = [line.strip() for line in lines if '?' in line]
    return questions

def _check_agreement(model_output: str) -> bool:
    """RARR Step 2: Agreement 모델의 출력 분석"""
    if "disagrees with what you said" in model_output.lower():
        return False
    return True

def _parse_edit_output(model_output: str) -> str:
    """RARR Step 3: Edit 모델의 출력에서 수정된 텍스트 추출"""
    match = re.search(r"My fix:\s*(.*)", model_output, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return model_output.strip()

# --- [4] RARR Main Logic Implementation ---

def run_rarr(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    RARR (Retrofit Attribution using Research and Revision) Pipeline
    Paper Reference: arXiv:2210.08726v3
    """
    history = {'query': query, 'model_name': model_name, 'params': {'method': 'rarr'}, 'steps': {}}
    
    try:
        retriever = RAGRetriever(config=config)
    except Exception as e:
        logging.error(f"RAG Retriever Init Failed: {e}", exc_info=True)
        history['error'] = str(e)
        return history

    try:
        # --- Stage 0: Generation ---
        # Generate initial text x 
        logging.info("  [RARR 0/3] Generating initial output...")
        initial_response = prompt_baseline(query, model_name, config)
        current_response = initial_response 
        history['initial_response'] = initial_response

        # --- Stage 1: Research (Query Gen & Retrieval) ---
        # Generate queries q_1...q_N 
        logging.info("  [RARR 1/3] Researching (Query Gen & Retrieval)...")
        step1_prompt = RARR_QUERY_GEN_TEMPLATE.format(text=initial_response)
        questions_raw = generate(step1_prompt, model_name, config)
        questions = _parse_rarr_questions(questions_raw)
        
        history['generated_queries'] = questions
        
        # Retrieve evidence for each query [cite: 165]
        evidence_pairs = []
        for q in questions:
            retrieved_docs = retriever.retrieve(q) # Assuming this returns a string or list of strings
            # RARR uses top-1 or top-K evidence. Assuming retriever returns best snippet.
            evidence_pairs.append({'query': q, 'evidence': retrieved_docs})
        
        history['evidence_pairs'] = evidence_pairs

        # --- Stage 2 & 3: Revision (Agreement & Edit Loop) ---
        # Iterate through evidence and revise [cite: 176]
        logging.info("  [RARR 2/3 & 3/3] Agreement Check & Revision Loop...")
        
        revisions = []
        
        for i, pair in enumerate(evidence_pairs):
            q_text = pair['query']
            e_text = pair['evidence']
            
            if not e_text: continue

            # 2.1 Agreement Model 
            # Checks if current_response agrees with e_text regarding q_text
            agreement_prompt = RARR_AGREEMENT_TEMPLATE.format(
                text=current_response,
                query=q_text,
                evidence=e_text
            )
            agreement_output = generate(agreement_prompt, model_name, config)
            is_agreed = _check_agreement(agreement_output)
            
            step_record = {
                'step': i,
                'query': q_text,
                'agreement_output': agreement_output,
                'is_agreed': is_agreed
            }

            # 2.2 Edit Model (if disagreement detected) [cite: 183]
            if not is_agreed:
                logging.info(f"    Disagreement detected at step {i}. Revising...")
                edit_prompt = RARR_EDIT_TEMPLATE.format(
                    text=current_response,
                    query=q_text,
                    evidence=e_text
                )
                edit_output_raw = generate(edit_prompt, model_name, config)
                revised_text = _parse_edit_output(edit_output_raw)
                
                # Update current response for next iteration [cite: 178]
                current_response = revised_text
                
                step_record['edit_output_raw'] = edit_output_raw
                step_record['revised_text'] = revised_text
            else:
                step_record['action'] = 'No Edit'

            revisions.append(step_record)

        history['revisions'] = revisions
        history['final_output'] = current_response
        logging.info(f"  RARR Final Output: {current_response[:100]}...")

    except Exception as e:
        logging.error(f"Error during RARR execution: {e}", exc_info=True)
        history['error'] = str(e)
        history['final_output'] = f"Error: {e}"

    return history

# --- Wrapper & Main (Same as before but calling run_rarr) ---

def run_single_item_wrapper(item: Dict[str, Any], model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    token_tracker.reset()
    try:
        query = item.get('question', item.get('query'))
        rarr_history = run_rarr(
            query=query,
            model_name=model_name,
            config=config
        )
        usage = token_tracker.get_usage()
        method_result = {
            'rarr_result': rarr_history, 
            'final_output': rarr_history.get('final_output', ''), 
            'status': 'success',
            'token_usage': usage
        }
    except Exception as e:
        logger.error(f"'{query}' processing failed: {e}", exc_info=False)
        method_result = {
            "error": f"Exception: {e}", 
            "status": "error",
            "token_usage": token_tracker.get_usage()
        }
    output_item = {
        **item, 
        "method_result": method_result,
        "method_used": "rarr"
    }
    return output_item

def main():
    parser = argparse.ArgumentParser(description="Run RARR (Retrofit Attribution using Research and Revision) Experiment.")
    
    default_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to config file.")
    parser.add_argument("--model", type=str, required=True, help="Model name (defined in config).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--end", type=int, default=None, help="End index.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval.")
    parser.add_argument("--output_dir", type=str, default="results/rarr", help="Output directory.")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix.")

    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return
        
    logging.info(f"--- RARR Experiment Start ---")
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
    
    output_filename = f"rarr_{args.start}-{end_idx}_{args.output_suffix}_{timestamp}.jsonl"
    output_path = os.path.join(results_dir, output_filename)

    results = []
    for i, item in enumerate(tqdm(data, desc="RARR Processing")):
        result_item = run_single_item_wrapper(item, args.model, config)
        results.append(result_item)
        
        if (i + 1) % args.save_interval == 0:
            save_jsonl(results, output_path)
    
    save_jsonl(results, output_path)
    logging.info(f"Experiment Complete. Saved to: {output_path}")

if __name__ == "__main__":
    main()