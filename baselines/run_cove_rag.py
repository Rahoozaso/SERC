import argparse
import os
import sys
import logging
import re
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import traceback

# --- [1] Project Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
# Add src folder to path
sys.path.append(os.path.join(PROJECT_ROOT, "src")) 

try:
    from src.model_wrappers import generate
    from src import prompts
    from src.utils import load_config, save_jsonl, get_timestamp, token_tracker
    from src.data_loader import load_dataset
    from src.rag_retriever import RAGRetriever
    from src.prompts import VERIFICATION_ANSWER_TEMPLATE_RAG
    from src.prompts import BASELINE_PROMPT_TEMPLATE_PN

except ImportError:
    logging.error("--- ImportError Traceback (Full Error Log) ---")
    logging.error(traceback.format_exc())
    logging.error("ImportError: Failed to import modules from 'src' folder. Please check your PYTHONPATH.")
    sys.exit(1)


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [2] CoVe-RAG Helper Functions ---

def prompt_baseline(query: str, model_name: str, config: dict) -> str:
    prompt = BASELINE_PROMPT_TEMPLATE_PN.format(query=query)
    return generate(prompt, model_name, config)

def _parse_cove_questions(text: str) -> List[str]:
    """Parses the list of questions generated during the CoVe planning stage."""
    lines = text.strip().splitlines()
    questions = [re.sub(r"^\s*(\d+\.|Q\d:)\s*", "", line).strip() for line in lines]
    return [q for q in questions if q and q != ""] # Remove empty lines

def _format_qa_evidence(qa_list: List[Dict[str, str]]) -> str:
    """Formats the verification Q&A list into a string to be inserted into the final prompt."""
    if not qa_list:
        return "No verification results."
    
    formatted_str = ""
    for i, qa in enumerate(qa_list, 1):
        formatted_str += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"
    return formatted_str.strip()

def _cove_get_rag_answer(question: str, context: str, model_name: str, config: dict) -> str:
    """CoVe Stage 3: Generate verification answer using RAG"""
    prompt = prompts.VERIFICATION_ANSWER_TEMPLATE_RAG.format( context=context, query=question)
    answer_params = {"temperature": 0.01, "max_new_tokens": 100}
    raw_response = generate(prompt, model_name, config, generation_params_override=answer_params)
    
    # Cleaning logic
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
    """Reusing SERC's _clean_model_output (For CoVe Stage 4 cleaning)"""
    if not raw_response: return ""
    def _final_scrub(line: str) -> str:
        line = re.sub(r'#.*$', '', line).strip()
        line = re.sub(r'\[.*?\]$', '', line).strip()
        line = re.sub(r'END OF INSTRUCTION.*$', '', line, flags=re.IGNORECASE).strip()
        line = re.sub(r'Note:.*$', '', line, flags=re.IGNORECASE).strip()
        line = re.sub(r'//', '', line, flags=re.IGNORECASE).strip()
        return line.strip().strip('"').strip("'")
    
    # Search for [FINAL REVISED RESPONSE] markers first
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

# --- [3] CoVe-RAG Main Execution Function ---

def run_cove_rag(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the 4 stages of CoVe (Chain-of-Verification).
    """
    cove_history = {'query': query, 'model_name': model_name, 'params': {'method': 'cove-rag'}, 'steps': {}}
    
    try:
        retriever = RAGRetriever(config=config)
    except Exception as e:
        logging.error(f"RAG Retriever (CoVe) initialization failed: {e}", exc_info=True)
        cove_history['error'] = f"RAG Retriever (CoVe) initialization failed: {e}"
        cove_history['final_output'] = "Error during CoVe initialization."
        return cove_history

    try:
        # --- Stage 1: Generate Initial Answer ---
        logging.info("  [CoVe-RAG 1/4] Generating initial answer...")
        initial_baseline = prompt_baseline(query, model_name, config)
        cove_history['steps']['1_initial_baseline'] = initial_baseline
        logging.info(f"    CoVe Baseline: {initial_baseline[:100]}...")

        # --- Stage 2: Establish Verification Plan ---
        logging.info("  [CoVe-RAG 2/4] Establishing verification plan...")
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
        logging.info(f"    CoVe Plan: {len(verification_questions)} questions generated.")

        # --- Stage 3: Execute Verification (Using RAG) ---
        logging.info(f"  [CoVe-RAG 3/4] Executing RAG verification for {len(verification_questions)} questions...")
        verification_results = []
        for q in verification_questions:
            retrieved_docs = retriever.retrieve(q)
            answer = _cove_get_rag_answer(q, retrieved_docs, model_name, config)
            verification_results.append({'question': q, 'answer': answer, 'retrieved_docs': retrieved_docs})
            logging.debug(f"      Q: {q}\n        A: {answer[:100]}...")
        
        cove_history['steps']['3_verification_results'] = verification_results
        logging.info(f"    CoVe Execution: {len(verification_results)} answers completed.")

        # --- Stage 4: Generate Final Answer ---
        logging.info("  [CoVe-RAG 4/4] Generating final answer...")
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
        logging.error(f"Error occurred during CoVe-RAG execution: {e}", exc_info=True)
        cove_history['error'] = str(e)
        cove_history['final_output'] = f"Error during CoVe-RAG: {e}"

    return cove_history

# --- Single Item Processing Wrapper ---
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
        logger.error(f"Error occurred while processing '{query}' (CoVe-RAG): {e}", exc_info=False)
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

# --- Main Function ---
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
        logger.error(f"Error occurred while loading config file: {e}")
        return
        
    logging.info(f"--- CoVe [RAG] Experiment Start ---")
    logging.info(f"Config: {args.config}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset} (Range: {args.start} ~ {args.end})")

    # Load Dataset
    dataset_config_key = args.dataset
    relative_path = config.get('data_paths', {}).get(dataset_config_key)
    if not relative_path:
         logger.error(f"Key '{dataset_config_key}' not found in Config file.")
         return
    dataset_path = os.path.join(PROJECT_ROOT, relative_path)
    
    try:
        data = load_dataset(dataset_config_key, dataset_path)
    except Exception as e:
        logger.error(f"Error occurred while loading dataset ({dataset_path}). Terminating.", exc_info=True)
        return
    if not data:
        logger.error("No data loaded.")
        return

    total_data_len = len(data)
    end_idx = args.end if args.end is not None else total_data_len
    if args.start < 0: args.start = 0
    if end_idx > total_data_len: end_idx = total_data_len
    
    data = data[args.start : end_idx]
    logging.info(f"Data slicing completed: Total {len(data)} items to be processed.")

    timestamp = get_timestamp()
    results_base_dir_abs = os.path.join(PROJECT_ROOT, args.output_dir)
    results_dir = os.path.join(results_base_dir_abs, args.model.replace('/', '_'), args.dataset)
    os.makedirs(results_dir, exist_ok=True) 
    
    suffix_str = f"_{args.output_suffix}" if args.output_suffix else ""
    output_filename = f"cove_rag_{args.start}-{end_idx}{suffix_str}_{timestamp}.jsonl"
    output_path = os.path.join(results_dir, output_filename)
    logging.info(f"Results will be saved to: {output_path}")

    results = []
    for i, item in enumerate(tqdm(data, desc=f"CoVe (RAG)")):
        result_item = run_single_item_wrapper(item=item, model_name=args.model, config=config)
        results.append(result_item)
        
        if (i + 1) % args.save_interval == 0:
            save_jsonl(results, output_path)
    
    save_jsonl(results, output_path)
    logging.info(f"\n--- CoVe [RAG] Experiment Completed. Total {len(results)} results saved to {output_path}. ---")

if __name__ == "__main__":
    main()