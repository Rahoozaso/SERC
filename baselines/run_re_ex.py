import argparse
import os
import sys
import logging
import re
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src")) 

try:
    from src.model_wrappers import generate
    from src.prompts import (BASELINE_PROMPT_TEMPLATE_PN)
    from src.utils import load_config, save_jsonl, get_timestamp, token_tracker
    from src.data_loader import load_dataset
    from src.rag_retriever import RAGRetriever

except ImportError:
    logging.error("--- ImportError Traceback ---")
    logging.error(traceback.format_exc())
    sys.exit(1)

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

# --- [3] RE-EX Helper Functions ---
def prompt_baseline(query: str, model_name: str, config: dict) -> str:
    prompt = BASELINE_PROMPT_TEMPLATE_PN.format(query=query)
    return generate(prompt, model_name, config)

def _parse_re_ex_questions(text: str) -> List[str]:
    """Parses the list of questions generated in RE-EX Step 1."""
    lines = text.strip().splitlines()
    questions = []
    for line in lines:
        # Remove numbered list (1. question -> question) and check for question mark
        cleaned = re.sub(r"^\s*(\d+\.|Q\d:|Sub-question \d+:)\s*", "", line).strip()
        if cleaned and '?' in cleaned:
            questions.append(cleaned)
    return questions

def _format_evidence(qa_list: List[Dict[str, str]]) -> str:
    """Formats retrieved evidence into prompt format."""
    if not qa_list:
        return "No evidence found."
    
    formatted_str = ""
    for i, item in enumerate(qa_list, 1):
        # Use retrieved context from RAGRetriever as Answer
        formatted_str += f"[Sub-question {i}]: {item['question']}\n[Sub-answer {i}]: {item['evidence']}\n\n"
    return formatted_str.strip()

def _clean_re_ex_output(raw_response: str) -> str:
    """Final answer cleaning (Reusing CoVe cleaning logic)"""
    if not raw_response: return ""
    
    # Try extracting only text after "Revised Response:"
    split_patterns = [r"Revised Response:", r"\[Revised Response\]", r"Output:"]
    for pattern in split_patterns:
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if match:
            raw_response = raw_response[match.end():].strip()
            break
            
    # Existing cleaning logic (remove unnecessary tags)
    clean_text = re.sub(r'\(Note:.*?\)', '', raw_response, flags=re.IGNORECASE)
    clean_text = clean_text.strip().strip('"').strip("'")
    return clean_text

# --- [4] RE-EX Main Execution Function ---

def run_re_ex(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes RE-EX (Revising Explanations) 3-Step Pipeline
    Step 1: Generate Questions & Retrieve Evidence
    Step 2: Explain Factual Errors
    Step 3: Revise Response
    """
    history = {'query': query, 'model_name': model_name, 'params': {'method': 're-ex'}, 'steps': {}}
    
    try:
        retriever = RAGRetriever(config=config)
    except Exception as e:
        logging.error(f"RAG Retriever Init Failed: {e}", exc_info=True)
        history['error'] = f"Retriever Init Error: {e}"
        return history

    try:
        # --- Stage 0: Generate Initial Answer (Baseline) ---
        logging.info("  [RE-EX 0/3] Generating initial answer...")
        initial_response = prompt_baseline(query, model_name, config)
        history['steps']['0_initial_response'] = initial_response
        logging.info(f"    Baseline Length: {len(initial_response)} chars")

        # --- Stage 1: Generate Questions & Retrieve Evidence ---
        logging.info("  [RE-EX 1/3] Generating questions & retrieving evidence...")
        
        # 1-1. Generate Questions (Batch)
        step1_prompt = RE_EX_STEP1_TEMPLATE.format(query=query, initial_response=initial_response)
        questions_raw = generate(step1_prompt, model_name, config)
        questions = _parse_re_ex_questions(questions_raw)
        
        history['steps']['1_questions_raw'] = questions_raw
        history['steps']['1_parsed_questions'] = questions
        logging.info(f"    Generated {len(questions)} sub-questions.")

        # 1-2. Retrieve Evidence
        evidence_list = []
        for q in questions:
            # Retrieve documents using RAGRetriever (Retrieved context itself is treated as Answer)
            retrieved_docs = retriever.retrieve(q)
            evidence_list.append({'question': q, 'evidence': retrieved_docs})
        
        history['steps']['1_evidence_list'] = evidence_list

        # --- Stage 2: Factual Error Explanation ---
        logging.info("  [RE-EX 2/3] Generating factual error explanation...")
        evidence_str = _format_evidence(evidence_list)
        
        step2_prompt = RE_EX_STEP2_TEMPLATE.format(
            evidence=evidence_str,
            query=query,
            initial_response=initial_response
        )
        explanation = generate(step2_prompt, model_name, config)
        history['steps']['2_explanation'] = explanation
        logging.info(f"    Explanation: {explanation[:100]}...")

        # --- Stage 3: Final Revision ---
        if "None" in explanation or "no factual errors" in explanation.lower():
            logging.info("    No errors detected -> Keeping original response")
            final_output = initial_response
            history['steps']['3_revision'] = "Skipped (No errors)"
        else:
            logging.info("  [RE-EX 3/3] Revising final response...")
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
        logging.error(f"Error during RE-EX execution: {e}", exc_info=True)
        history['error'] = str(e)
        history['final_output'] = f"Error: {e}"

    return history

# --- Single Item Processing Wrapper ---
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
        logger.error(f"Error processing '{query}' (RE-EX): {e}", exc_info=False)
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

# --- Main Function ---
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
        logger.error(f"Config load failed: {e}")
        return
        
    logging.info(f"--- RE-EX Experiment Start ---")
    logging.info(f"Dataset: {args.dataset} | Model: {args.model}")

    # Load Dataset
    dataset_path = os.path.join(PROJECT_ROOT, config['data_paths'][args.dataset])
    data = load_dataset(args.dataset, dataset_path)
    
    if not data: return

    # Slicing
    end_idx = args.end if args.end is not None else len(data)
    data = data[args.start : end_idx]

    # Path Setup
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
    logging.info(f"Experiment Complete. Saved to: {output_path}")

if __name__ == "__main__":
    main()