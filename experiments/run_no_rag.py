import argparse
import os
import sys
import logging
import re
from typing import Dict, Any, List, Optional
from collections import defaultdict
from tqdm import tqdm
import random
import numpy as np
import torch
import gc
from difflib import get_close_matches

# --- Project path setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

try:
    from src.utils import token_tracker
    from src import programmatic_helpers as ph
    from src.utils import load_config, save_jsonl, get_timestamp
    from src.data_loader import load_dataset
    from src.model_wrappers import generate
    from src.prompts import (
        BASELINE_PROMPT_TEMPLATE_PN,
        EXTRACT_FACTS_TEMPLATE_PN,
        RECONSTRUCT_LOCAL_SENTENCE_TEMPLATE,
        GLOBAL_POLISH_TEMPLATE,
        SELF_VALIDATE_TEMPLATE,
        SELF_BP_CORRECTION_TEMPLATE   
    )
except ImportError as e:
    logging.error(f"ImportError: {e}. Check if 'src/prompts.py' defines all required templates.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions: String & Formatting
# =============================================================================

def _extract_xml_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def _clean_model_output(raw: str) -> str:
    if not raw:
        return ""
    if "</" in raw:
        raw = raw.split("</")[0]
    return raw.strip().strip('"').strip("'")

# =============================================================================
# Helper Functions: LLM Prompts
# =============================================================================

def prompt_baseline(query: str, model_name: str, config: dict) -> str:
    prompt = BASELINE_PROMPT_TEMPLATE_PN.format(query=query)
    return generate(prompt, model_name, config)

def prompt_extract_facts_from_sentence(sentence: str, model_name: str, config: dict, main_subject: str) -> List[str]:
    prompt = EXTRACT_FACTS_TEMPLATE_PN.format(sentence=sentence, main_subject=main_subject)
    raw = generate(prompt, model_name, config)
    
    facts = re.findall(r"<fact>(.*?)</fact>", raw, re.DOTALL | re.IGNORECASE)
    
    if not facts:
        facts = [line.strip().lstrip("- ").strip() for line in raw.split('\n') if line.strip().startswith('- ')]
    
    if not facts:
        return [sentence] 
    
    return [f.strip() for f in facts if f.strip()]

def prompt_reconstruct_local_sentence(original_sentence: str, updated_facts: List[str],
                                      query: str, model_name: str, config: dict, previous_context: str = "") -> str:
    fact_list_str = "\n".join(f"- {f}" for f in updated_facts)
    prompt = RECONSTRUCT_LOCAL_SENTENCE_TEMPLATE.format(
        previous_context=previous_context,
        original_sentence=original_sentence,
        updated_facts=fact_list_str
    )
    raw = generate(prompt, model_name, config)
    modified_raw = f"<generated_sentence>{raw}"
    return _extract_xml_tag(modified_raw, "generated_sentence") or _clean_model_output(modified_raw)

def prompt_global_polish(query: str, draft_text: str, model_name: str, config: dict) -> str:
    prompt = GLOBAL_POLISH_TEMPLATE.format(query=query, draft_text=draft_text)
    raw = generate(prompt, model_name, config, generation_params_override={"temperature": 0.1, "max_new_tokens": 256})
    modified_raw = f"<final_response>{raw}"
    return _extract_xml_tag(modified_raw, "final_response") or _clean_model_output(modified_raw)

# --- [NEW] Self-Validation Prompt Function ---
def prompt_self_validate_fact(fact: str, model_name: str, config: dict) -> str:
    prompt = SELF_VALIDATE_TEMPLATE.format(fact_text=fact)
    raw = generate(prompt, model_name, config, generation_params_override={"temperature": 0.0, "max_new_tokens": 200})
    
    judgment = _extract_xml_tag(raw, "judgment").upper()
    if "CONTRADICTED" in judgment: return "CONTRADICTED"
    if "SUPPORTED" in judgment: return "SUPPORTED"
    return "SUPPORTED"

# =============================================================================
# Batch Processing Functions (No-RAG Version)
# =============================================================================

def _detect_syndromes_self_check(sentence_batches: List[Dict], 
                                 model_name: str, 
                                 config: Dict) -> Dict[str, Any]:
    clean_facts = []
    syndromes_buffer = [] 
    
    logging.info(">>> [Phase 1] Self-Correction Detection Started (No RAG)")
    
    for batch in tqdm(sentence_batches, desc="Detecting (Internal)"):
        facts = batch["original_facts"]
        if not facts: continue
        
        for fact in facts:
            verdict = prompt_self_validate_fact(fact, model_name, config)
            
            if verdict == "SUPPORTED":
                clean_facts.append(fact)
            else: # CONTRADICTED
                error_package = {
                    "original_fact": fact,
                    "origin_sentence": batch["sentence"]
                    # no evidence 
                }
                syndromes_buffer.append(error_package)
                logging.info(f" Self-Correction Triggered: {fact[:40]}...")

    return {
        "clean_facts": clean_facts,
        "syndromes_buffer": syndromes_buffer
    }

def _correct_syndromes_self_check(syndromes_buffer: List[Dict], 
                                  model_name: str, 
                                  config: Dict) -> Dict[str, str]:
    fact_correction_map = {}
    
    if not syndromes_buffer:
        logging.info(">>> [Phase 2] No errors found by Self-Check. Skipping correction.")
        return {}

    error_groups = defaultdict(list)
    for item in syndromes_buffer:
        error_groups[item["origin_sentence"]].append(item)

    logging.info(f">>> [Phase 2] Self-Correction with Belief Propagation ({len(error_groups)} sentence groups)")

    for sentence, items in tqdm(error_groups.items(), desc="Correcting (Internal)"):
        original_facts_list = [item['original_fact'] for item in items]

        error_block = ""
        for i, fact in enumerate(original_facts_list, 1):
            error_block += f"{i}. {fact}\n"
        
        prompt = SELF_BP_CORRECTION_TEMPLATE.format(error_block=error_block)
        
        prompt_with_prefill = prompt.strip() + "\n<correction>"
        
        raw_output_fragment = generate(prompt_with_prefill, model_name, config, 
                                       generation_params_override={"max_new_tokens": 256, "temperature": 0.1})
        raw_output = "<correction>" + raw_output_fragment

        correction_blocks = re.findall(r"<correction>(.*?)</correction>", raw_output, re.DOTALL | re.IGNORECASE)
        
        for block in correction_blocks:
            orig_match = re.search(r"<original>(.*?)</original>", block, re.DOTALL | re.IGNORECASE)
            fixed_match = re.search(r"<fixed>(.*?)</fixed>", block, re.DOTALL | re.IGNORECASE)
            
            if orig_match and fixed_match:
                llm_orig = orig_match.group(1).strip()
                clean_corr = fixed_match.group(1).strip()
                
                llm_orig_clean = re.sub(r'^[\d\-\.\)\s]+', '', llm_orig)
                
                # Fuzzy Matching
                matches = get_close_matches(llm_orig_clean, original_facts_list, n=1, cutoff=0.6)
                
                if matches:
                    true_key = matches[0]
                    fact_correction_map[true_key] = clean_corr
                    logging.info(f"🔧 Fixed: '{true_key[:20]}...' -> '{clean_corr[:20]}...'")
            
    return fact_correction_map

# =============================================================================
# Main SERC Implementation (No-RAG)
# =============================================================================

def SERC_NoRAG(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:

    logging.info(f"--- SERC (No-RAG) Started --- Query: '{query[:60]}...'")
    
    # RAG Retriever 초기화 삭제됨
    history = {"query": query, "model_name": model_name, "steps": {}}

    # Step 1: Baseline Generation
    baseline = prompt_baseline(query, model_name, config)
    
    # RAG Based Cold Start / Refusal Check delete
    history["initial_baseline"] = baseline
    main_subject = query 

    # Step 2: Fact Extraction per Sentence
    sentences = ph.programmatic_split_into_sentences(baseline)
    sentence_batches = []
    for s in sentences:
        if not s.strip(): continue
        facts = prompt_extract_facts_from_sentence(s, model_name, config, main_subject)
        facts = [f for f in facts if len(f) > 5] 
        if facts:
            sentence_batches.append({"sentence": s, "original_facts": facts})

    history["steps"]["sentence_batches"] = sentence_batches

    # Step 3: Detection (Internal Check)
    detection_result = _detect_syndromes_self_check(
        sentence_batches=sentence_batches,
        model_name=model_name,
        config=config
    )
    
    syndromes_buffer = detection_result["syndromes_buffer"]

    # Step 4: Correction (Internal Belief Propagation)
    fact_correction_map = _correct_syndromes_self_check(
        syndromes_buffer=syndromes_buffer,
        model_name=model_name,
        config=config
    )

    history["steps"]["syndromes_detected"] = len(syndromes_buffer)
    history["steps"]["fact_correction_map"] = fact_correction_map

    # Step 5: Local Sentence Reconstruction
    logging.info("--- Local Sentence Reconstruction ---")
    local_sentences = []
    
    for batch in sentence_batches:
        orig_sent = batch["sentence"]
        old_facts = batch["original_facts"]
        
        updated_facts_list = []
        has_changes = False 
        
        for f in old_facts:
            if f in fact_correction_map:
                updated_facts_list.append(fact_correction_map[f])
                has_changes = True
            else:
                updated_facts_list.append(f)
        
        prev_context_str = " ".join(local_sentences[-2:]) if local_sentences else ""
        
        if not has_changes:
            local_sentences.append(orig_sent)
            continue
        reconstructed = prompt_reconstruct_local_sentence(
            original_sentence=orig_sent,
            updated_facts=updated_facts_list,
            query=query,
            model_name=model_name,
            config=config,
            previous_context=prev_context_str
        )
        final_sent = reconstructed.strip() if reconstructed and len(reconstructed) > 10 else orig_sent
        local_sentences.append(final_sent)

    history["steps"]["local_sentences"] = local_sentences

    # Step 6: Global Polishing
    logging.info("--- Global Polishing ---")
    draft = "\n\n".join(local_sentences)

    if len(draft.strip()) < 10:
        final_output = baseline 
    else:
        final_output = prompt_global_polish(query=query, draft_text=draft,
                                            model_name=model_name, config=config).strip()

    final_output = final_output or baseline
    history["final_output"] = final_output
    
    logging.info("--- SERC Completed ---")
    return history

# =============================================================================
# Execution Wrapper & Main
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_single_item(item: Dict[str, Any], model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    q = item.get("question") or item.get("query")
    token_tracker.reset()
    set_seed(42)
    try:
        # SERC_NoRAG 호출
        result = SERC_NoRAG(query=q, model_name=model_name, config=config)
        usage = token_tracker.get_usage()
        
        return {
            **item, 
            "method_result": {
                "final_output": result["final_output"], 
                "history": result, 
                "status": "success",
                "token_usage": usage
            }
        }
    
    except Exception as e:
        logger.error(f"Error on '{q[:60]}...': {e}", exc_info=True)
        return {
            **item, 
            "method_result": {
                "error": str(e), 
                "status": "error",
                "token_usage": token_tracker.get_usage()
            }
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="SERC (No-RAG): Self-Correction Framework")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/serc_no_rag")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = os.path.join(PROJECT_ROOT, config['data_paths'][args.dataset])
    data = load_dataset(args.dataset, data_path)
    
    end_idx = args.end if args.end is not None else len(data)
    data = data[args.start: end_idx]

    timestamp = get_timestamp()
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir, args.model.replace('/', '_'), args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"serc_norag_{args.start}-{args.start+len(data)}_{timestamp}.jsonl")

    results = []
    logging.info(f"Processing {len(data)} items...")
    
    for i, item in enumerate(tqdm(data, desc="SERC No-RAG Processing")):
        results.append(run_single_item(item, args.model, config))
        if (i + 1) % args.save_interval == 0:
            save_jsonl(results, output_path)

    save_jsonl(results, output_path)
    logging.info(f"Done. Results -> {output_path}")

if __name__ == "__main__":
    main()