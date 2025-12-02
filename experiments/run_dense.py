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
    from src.rag_retriever import RAGRetriever

    from src.prompts import (
        generate_sentence_group_question_prompt,
        VERIFICATION_ANSWER_TEMPLATE_RAG,
        VALIDATE_EVIDENCE_TEMPLATE,
        CORRECT_FACT_TEMPLATE_RAG,
        BASELINE_PROMPT_TEMPLATE_PN,
        EXTRACT_FACTS_TEMPLATE_PN,
        QUERY_ENTITY_EXTRACTOR_TEMPLATE,
        BASELINE_ENTITY_EXTRACTOR_TEMPLATE,
        RAG_DOMINANT_ENTITY_TEMPLATE,
        ENTITY_CONSISTENCY_JUDGE_TEMPLATE,
        BASELINE_PROMPT_TEMPLATE_RAG_FIRST,
        RECONSTRUCT_LOCAL_SENTENCE_TEMPLATE,
        GLOBAL_POLISH_TEMPLATE,
        BP_CORRECTION_TEMPLATE 
    )
except ImportError as e:
    logging.error(f"ImportError: {e}. Check your src/ folder and PYTHONPATH.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions: String & Formatting
# =============================================================================

def _extract_xml_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    pattern1 = f"<{tag}>(.*?)</{tag}>"
    pattern2 = f"({tag})(.*?)</{tag}>"
    match = re.search(pattern1 or pattern2, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    close_tag = f"</{tag}>"
    if close_tag in text:
        return text.split(close_tag)[0].strip()
    return ""

def _clean_model_output(raw: str) -> str:
    if not raw:
        return ""
    if "</" in raw:
        raw = raw.split("</")[0]
    stop_patterns = ["[END", "[/FINAL", "[ANSWER", "[SOLUTION"]
    for pat in stop_patterns:
        if pat in raw:
            if raw.find(pat) > 5: 
                raw = raw.split(pat)[0]
    cleaned = re.sub(r'#.*$', '', raw, flags=re.MULTILINE) 
    return cleaned.strip().strip('"').strip("'")

# =============================================================================
# Helper Functions: LLM Prompts
# =============================================================================

def prompt_baseline(query: str, model_name: str, config: dict) -> str:
    prompt = BASELINE_PROMPT_TEMPLATE_PN.format(query=query)
    return generate(prompt, model_name, config)

def prompt_extract_entity_desc(text: str, model_name: str, config: dict, is_query: bool = False) -> str:
    template = QUERY_ENTITY_EXTRACTOR_TEMPLATE if is_query else BASELINE_ENTITY_EXTRACTOR_TEMPLATE
    prompt = template.format(query=text) if is_query else template.format(baseline_text=text)
    raw = generate(prompt, model_name, config)
    match = re.search(r"(.+?)\s*\(([^)]+)\)", raw)
    if match and len(match.group(1).split()) <= 6:
        return f"{match.group(1).strip()} ({match.group(2).strip()})"
    return ""

def prompt_extract_rag_desc(query: str, context: str, model_name: str, config: dict) -> str:
    prompt = RAG_DOMINANT_ENTITY_TEMPLATE.format(query=query, context=context)
    raw = generate(prompt, model_name, config)
    match = re.search(r"(.+?)\s*\(([^)]+)\)", raw)
    if match:
        return f"{match.group(1).strip()} ({match.group(2).strip()})"
    return ""

def prompt_judge_entity_consistency(a: str, b: str, model_name: str, config: dict) -> bool:
    prompt = ENTITY_CONSISTENCY_JUDGE_TEMPLATE.format(desc_a=a, desc_b=b)
    raw = generate(prompt, model_name, config)
    if "YES" in _extract_xml_tag(raw, "judgment").upper():
        return True
    elif "NO" in _extract_xml_tag(raw, "judgment").upper():
        return False
    elif "YES" in raw.upper():
         return True
    elif "NO" in raw.upper():
        return False
    else:
        return True

def prompt_regenerate_baseline_rag(query: str, context: str, model_name: str, config: dict) -> str:
    prompt = BASELINE_PROMPT_TEMPLATE_RAG_FIRST.format(context=context, query=query)
    return generate(prompt, model_name, config)

def prompt_extract_facts_from_sentence(sentence: str, model_name: str, config: dict, main_subject: str) -> List[str]:
    prompt = EXTRACT_FACTS_TEMPLATE_PN.format(sentence=sentence, main_subject=main_subject)
    raw = generate(prompt, model_name, config)

    stop_markers = ["[/INSTRUCTION]", "[/INSTURCTION]", "[/INST]", "[RESPONSE]"]
    for marker in stop_markers:
        if marker in raw:
            raw = raw.split(marker)[0]

    facts_block_match = re.search(r"<facts>(.*?)</facts>", raw, re.DOTALL | re.IGNORECASE)
    
    if facts_block_match:
        content_to_search = facts_block_match.group(1)
    else:
        content_to_search = raw

    facts = re.findall(r"<fact>(.*?)</fact>", content_to_search, re.DOTALL | re.IGNORECASE)
    
    if not facts:
        facts = [line.strip().lstrip("- ").strip() for line in content_to_search.split('\n') if line.strip().startswith('- ')]
        if facts:
            logging.warning("  [Extract Method] SUCCESS: Used hyphen fallback (XML failed).")
    
    if facts:
        facts = [f.strip() for f in facts if f.strip()]

    if not facts:
        logging.warning(f"  [CRITICAL FAIL] Fact extraction failed completely. Using sentence as fact.")
        return [sentence] 

    return facts

def _prompt_generate_question_for_sentence_group(facts: List[str], model_name: str, config: dict, main_subject: str) -> str:
    prompt = generate_sentence_group_question_prompt(facts)
    raw = generate(prompt, model_name, config)
    q = _extract_xml_tag(raw, "query")
    return q if q else f"{_clean_model_output(raw)} {main_subject}"

def _prompt_get_verification_answer(question: str, model_name: str, config: dict, context: str) -> str:
    prompt = VERIFICATION_ANSWER_TEMPLATE_RAG.format(query=question, context=context,generation_params_override={"temperature": 0.1, "max_new_tokens": 512})
    raw = generate(prompt, model_name, config,generation_params_override={"max_new_tokens": 400, "temperature": 0.1})
    return _clean_model_output(raw)

def prompt_validate_one_fact_against_evidence(fact: str, evidence: str, model_name: str, config: dict) -> str:
    prompt = VALIDATE_EVIDENCE_TEMPLATE.format(fact_text=fact, evidence_text=evidence)
    raw = generate(prompt, model_name, config,generation_params_override={"temperature": 0.0, "max_new_tokens": 400})
    judgment = _extract_xml_tag(raw, "judgment").upper()
    if "CONTRADICTED" in judgment: return "CONTRADICTED"
    if "SUPPORTED" in judgment: return "SUPPORTED"
    return "NOT_FOUND"

def prompt_generate_correct_fact(error_fact: str, model_name: str, config: dict, context: str) -> str:
    prompt = CORRECT_FACT_TEMPLATE_RAG.format(fact_text=error_fact, context=context)
    raw = generate(prompt, model_name, config)
    return _extract_xml_tag(raw, "corrected_fact") or _clean_model_output(raw)

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
    raw = generate(prompt, model_name, config,generation_params_override={"temperature": 0.1, "max_new_tokens": 256})
    modified_raw = f"<final_response>{raw}"
    return _extract_xml_tag(modified_raw, "final_response") or _clean_model_output(modified_raw)

# =============================================================================
# Batch Processing Functions (Detection & Correction Split)
# =============================================================================

def _detect_syndromes_batch(sentence_batches: List[Dict], 
                            model_name: str, 
                            config: Dict, 
                            retriever: RAGRetriever, 
                            main_subject: str) -> Dict[str, Any]:
    """
    [MODIFIED] Atomic Fact 단위로 개별 검색 및 검증 수행
    """
    clean_facts = []
    syndromes_buffer = [] 
    facts_to_delete = []

    logging.info(">>> [Step 1] Syndrome Detection Started (Atomic Retrieval Mode)")
    
    # 
    for batch in tqdm(sentence_batches, desc="Phase 1: Detecting (Atomic)"):
        facts = batch["original_facts"]
        if not facts: continue

        # --- [변경 구간 시작] ---
        # 기존: 문장 단위로 facts 전체를 묶어서 1회 검색
        # 변경: fact 하나하나마다 개별 검색 수행
        
        for fact in facts:
            # 1. 개별 Fact를 위한 검색 쿼리 생성
            # (기존 함수에 리스트 형태로 [fact] 하나만 전달)
            search_q = _prompt_generate_question_for_sentence_group([fact], model_name, config, main_subject)
            
            # 2. 개별 Retrieval 수행
            context = retriever.retrieve(search_q)
            
            # 3. 개별 Evidence 생성
            evidence = _prompt_get_verification_answer(search_q, model_name, config, context)

            # 4. 검증 (Verdict)
            verdict = prompt_validate_one_fact_against_evidence(fact, evidence, model_name, config)
            
            if verdict == "SUPPORTED":
                clean_facts.append(fact)
            elif verdict == "CONTRADICTED":
                error_package = {
                    "original_fact": fact,  
                    "evidence": evidence,   # 이 팩트만을 위한 구체적 Evidence
                    "context": context,     # 이 팩트만을 위한 구체적 Context
                    "origin_sentence": batch["sentence"] 
                }
                syndromes_buffer.append(error_package)
                logging.info(f"Error Detected: {fact[:30]}...")
                logging.warning(f"   📌 Fact: {fact}")
                logging.warning(f"   🔎 Evidence: {evidence[:50]}...")
            elif verdict == "NOT_FOUND":
                facts_to_delete.append(fact)
                logging.warning(f"🗑️ Not Found (Unverified): {fact[:30]}")
        # --- [변경 구간 끝] ---
    
    return {
        "clean_facts": clean_facts,
        "syndromes_buffer": syndromes_buffer,
        "facts_to_delete": facts_to_delete
    }

def _correct_syndromes_batch(syndromes_buffer: List[Dict], 
                             model_name: str, 
                             config: Dict) -> Dict[str, str]:
    fact_correction_map = {}
    
    if not syndromes_buffer:
        logging.info(">>> [Step 2] No errors to fix. Skipping correction.")
        return {}

    # 1. 문장별로 오류 그룹화
    error_groups = defaultdict(list)
    for item in syndromes_buffer:
        error_groups[item["origin_sentence"]].append(item)

    logging.info(f">>> [Step 2] BP Correction Started ({len(error_groups)} sentence groups)")

    for sentence, items in tqdm(error_groups.items(), desc="Phase 2: Correcting"):
        all_evidences = [item["evidence"] for item in items]
        combined_evidence = "\n".join(all_evidences)
        
        # [중요] 이 리스트가 바로 '정답지(Key)' 목록입니다.
        original_facts_list = [item['original_fact'] for item in items]

        # 프롬프트에 넣을 에러 블록 생성
        error_block = ""
        for i, fact in enumerate(original_facts_list, 1):
            error_block += f"{i}. {fact}\n"
        
        # 템플릿 포맷팅 (합쳐진 evidence 사용)
        prompt = BP_CORRECTION_TEMPLATE.format(
            context=combined_evidence, 
            error_block=error_block
        )
        
        prompt_with_prefill = prompt.strip() + "\n<correction>"

        raw_output_fragment = generate(prompt_with_prefill, model_name, config, 
                                       generation_params_override={
                                           "max_new_tokens": 256, 
                                           "temperature": 0.1,
                                       })
        
        raw_output = "<correction>" + raw_output_fragment

        # XML 파싱  
        correction_blocks = re.findall(r"<correction>(.*?)</correction>", raw_output, re.DOTALL | re.IGNORECASE)
        
        for block in correction_blocks:
            orig_match = re.search(r"<original>(.*?)</original>", block, re.DOTALL | re.IGNORECASE)
            fixed_match = re.search(r"<fixed>(.*?)</fixed>", block, re.DOTALL | re.IGNORECASE)
            
            if orig_match and fixed_match:
                llm_orig = orig_match.group(1).strip()
                clean_corr = fixed_match.group(1).strip()
                
                llm_orig_clean = re.sub(r'^[\d\-\.\)\s]+', '', llm_orig)
                matches = get_close_matches(llm_orig_clean, original_facts_list, n=1, cutoff=0.7)
                
                if matches:
                    true_key = matches[0] 
                    fact_correction_map[true_key] = clean_corr
                    
                    if clean_corr:
                        logging.info(f"🔗 Matched: '{llm_orig_clean[:15]}...' -> Key: '{true_key[:15]}...'")
                    else:
                        logging.info(f"🗑️ Matched (Delete): '{true_key[:15]}...'")
                else:
                    logging.warning(f"⚠️ Match Failed: LLM said '{llm_orig_clean}' but not found in list.")
            
    return fact_correction_map

# =============================================================================
# Main SERC Implementation
# =============================================================================

def SERC(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:

    logging.info(f"--- SERC Started --- Query: '{query[:60]}...'")

    retriever = RAGRetriever(config=config)
    history = {"query": query, "model_name": model_name, "steps": {}}

    # Step 1: Baseline Generation
    baseline = prompt_baseline(query, model_name, config)
    is_refusal = (
        len(baseline) < 50 or 
        "sorry" in baseline.lower() or 
        "cannot answer" in baseline.lower() or
        "don't have information" in baseline.lower() or
        "unable to provide" in baseline.lower() or
        "not have access" in baseline.lower() or
        "i couldn't find" in baseline.lower() or
        "limited information" in baseline.lower()
    )
    logging.info(f"Is Refusal: {is_refusal}")

    if is_refusal:
        logging.warning(" Baseline refused to answer (Source Dropout). Attempting RAG-First Cold Start...")
        rag_context = retriever.retrieve(query)
        
        if rag_context and len(rag_context) > 10:
            baseline = prompt_regenerate_baseline_rag(query, rag_context, model_name, config)
            history["regenerated_baseline"] = baseline
            logging.info("RAG-First Cold Start Successful.")
        else:
            logging.error("RAG lookup also failed. Terminating process.")
            return {
                "query": query, 
                "final_output": "I apologize, but I could not find verified information regarding your query.",
                "status": "no_info"
            }
    
    history["initial_baseline"] = baseline
    logging.info("--- [Step 1.5] Entity Firewall check ---")
    query_entity = prompt_extract_entity_desc(query, model_name, config, is_query=True)
    model_entity = prompt_extract_entity_desc(baseline, model_name, config, is_query=False)
    rag_context = retriever.retrieve(query)
    rag_entity = prompt_extract_rag_desc(query, rag_context, model_name, config)
    logging.info(f" Model Entity: {model_entity} / RAG Entity: {rag_entity}")
    is_consistent = False
    if not model_entity or not rag_entity:
        is_consistent = True
    else:
        is_consistent = prompt_judge_entity_consistency(model_entity, rag_entity, model_name, config)
    if not is_consistent:
        logging.warning(f" Entity Mismatch Detected! (Model: {model_entity} vs RAG: {rag_entity})")
        logging.warning("Triggering Hard Reset: Regenerating Baseline with RAG Context...")
        baseline = prompt_regenerate_baseline_rag(query, rag_context, model_name, config)
        history["regenerated_baseline"] = baseline
        history["firewall_triggered"] = True
        main_subject = rag_entity
    else:
        logging.info("Entity Check Passed.")
        main_subject = model_entity if model_entity and len(model_entity) > len(rag_entity) else (rag_entity or query_entity or query)
    if not main_subject:
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

    # Step 3: Detection (Batch 1:1 Detection)
    # [수정됨] Atomic Fact 단위 검색/검증 함수 호출
    detection_result = _detect_syndromes_batch(
        sentence_batches=sentence_batches,
        model_name=model_name,
        config=config,
        retriever=retriever,
        main_subject=main_subject
    )
    
    clean_facts = detection_result["clean_facts"]
    syndromes_buffer = detection_result["syndromes_buffer"]
    facts_to_delete = detection_result["facts_to_delete"]

    # Step 4: Correction (Grouped Belief Propagation)
    fact_correction_map = _correct_syndromes_batch(
        syndromes_buffer=syndromes_buffer,
        model_name=model_name,
        config=config
    )
    if facts_to_delete:
        logging.info(f"Applying direct deletion for {len(facts_to_delete)} unverified facts.")
        for f in facts_to_delete:
            fact_correction_map[f] = ""  # 빈 문자열 = 삭제 (Step 5 로직에 의해)

    history["steps"]["syndromes_detected"] = len(syndromes_buffer)
    history["steps"]["fact_correction_map"] = fact_correction_map

    # Step 5: Local Sentence Reconstruction
    logging.info("--- Local Sentence Reconstruction ---")
    local_sentences = []
    accumulated_facts = []

    for batch in sentence_batches:
        orig_sent = batch["sentence"]
        old_facts = batch["original_facts"]
        
        updated_facts_list = []
        has_changes = False  # [최적화] 변경 사항 감지 플래그
        
        for f in old_facts:
            if f in fact_correction_map:
                updated_facts_list.append(fact_correction_map[f])
                has_changes = True  # 변경 발생!
            else:
                updated_facts_list.append(f)
        prev_context_str = "\n".join(f"- {f}" for f in accumulated_facts)
        
        if not has_changes:
            local_sentences.append(orig_sent)
            logging.info(f"Skipped Reconstruction (No Errors): {orig_sent[:30]}...")
            continue
            
        # 변경된 사실이 있다면? -> 새로 생성 (Generate from Scratch)
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
        if updated_facts_list:
            accumulated_facts.extend(updated_facts_list)

    history["steps"]["local_sentences"] = local_sentences

    # Step 6: Global Polishing
    logging.info("--- Global Polishing ---")
    draft = "\n\n".join(local_sentences)

    if len(draft.strip()) < 50:
        logging.warning("Draft too short → fallback to baseline")
        final_output = baseline
    else:
        final_output = prompt_global_polish(query=query, draft_text=draft,
                                            model_name=model_name, config=config).strip()

    final_output = final_output or baseline
    history["final_output"] = final_output
    history["steps"]["global_draft"] = draft

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
        result = SERC(query=q, model_name=model_name, config=config)
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
                "token_usage": token_tracker.get_usage() # 에러 나기 전까지 쓴 거라도 기록
            }
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="SERC: Hierarchical Belief Propagation Hallucination Correction")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/serc")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = os.path.join(PROJECT_ROOT, config['data_paths'][args.dataset])
    data = load_dataset(args.dataset, data_path)
    data = data[args.start: args.end]

    timestamp = get_timestamp()
    os.makedirs(os.path.join(PROJECT_ROOT, args.output_dir, args.model.replace('/', '_'), args.dataset), exist_ok=True)
    output_path = os.path.join(PROJECT_ROOT, args.output_dir, args.model.replace('/', '_'), args.dataset,
                               f"serc_{args.start}-{len(data)+args.start}_{timestamp}.jsonl")

    results = []
    for i, item in enumerate(tqdm(data, desc="SERC Processing")):
        results.append(run_single_item(item, args.model, config))
        if (i + 1) % args.save_interval == 0:
            save_jsonl(results, output_path)

    save_jsonl(results, output_path)
    logging.info(f"Done. Results → {output_path}")

if __name__ == "__main__":
    main()