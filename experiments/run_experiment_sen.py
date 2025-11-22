import argparse
import os
import sys
import logging
import re
from typing import Dict, Any, List, Optional
from collections import defaultdict
from tqdm import tqdm

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
        BP_CORRECTION_TEMPLATE  # [확인] src/prompts.py에 정의되어 있어야 함
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
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
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
    return "YES" in _extract_xml_tag(raw, "judgment").upper() or "YES" in raw.upper()

def prompt_regenerate_baseline_rag(query: str, context: str, model_name: str, config: dict) -> str:
    prompt = BASELINE_PROMPT_TEMPLATE_RAG_FIRST.format(context=context, query=query)
    return generate(prompt, model_name, config)

def prompt_extract_facts_from_sentence(sentence: str, model_name: str, config: dict, main_subject: str) -> List[str]:
    prompt = EXTRACT_FACTS_TEMPLATE_PN.format(sentence=sentence, main_subject=main_subject)
    raw = generate(prompt, model_name, config)

    # 1. 불필요한 종료 태그 이후 내용 제거 (Prompt Leakage 방지)
    stop_markers = ["[/INSTRUCTION]", "[/INSTURCTION]", "[/INST]", "[RESPONSE]"]
    for marker in stop_markers:
        if marker in raw:
            raw = raw.split(marker)[0]

    # 2. <facts> 블록 범위 확인 및 로깅
    facts_block_match = re.search(r"<facts>(.*?)</facts>", raw, re.DOTALL | re.IGNORECASE)
    
    if facts_block_match:
        # 1순위: <facts> 블록 안에서 검색
        content_to_search = facts_block_match.group(1)
        logging.info("  [Extract Method] PRIMARY: Targeting content inside <facts> block.")
    else:
        # 2순위: 전체 텍스트에서 검색 (Fallback)
        content_to_search = raw
        logging.info("  [Extract Method] FALLBACK: <facts> block not found. Searching raw output.")

    # 3. XML 태그 추출
    facts = re.findall(r"<fact>(.*?)</fact>", content_to_search, re.DOTALL | re.IGNORECASE)
    
    # 4. XML 실패 시 백업 파싱 (하이픈 - )
    if not facts:
        facts = [line.strip().lstrip("- ").strip() for line in content_to_search.split('\n') if line.strip().startswith('- ')]
        if facts:
            logging.warning("  [Extract Method] SUCCESS: Used hyphen fallback (XML failed).")
    
    # 5. 공백 제거 및 정리
    if facts:
        facts = [f.strip() for f in facts if f.strip()]

    # 6. [최종 방어] 팩트가 추출되지 않으면, 원본 문장 전체를 하나의 팩트로 반환
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
    prompt = VERIFICATION_ANSWER_TEMPLATE_RAG.format(query=question, context=context)
    raw = generate(prompt, model_name, config)
    return _clean_model_output(raw)

def prompt_validate_one_fact_against_evidence(fact: str, evidence: str, model_name: str, config: dict) -> str:
    prompt = VALIDATE_EVIDENCE_TEMPLATE.format(fact_text=fact, evidence_text=evidence)
    raw = generate(prompt, model_name, config,
                   generation_params_override={"temperature": 0.1, "max_new_tokens": 1024})
    judgment = _extract_xml_tag(raw, "judgment").upper()
    if "CONTRADICTED" in judgment: return "CONTRADICTED"
    if "SUPPORTED" in judgment: return "SUPPORTED"
    return "NOT_FOUND"

def prompt_generate_correct_fact(error_fact: str, model_name: str, config: dict, context: str) -> str:
    prompt = CORRECT_FACT_TEMPLATE_RAG.format(fact_text=error_fact, context=context)
    raw = generate(prompt, model_name, config)
    return _extract_xml_tag(raw, "corrected_fact") or _clean_model_output(raw)

def prompt_reconstruct_local_sentence(original_sentence: str, updated_facts: List[str],
                                      query: str, model_name: str, config: dict) -> str:
    # [핵심] XML 방식: 원문은 무시하고 팩트 리스트로 새로 짓기
    fact_list_str = "\n".join(f"- {f}" for f in updated_facts)
    prompt = RECONSTRUCT_LOCAL_SENTENCE_TEMPLATE.format(
        original_sentence=original_sentence,
        updated_facts=fact_list_str
    )
    raw = generate(prompt, model_name, config,
                   generation_params_override={"temperature": 0.3, "max_new_tokens": 512})
    modified_raw = f"<generated_sentence>{raw}"
    # XML 태그 추출
    return _extract_xml_tag(modified_raw, "generated_sentence") or _clean_model_output(modified_raw)

def prompt_global_polish(query: str, draft_text: str, model_name: str, config: dict) -> str:
    prompt = GLOBAL_POLISH_TEMPLATE.format(query=query, draft_text=draft_text)
    raw = generate(prompt, model_name, config,
                   generation_params_override={"temperature": 0.5, "max_new_tokens": 1024})
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
    clean_facts = []
    syndromes_buffer = [] 

    logging.info(">>> [Step 1] Syndrome Detection Started (Collecting Errors...)")
    
    for batch in tqdm(sentence_batches, desc="Phase 1: Detecting"):
        facts = batch["original_facts"]
        if not facts: continue
        search_q = _prompt_generate_question_for_sentence_group(facts, model_name, config, main_subject)
        context = retriever.retrieve(search_q)
        evidence = _prompt_get_verification_answer(search_q, model_name, config, context)

        for fact in facts:
            verdict = prompt_validate_one_fact_against_evidence(fact, evidence, model_name, config)
            
            if verdict == "SUPPORTED":
                clean_facts.append(fact)
            elif verdict == "CONTRADICTED":
                error_package = {
                    "original_fact": fact,  
                    "evidence": evidence,   
                    "context": context,
                    "origin_sentence": batch["sentence"] 
                }
                syndromes_buffer.append(error_package)
                logging.info(f"Error Detected: {fact[:30]}...")
                logging.warning(f"   📌 Fact: {fact}")
                logging.warning(f"   🔗 Origin: {batch['sentence'][:50]}...")
    
    return {
        "clean_facts": clean_facts,
        "syndromes_buffer": syndromes_buffer
    }

def _correct_syndromes_batch(syndromes_buffer: List[Dict], 
                             model_name: str, 
                             config: Dict) -> Dict[str, str]:
    fact_correction_map = {}
    
    if not syndromes_buffer:
        logging.info(">>> [Step 2] No errors to fix. Skipping correction.")
        return {}
    error_groups = defaultdict(list)
    for item in syndromes_buffer:
        error_groups[item["origin_sentence"]].append(item)

    logging.info(f">>> [Step 2] BP Correction Started ({len(error_groups)} sentence groups)")

    for sentence, items in tqdm(error_groups.items(), desc="Phase 2: Correcting with XML"):
        context = items[0]["context"]
        error_block = ""
        for i, item in enumerate(items, 1):
            error_block += f"{i}. {item['original_fact']}\n"
        prompt = BP_CORRECTION_TEMPLATE.format(
            context=context,
            error_block=error_block
        )
        
        raw_output = generate(prompt, model_name, config,generation_params_override={"max_new_tokens": 512, "temperature": 0.1 })
        correction_blocks = re.findall(r"<correction>(.*?)</correction>", raw_output, re.DOTALL | re.IGNORECASE)
        
        for block in correction_blocks:
            # 2. 블록 내부에서 original과 fixed 추출
            orig_match = re.search(r"<original>(.*?)</original>", block, re.DOTALL | re.IGNORECASE)
            fixed_match = re.search(r"<fixed>(.*?)</fixed>", block, re.DOTALL | re.IGNORECASE)
            
            if orig_match and fixed_match:
                # 태그 안의 텍스트 추출 및 정리
                clean_orig = orig_match.group(1).strip()
                clean_corr = fixed_match.group(1).strip()
                clean_orig = re.sub(r'^[\d\-\.\)\s]+', '', clean_orig)
                
                if clean_orig and clean_corr:
                    fact_correction_map[clean_orig] = clean_corr
                    logging.info(f"XML Fixed: {clean_orig[:15]}... -> {clean_corr[:15]}...")
            
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
        "i couldn't find" in baseline.lower()
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
    detection_result = _detect_syndromes_batch(
        sentence_batches=sentence_batches,
        model_name=model_name,
        config=config,
        retriever=retriever,
        main_subject=main_subject
    )
    
    clean_facts = detection_result["clean_facts"]
    syndromes_buffer = detection_result["syndromes_buffer"]

    # Step 4: Correction (Grouped Belief Propagation)
    fact_correction_map = _correct_syndromes_batch(
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
        has_changes = False  # [최적화] 변경 사항 감지 플래그
        
        for f in old_facts:
            if f in fact_correction_map:
                updated_facts_list.append(fact_correction_map[f])
                has_changes = True  # 변경 발생!
            else:
                updated_facts_list.append(f)
        
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
            config=config
        )
        final_sent = reconstructed.strip() if reconstructed and len(reconstructed) > 10 else orig_sent
        local_sentences.append(final_sent)

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

def run_single_item(item: Dict[str, Any], model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    q = item.get("question") or item.get("query")
    token_tracker.reset()
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