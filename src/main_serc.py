import argparse
import os
import sys
import logging
import re
from typing import Dict, Any, List, Optional
from collections import defaultdict
from tqdm import tqdm

# --- Project path setup ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

try:
    from src import programmatic_helpers as ph
    from src.utils import load_config, save_jsonl, get_timestamp
    from src.data_loader import load_dataset
    from src.model_wrappers import generate
    # RAGRetriever는 import 하지 않습니다. (Ablation)

    from src.prompts import (
        BASELINE_PROMPT_TEMPLATE_PN,
        EXTRACT_FACTS_TEMPLATE_PN,
        generate_sentence_group_question_prompt, # 질문 생성 함수 import
        VERIFICATION_ANSWER_TEMPLATE,            # 내부 답변 생성용
        VALIDATE_EVIDENCE_TEMPLATE,              # 1:1 검증용
        BP_CORRECTION_TEMPLATE,                  # BP 수정용
        RECONSTRUCT_LOCAL_SENTENCE_TEMPLATE,
        GLOBAL_POLISH_TEMPLATE,
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
    if not text: return ""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return ""

def _clean_model_output(raw: str) -> str:
    if not raw: return ""
    if "</" in raw: raw = raw.split("</")[0]
    stop_patterns = ["[END", "[/FINAL", "[ANSWER", "[SOLUTION"]
    for pat in stop_patterns:
        if pat in raw:
            if raw.find(pat) > 5: raw = raw.split(pat)[0]
    return re.sub(r'#.*$', '', raw, flags=re.MULTILINE).strip().strip('"').strip("'")

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
    facts = [f.strip() for f in facts if f.strip()]
    if not facts:
        facts = [line[2:].strip() for line in raw.split('\n') if line.strip().startswith('- ')]
    return facts

def _prompt_generate_question_for_sentence_group(facts: List[str], model_name: str, config: dict, main_subject: str) -> str:
    # src/prompts.py에 있는 함수 사용
    prompt = generate_sentence_group_question_prompt(facts) 
    raw = generate(prompt, model_name, config)
    q = _extract_xml_tag(raw, "query")
    return q if q else f"{_clean_model_output(raw)} {main_subject}"

def prompt_reconstruct_local_sentence(original_sentence: str, updated_facts: List[str],
                                      query: str, model_name: str, config: dict) -> str:
    fact_list_str = "\n".join(f"- {f}" for f in updated_facts)
    prompt = RECONSTRUCT_LOCAL_SENTENCE_TEMPLATE.format(
        original_sentence=original_sentence,
        updated_facts=fact_list_str
    )
    raw = generate(prompt, model_name, config,
                   generation_params_override={"temperature": 0.3, "max_new_tokens": 512})
    # XML 태그 강제 부착 후 추출
    modified_raw = f"<generated_sentence>{raw}"
    return _extract_xml_tag(modified_raw, "generated_sentence") or _clean_model_output(modified_raw)

def prompt_global_polish(query: str, draft_text: str, model_name: str, config: dict) -> str:
    prompt = GLOBAL_POLISH_TEMPLATE.format(query=query, draft_text=draft_text)
    raw = generate(prompt, model_name, config,
                   generation_params_override={"temperature": 0.5, "max_new_tokens": 1024})
    modified_raw = f"<final_response>{raw}"
    return _extract_xml_tag(modified_raw, "final_response") or _clean_model_output(modified_raw)

# =============================================================================
# Ablation Processing Functions (No RAG)
# =============================================================================

def _detect_syndromes_batch_no_rag(sentence_batches: List[Dict], 
                                   model_name: str, 
                                   config: Dict,
                                   main_subject: str) -> Dict[str, Any]:
    """
    [Ablation] Self-Verification (No RAG)
    1. 질문 생성
    2. 내부 지식으로 답변 생성 (Internal Evidence) -> VERIFICATION_ANSWER_TEMPLATE 사용
    3. 1:1 대조 및 검증
    """
    clean_facts = []
    syndromes_buffer = []

    logging.info(">>> [Ablation Step 1] Self-Verification Started (Internal Knowledge Only)")
    
    for batch in tqdm(sentence_batches, desc="Detecting (No RAG)"):
        facts = batch["original_facts"]
        if not facts: continue

        # 1. 검증 질문 생성
        search_q = _prompt_generate_question_for_sentence_group(facts, model_name, config, main_subject)

        # 2. [핵심] 내부 지식으로 답변 생성 (Internal Evidence Generation)
        # RAGRetriever 대신, LLM에게 질문을 던져서 기억을 끄집어냅니다.
        prompt_internal = VERIFICATION_ANSWER_TEMPLATE.format(question=search_q)
        internal_evidence = generate(prompt_internal, model_name, config)
        
        # 3. 1:1 검증 (Internal Evidence vs Fact)
        for fact in facts:
            # 검증 프롬프트 (기존 VALIDATE_EVIDENCE_TEMPLATE 사용)
            # 여기서 evidence_text 자리에 internal_evidence가 들어갑니다.
            prompt_verify = VALIDATE_EVIDENCE_TEMPLATE.format(
                fact_text=fact, 
                evidence_text=internal_evidence
            )
            raw_output = generate(prompt_verify, model_name, config)
            
            verdict = _extract_xml_tag(raw_output, "judgment").upper()
            if not verdict: 
                verdict = "CONTRADICTED" if "CONTRADICTED" in raw_output.upper() else "SUPPORTED"

            if "SUPPORTED" in verdict:
                clean_facts.append(fact)
            else:
                # 오류로 판단
                error_package = {
                    "original_fact": fact,  
                    "evidence": internal_evidence, # 외부 문서 대신 내부 지식을 증거로 저장
                    "context": internal_evidence,  # 수정 단계에서도 이 내부 지식을 context로 씀
                    "origin_sentence": batch["sentence"]
                }
                syndromes_buffer.append(error_package)
                logging.info(f"🚫 Self-Detected Error: {fact[:30]}... (vs Internal Belief)")
    
    return {
        "clean_facts": clean_facts,
        "syndromes_buffer": syndromes_buffer
    }

def _correct_syndromes_batch_no_rag(syndromes_buffer: List[Dict], 
                                    model_name: str, 
                                    config: Dict) -> Dict[str, str]:
    """
    [Ablation] Self-Correction with BP (No RAG)
    외부 검색 없이 내부 지식(Internal Evidence)을 Context로 사용하여 연쇄 수정합니다.
    """
    fact_correction_map = {}
    if not syndromes_buffer:
        return {}

    # 1. 문장별로 오류 그룹화 (BP 적용)
    error_groups = defaultdict(list)
    for item in syndromes_buffer:
        error_groups[item["origin_sentence"]].append(item)

    logging.info(f">>> [Ablation Step 2] Self-Correction Started ({len(error_groups)} groups)")

    for sentence, items in tqdm(error_groups.items(), desc="Correcting (No RAG)"):
        # 컨텍스트는 그룹 내 첫 번째 것 사용 (내부 지식)
        context = items[0]["context"]

        # 입력 블록 생성
        error_block = ""
        for i, item in enumerate(items, 1):
            error_block += f"{i}. {item['original_fact']}\n"
        
        # XML BP 프롬프트 호출 (Main과 동일한 템플릿 사용)
        prompt = BP_CORRECTION_TEMPLATE.format(
            context=context,
            error_block=error_block
        )
        
        raw_output = generate(prompt, model_name, config)
        
        # XML 파싱
        correction_blocks = re.findall(r"<correction>(.*?)</correction>", raw_output, re.DOTALL | re.IGNORECASE)
        
        for block in correction_blocks:
            orig_match = re.search(r"<original>(.*?)</original>", block, re.DOTALL | re.IGNORECASE)
            fixed_match = re.search(r"<fixed>(.*?)</fixed>", block, re.DOTALL | re.IGNORECASE)
            
            if orig_match and fixed_match:
                clean_orig = re.sub(r'^[\d\-\.\)\s]+', '', orig_match.group(1).strip().strip("-").strip())
                clean_corr = fixed_match.group(1).strip()
                
                if clean_orig and clean_corr:
                    fact_correction_map[clean_orig] = clean_corr
                    logging.info(f"🔗 Self-Fixed: {clean_orig[:15]}... -> {clean_corr[:15]}...")
            
    return fact_correction_map

# =============================================================================
# Main SERC Loop (No RAG Version)
# =============================================================================

def SERC_NO_RAG(query: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:

    logging.info(f"--- SERC (Ablation: No RAG) Started --- Query: '{query[:60]}...'")
    
    history = {"query": query, "model_name": model_name, "steps": {}}

    # Step 1: Baseline Generation
    baseline = prompt_baseline(query, model_name, config)
    
    # Refusal Check: RAG가 없으므로 Cold Start 불가능. 거절하면 그대로 반환.
    is_refusal = (
        len(baseline) < 50 or 
        "sorry" in baseline.lower() or 
        "cannot answer" in baseline.lower() or
        "don't have information" in baseline.lower()
    )
    if is_refusal:
        logging.warning("Baseline refused. Since this is No-RAG mode, we cannot perform Cold Start.")
        return {"query": query, "final_output": baseline, "status": "refusal_no_rag"}

    history["initial_baseline"] = baseline

    # Step 2: Fact Extraction
    sentences = ph.programmatic_split_into_sentences(baseline)
    sentence_batches = []
    
    # Entity Extraction (No RAG, so use Baseline Entity or Query)
    # 여기서는 간단하게 쿼리 전체를 주제로 사용하거나 Baseline Entity 추출
    main_subject = query # Ablation에서는 단순화

    for s in sentences:
        if not s.strip(): continue
        facts = prompt_extract_facts_from_sentence(s, model_name, config, main_subject=main_subject)
        facts = [f for f in facts if len(f) > 5]
        if facts:
            sentence_batches.append({"sentence": s, "original_facts": facts})

    history["steps"]["sentence_batches"] = sentence_batches

    # Step 3: Detection (Self-Verification)
    detection_result = _detect_syndromes_batch_no_rag(
        sentence_batches=sentence_batches,
        model_name=model_name,
        config=config,
        main_subject=main_subject
    )
    
    clean_facts = detection_result["clean_facts"]
    syndromes_buffer = detection_result["syndromes_buffer"]

    # Step 4: Correction (Self-Correction BP)
    fact_correction_map = _correct_syndromes_batch_no_rag(
        syndromes_buffer=syndromes_buffer,
        model_name=model_name,
        config=config
    )

    history["steps"]["syndromes_detected"] = len(syndromes_buffer)
    history["steps"]["fact_correction_map"] = fact_correction_map

    # Step 5: Reconstruction (Conditional Zero-Base)
    logging.info("--- Reconstruction (No RAG) ---")
    local_sentences = []

    for batch in sentence_batches:
        orig_sent = batch["sentence"]
        old_facts = batch["original_facts"]
        
        updated_facts_list = []
        has_changes = False  # 변경 감지 플래그
        
        for f in old_facts:
            if f in fact_correction_map:
                updated_facts_list.append(fact_correction_map[f])
                has_changes = True
            else:
                updated_facts_list.append(f)
        
        # 변경 없으면 Skip
        if not has_changes:
            local_sentences.append(orig_sent)
            continue
            
        # 변경 있으면 새로 생성
        reconstructed = prompt_reconstruct_local_sentence(
            original_sentence=orig_sent,
            updated_facts=updated_facts_list,
            query=query,
            model_name=model_name,
            config=config
        )
        final_sent = reconstructed.strip() if reconstructed and len(reconstructed) > 10 else orig_sent
        local_sentences.append(final_sent)

    # Step 6: Global Polish
    draft = "\n\n".join(local_sentences)
    if len(draft.strip()) < 50:
        final_output = baseline
    else:
        final_output = prompt_global_polish(query=query, draft_text=draft, model_name=model_name, config=config).strip()
    
    history["final_output"] = final_output
    return history

# =============================================================================
# Execution Wrapper
# =============================================================================

def run_single_item(item: Dict[str, Any], model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    q = item.get("question") or item.get("query")
    try:
        result = SERC_NO_RAG(query=q, model_name=model_name, config=config)
        return {**item, "method_result": {"final_output": result["final_output"], "history": result, "status": "success"}}
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {**item, "method_result": {"error": str(e), "status": "error"}}

def main():
    parser = argparse.ArgumentParser(description="SERC Ablation: No RAG")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/serc_no_rag")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = os.path.join(PROJECT_ROOT, config['data_paths'][args.dataset])
    data = load_dataset(args.dataset, data_path)
    data = data[args.start: args.end]

    timestamp = get_timestamp()
    os.makedirs(os.path.join(PROJECT_ROOT, args.output_dir, args.model.replace('/', '_'), args.dataset), exist_ok=True)
    output_path = os.path.join(PROJECT_ROOT, args.output_dir, args.model.replace('/', '_'), args.dataset,
                               f"serc_no_rag_{args.start}-{len(data)+args.start}_{timestamp}.jsonl")

    results = []
    for i, item in enumerate(tqdm(data, desc="SERC No-RAG Processing")):
        results.append(run_single_item(item, args.model, config))
        if (i + 1) % args.save_interval == 0:
            save_jsonl(results, output_path)

    save_jsonl(results, output_path)
    logging.info(f"Done. Results → {output_path}")

if __name__ == "__main__":
    main()