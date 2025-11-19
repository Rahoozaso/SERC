import argparse
import os
import sys
import logging
from typing import Dict, Any, List, Optional
import re 
import traceback
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

try:
    from src import programmatic_helpers as ph
    from src.utils import load_config, save_jsonl, get_timestamp
    from src.data_loader import load_dataset
    from src import prompts 
    from src.prompts import (
        generate_sentence_group_question_prompt,
        VERIFICATION_ANSWER_TEMPLATE_RAG,
        VALIDATE_EVIDENCE_TEMPLATE, 
        CORRECT_FACT_TEMPLATE_RAG, 
        RECOMPOSE_PROMPT_TEMPLATE, 
        BASELINE_PROMPT_TEMPLATE_PN, 
        EXTRACT_FACTS_TEMPLATE_PN, 
        REWRITE_SENTENCE_TEMPLATE,
        QUERY_ENTITY_EXTRACTOR_TEMPLATE,
        BASELINE_ENTITY_EXTRACTOR_TEMPLATE,
        RAG_DOMINANT_ENTITY_TEMPLATE,
        ENTITY_CONSISTENCY_JUDGE_TEMPLATE,
        BASELINE_PROMPT_TEMPLATE_RAG_FIRST
    )
    from src.model_wrappers import generate 
    from src.rag_retriever import RAGRetriever 
    
except ImportError:
    logging.error("ImportError: Check PYTHONPATH.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- XML Parsing Helper ---
def _extract_xml_tag(text: str, tag: str) -> str:
    if not text: return ""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def _clean_model_output(raw_response: str) -> str:
    if not raw_response: return ""
    line = re.sub(r'#.*$', '', raw_response).strip()
    line = re.sub(r'\[.*?\]$', '', line).strip()
    return line.strip().strip('"').strip("'")


# --- Wrappers ---

def prompt_baseline(query: str, model_name: str, config: dict) -> str:
    prompt = prompts.BASELINE_PROMPT_TEMPLATE_PN.format(query=query)
    return generate(prompt, model_name, config)

def prompt_extract_facts_from_sentence(sentence: str, model_name: str, config: dict, main_subject: str) -> List[str]:
    prompt = prompts.EXTRACT_FACTS_TEMPLATE_PN.format(sentence=sentence, main_subject=main_subject)
    raw = generate(prompt, model_name, config)
    facts = re.findall(r"<fact>(.*?)</fact>", raw, re.DOTALL | re.IGNORECASE)
    cleaned_facts = [f.strip() for f in facts if f.strip()]
    if not cleaned_facts: 
        lines = raw.strip().split('\n')
        for line in lines:
            if line.strip().startswith('- '):
                cleaned_facts.append(line.strip()[2:].strip())
    return cleaned_facts

def prompt_validate_one_fact_against_evidence(fact_text: str, evidence_text: str, model_name: str, config: dict) -> str:
    prompt = prompts.VALIDATE_EVIDENCE_TEMPLATE.format(fact_text=fact_text, evidence_text=evidence_text)
    raw = generate(prompt, model_name, config)
    judgment = _extract_xml_tag(raw, "judgment").upper()
    if "CONTRADICTED" in judgment: return "CONTRADICTED"
    if "SUPPORTED" in judgment: return "SUPPORTED"
    if "NOT_FOUND" in judgment: return "NOT_FOUND"
    upper_raw = raw.upper()
    if "CONTRADICTED" in upper_raw: return "CONTRADICTED"
    if "SUPPORTED" in upper_raw: return "SUPPORTED"
    return "NOT_FOUND"

def prompt_generate_correct_fact(fact_text: str, model_name: str, config: dict, context: str) -> str:
    prompt = prompts.CORRECT_FACT_TEMPLATE_RAG.format(fact_text=fact_text, context=context)
    raw = generate(prompt, model_name, config)
    res = _extract_xml_tag(raw, "corrected_fact")
    return res if res else _clean_model_output(raw)

def prompt_rewrite_sentence(bad_sentence: str, correct_fact_text: str, model_name: str, config: dict, main_subject: str) -> str:
    prompt = prompts.REWRITE_SENTENCE_TEMPLATE.format(bad_sentence=bad_sentence, correct_fact_text=correct_fact_text)
    raw = generate(prompt, model_name, config)
    res = _extract_xml_tag(raw, "rewritten_sentence")
    return res if res else _clean_model_output(raw)

def _prompt_generate_question_for_sentence_group(fact_texts_list: List[str], model_name: str, config: dict, main_subject_info: str) -> str:
    prompt = prompts.generate_sentence_group_question_prompt(fact_texts_list)
    question_params = {"temperature": 0.1, "max_new_tokens": 100}
    raw_response = generate(prompt, model_name, config, generation_params_override=question_params)
    generated_query = _extract_xml_tag(raw_response, "query")
    if not generated_query:
        clean_text = raw_response
        tags = ["[SENTENCE]", "[INSTRUCTION]", "[ANSWER]"]
        for t in tags:
            if t in clean_text: clean_text = clean_text.split(t)[0]
        generated_query = clean_text.strip()
    
    # [Query Concatenation]
    clean_subject_info = main_subject_info[:100] if main_subject_info else ""
    final_query = f"{generated_query} {clean_subject_info}"
    return final_query

def _prompt_get_verification_answer(question: str, model_name: str, config: dict, context: str) -> str:
    prompt = VERIFICATION_ANSWER_TEMPLATE_RAG.format(query=question, context=context)
    raw = generate(prompt, model_name, config)
    return _clean_model_output(raw)

def prompt_recompose(query: str, final_facts_map: Dict[str, str], model_name: str, config: dict) -> str:
    fact_texts = list(final_facts_map.values())
    if not fact_texts: return "N/A"
    fact_list_str = "\n".join([f"- {f}" for f in fact_texts])
    prompt = prompts.RECOMPOSE_PROMPT_TEMPLATE.format(query=query, fact_list_str=fact_list_str)
    raw = generate(prompt, model_name, config)
    return _clean_model_output(raw)

# --- Entity Helpers ---
def prompt_extract_entity_desc(text: str, model_name: str, config: dict, is_query: bool = False) -> str:
    prompt = prompts.QUERY_ENTITY_EXTRACTOR_TEMPLATE.format(query=text) if is_query else prompts.BASELINE_ENTITY_EXTRACTOR_TEMPLATE.format(baseline_text=text)
    raw = generate(prompt, model_name, config)
    match = re.search(r"(.+?)\s*\(([^)]+)\)", raw)
    if match:
        name = match.group(1).strip()
        if name.startswith("def ") or "extract" in name.lower(): return ""
        char = match.group(2).strip()
        if len(name.split()) > 6: return ""
        return f"{name} ({char})"
    return ""

def prompt_extract_rag_desc(query: str, context: str, model_name: str, config: dict) -> str:
    prompt = prompts.RAG_DOMINANT_ENTITY_TEMPLATE.format(query=query, context=context)
    raw = generate(prompt, model_name, config)
    match = re.search(r"(.+?)\s*\(([^)]+)\)", raw)
    if match:
        name = match.group(1).strip()
        char = match.group(2).strip()
        if len(name.split()) > 6: return ""
        return f"{name} ({char})"
    return ""

def prompt_judge_entity_consistency(desc_a: str, desc_b: str, model_name: str, config: dict) -> bool:
    # Soft Match
    name_a = desc_a.split('(')[0].strip().lower()
    name_b = desc_b.split('(')[0].strip().lower()
    if name_a in name_b or name_b in name_a: return True
    
    prompt = prompts.ENTITY_CONSISTENCY_JUDGE_TEMPLATE.format(desc_a=desc_a, desc_b=desc_b)
    raw = generate(prompt, model_name, config)
    judgment = _extract_xml_tag(raw, "judgment").upper()
    if "YES" in judgment: return True
    if "NO" in judgment: return False
    if "YES" in raw.upper(): return True
    return False

def prompt_regenerate_baseline_rag(query: str, context: str, model_name: str, config: dict) -> str:
    prompt = prompts.BASELINE_PROMPT_TEMPLATE_RAG_FIRST.format(context=context, query=query)
    return generate(prompt, model_name, config)


# --- Main Logic ---
def SERC_FactInSentence_Iterative(query: str, model_name: str, config: Dict[str, Any],
                                    t_max: Optional[int] = None,
                                    max_facts_per_group: Optional[int] = None
                                    ) -> Dict[str, Any]:

    T_MAX = t_max if t_max is not None else 3
    MAX_FACTS_PER_GROUP = max_facts_per_group if max_facts_per_group is not None else 5
    
    logging.info(f"--- SERC Start --- Query: '{query[:50]}...'")

    try:
        retriever = RAGRetriever(config=config)
    except Exception as e:
        logging.error(f"Retriever Init Error: {e}")
        raise e

    history = {'query': query, 'model_name': model_name, 'cycles': []}

    # 1. Baseline
    logging.info("--- [Step 1] Baseline ---")
    current_baseline = prompt_baseline(query, model_name, config)
    history['initial_baseline'] = current_baseline
    
    # 1.5 Entity Firewall
    logging.info("--- [Step 1.5] Firewall ---")
    query_desc = prompt_extract_entity_desc(query, model_name, config, is_query=True)
    model_desc = prompt_extract_entity_desc(current_baseline, model_name, config, is_query=False)
    rag_context = retriever.retrieve(query)
    rag_desc = prompt_extract_rag_desc(query, rag_context, model_name, config)
    
    logging.info(f"  Q: {query_desc}, M: {model_desc}, R: {rag_desc}")
    main_subject_info = ""
    is_query_ambiguous = (not query_desc) or "None" in query_desc
    
    if not is_query_ambiguous:
        main_subject_info = query_desc
    else:
        if not model_desc:
            main_subject_info = rag_desc if rag_desc else query
        elif rag_desc and not prompt_judge_entity_consistency(model_desc, rag_desc, model_name, config):
            logging.warning(f"  [Hard Reset] Model != RAG. Regenerating.")
            current_baseline = prompt_regenerate_baseline_rag(query, rag_context, model_name, config)
            history['regenerated_baseline'] = current_baseline
            main_subject_info = rag_desc
        else:
            main_subject_info = model_desc

    if not main_subject_info: main_subject_info = query
    main_subject_name = main_subject_info.split('(')[0].strip()
    logging.info(f"  >> LOCKED: '{main_subject_info}'")

    final_verified_facts_map: Dict[str, str] = {}

    # 2. Iterative Cycle
    for t in range(1, T_MAX + 1):
        cycle_log = {'cycle': t, 'steps': {}, 'baseline_before_cycle': current_baseline}
        logging.info(f"\n--- [Cycle {t}] ---")

        # 2a. Fact Extraction
        sentences = ph.programmatic_split_into_sentences(current_baseline)
        all_facts = {}
        sentence_groups = []
        fid_counter = 1
        
        for s in sentences:
            if not s.strip(): continue
            facts = prompt_extract_facts_from_sentence(s, model_name, config, main_subject_name)
            valid_facts = {}
            for f in facts:
                if len(f) < 5: continue
                fid = f"f{fid_counter}"
                valid_facts[fid] = f
                all_facts[fid] = f
                final_verified_facts_map[fid] = f 
                fid_counter += 1
            if valid_facts:
                sentence_groups.append({'sentence': s, 'facts': valid_facts})
        
        cycle_log['steps']['2_fact_extraction'] = {'sentence_groups': sentence_groups}
        if not all_facts: break

        # 2b. Verification
        syndrome = {}
        validation_details = []
        
        for group in sentence_groups:
            sent_text = group['sentence']
            facts_in_group = list(group['facts'].items())
            
            for i in range(0, len(facts_in_group), MAX_FACTS_PER_GROUP):
                chunk = facts_in_group[i:i+MAX_FACTS_PER_GROUP]
                chunk_texts = [item[1] for item in chunk]
                
                search_query = _prompt_generate_question_for_sentence_group(
                    chunk_texts, model_name, config, main_subject_info
                )
                context = retriever.retrieve(search_query)
                verified_answer = _prompt_get_verification_answer(search_query, model_name, config, context)

                for fid, ftext in chunk:
                    judgment = prompt_validate_one_fact_against_evidence(ftext, verified_answer, model_name, config)
                    val_entry = {'fact_id': fid, 'text': ftext, 'judgment': judgment}
                    validation_details.append(val_entry)
                    
                    if judgment == "CONTRADICTED":
                        logging.info(f"    [CONTRADICTED] {ftext}")
                        syndrome[fid] = {'fact_text': ftext, 'original_sentence': sent_text, 'evidence_docs': context}
                        if fid in final_verified_facts_map: del final_verified_facts_map[fid]

        cycle_log['steps']['2_verification'] = validation_details
        
        # 2c. Correction
        if not syndrome:
            history['termination_reason'] = 'converged'
            break
            
        logging.info(f"  Correcting {len(syndrome)} facts...")
        new_baseline = current_baseline
        correction_log = []
        
        for fid, err_data in syndrome.items():
            if err_data['original_sentence'] not in new_baseline: continue
            corrected_fact = prompt_generate_correct_fact(err_data['fact_text'], model_name, config, err_data['evidence_docs'])
            if not corrected_fact: continue
            
            final_verified_facts_map[fid] = corrected_fact
            rewritten = prompt_rewrite_sentence(err_data['original_sentence'], corrected_fact, model_name, config, main_subject_name)
            
            if rewritten and len(rewritten) > 10:
                new_baseline = new_baseline.replace(err_data['original_sentence'], rewritten)
                correction_log.append({'fid': fid, 'rewritten': rewritten})
        
        cycle_log['steps']['5_correction'] = correction_log
        current_baseline = new_baseline
        cycle_log['baseline_after_cycle'] = current_baseline
        history['cycles'].append(cycle_log)

    # 3. Final Recomposition
    logging.info("\n--- Final Recomposition ---")
    if not final_verified_facts_map:
        final_output = current_baseline
    else:
        final_output = prompt_recompose(query, final_facts_map=final_verified_facts_map, model_name=model_name, config=config)

    history['final_baseline'] = final_output
    return history

def run_single_item_wrapper(item: Dict[str, Any], model_name: str, config: Dict[str, Any], t_max: int) -> Dict[str, Any]:
    try:
        q = item.get('question', item.get('query'))
        serc_history = SERC_FactInSentence_Iterative(query=q, model_name=model_name, config=config, t_max=t_max)
        result = {'serc_result': serc_history, 'final_output': serc_history.get('final_baseline', ''), 'status': 'success'}
    except Exception as e:
        logger.error(f"Error processing '{item.get('query')}': {e}", exc_info=True)
        result = {"error": str(e), "status": "error"}
    return {**item, "method_result": result}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/fact_in_sentence_iterative_rag")
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--t_max", type=int, default=3)
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Config Load Error: {e}")
        return

    dataset_path = os.path.join(PROJECT_ROOT, config['data_paths'][args.dataset])
    try:
        data = load_dataset(args.dataset, dataset_path)
    except Exception as e:
        logger.error(f"Dataset Load Error: {e}")
        return

    start = args.start
    end = args.end if args.end is not None else len(data)
    data_slice = data[start:end]
    
    logging.info(f"Processing {len(data_slice)} items.")

    timestamp = get_timestamp()
    results_dir = os.path.join(PROJECT_ROOT, args.output_dir, args.model.replace('/', '_'), args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"serc_revised_t{args.t_max}_{start}-{end}_{timestamp}.jsonl")

    results = []
    for item in tqdm(data_slice):
        res = run_single_item_wrapper(item, args.model, config, args.t_max)
        results.append(res)
        
    save_jsonl(results, output_path)
    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()