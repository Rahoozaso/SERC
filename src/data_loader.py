import os
import json
import logging
import csv
from typing import List, Dict, Any
import pandas as pd

try:
    # When running normally (e.g., called from experiments/run_experiment.py)
    from .utils import load_jsonl, save_jsonl
except ImportError:
    import sys
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    from src.utils import load_jsonl, save_jsonl

# Logging configuration
logger = logging.getLogger(__name__)
# Default handler setup (to show logs when running script directly)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


# --- Loader Functions per Benchmark ---

def _load_longform_biographies(file_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading Longform Biographies prompts: {file_path}")
    processed_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                entity_name = ""
                item_data = {} # To store other data from original .jsonl

                # Check if it's a .jsonl file (or .txt)
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        # Search keys used in FActScore/HalluLens LongWiki, etc.
                        potential_keys = ['topic', 'entity', 'name', 'title', 'prompt']
                        for key in potential_keys:
                            if key in item:
                                entity_name = item.get(key)
                                item_data = {k:v for k,v in item.items() if k != key}
                                break
                        if not entity_name:
                             logger.warning(f"Could not find entity key in JSONL line {i+1}: {line}")
                             continue
                    else: # JSONL but not a dictionary
                         entity_name = str(item) # Convert to string just in case
                except json.JSONDecodeError:
                    # Simple .txt file (only entity name per line)
                    entity_name = line

                if entity_name: 
                    # Prompt format used in CoVe paper (Section 4.1.4)
                    query = f"Tell me a bio of {entity_name}"
                    
                    processed_item = {
                        'query': query, # Standardized 'query' key
                        'topic': entity_name, # 'topic' needed for FactScore evaluation
                        **item_data # Remaining fields if it was .jsonl
                    }
                    processed_data.append(processed_item)

    except FileNotFoundError:
         logger.error(f"Entity file ({file_path}) not found.")
         raise
    except Exception as e:
         logger.error(f"Error processing entity file ({file_path}): {e}", exc_info=True)
         raise
             
    logger.info(f"Generated {len(processed_data)} prompts from Longform Biographies ({file_path}).")
    return processed_data

def _load_hallulens_precisewikiqa_prompts(file_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading HalluLens PreciseWikiQA prompts: {file_path}")
    if not file_path.endswith(".jsonl"):
        logger.warning(f"HalluLens PreciseWikiQA prompt file format might not be JSONL: {file_path}")
        
    data = load_jsonl(file_path) # Using function from utils.py
    processed_data = []
    
    if not data or 'question' not in data[0]:
         raise ValueError(f"'question' key missing in PreciseWikiQA prompt file: {file_path}")

    for item in data:
        query_text = item.get('question')
        if query_text:
             # Standardize answer to list format for evaluation
             ground_truth_answer = item.get('answer') # Expecting single string for original answer
             answers_list = [ground_truth_answer] if ground_truth_answer is not None else []

             processed_item = {
                 'query': query_text, # Standardized 'query' key
                 'answers': answers_list, # Maintain answer list for evaluation
                 **{k:v for k,v in item.items() if k not in ['question', 'answer']} # Remaining fields
             }
             processed_data.append(processed_item)
    logger.info(f"Processed {len(processed_data)} items from HalluLens PreciseWikiQA prompts ({file_path}).")
    return processed_data

def _load_truthfulqa(file_path: str) -> List[Dict[str, Any]]:
    """Loading TruthfulQA dataset (Assuming CSV format)"""
    logger.info(f"Loading TruthfulQA ({file_path})...")
    if not file_path.endswith(".csv"):
        logger.warning(f"TruthfulQA file format might not be CSV: {file_path}")

    processed_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Check CSV header
            fieldnames = reader.fieldnames
            if fieldnames is None:
                 raise ValueError("TruthfulQA CSV file is empty or header cannot be read.")
            
            expected_columns = ['Question', 'Correct Answers', 'Incorrect Answers']
            if not all(col in fieldnames for col in expected_columns):
                 raise ValueError(f"Essential columns ({expected_columns}) missing in TruthfulQA CSV file. Current columns: {fieldnames}")

            for row in reader:
                question_text = row.get('Question')
                if question_text and question_text.strip():
                    processed_item = {
                        'query': question_text.strip(), # Standardized 'query' key
                        # Store original fields needed for evaluation
                        'truthfulqa_type': row.get('Type'),
                        'truthfulqa_category': row.get('Category'),
                        'correct_answers_truthfulqa': row.get('Correct Answers'),
                        'incorrect_answers_truthfulqa': row.get('Incorrect Answers'),
                        'source': row.get('Source'),
                        'original_row': row # Store entire original row (Optional)
                    }
                    processed_data.append(processed_item)
                else:
                     logger.warning("Found empty 'Question' field in TruthfulQA row, skipping.")

        logger.info(f"Processed {len(processed_data)} questions from TruthfulQA ({file_path}).")
        return processed_data
    except Exception as e:
         logger.error(f"Error occurred while processing TruthfulQA CSV file: {e}", exc_info=True)
         raise


def load_dataset(dataset_name: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Calls appropriate loader function based on dataset name (key from config.yaml).
    """
    logger.info(f"Attempting to load dataset: {dataset_name} from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found: {file_path}")
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    loader_func = None
    name_lower = dataset_name.lower() 

    if 'longform_bio' in name_lower:
        loader_func = _load_longform_biographies
    elif 'hallulens_precisewikiqa' in name_lower:
        loader_func = _load_hallulens_precisewikiqa_prompts
    elif 'truthfulqa' in name_lower:
        loader_func = _load_truthfulqa
    else:
        logger.warning(f"No specific loader found for '{dataset_name}'. Attempting default loading based on file extension.")
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".jsonl":
             data = load_jsonl(file_path)
        elif file_extension == ".json":
             with open(file_path, 'r', encoding='utf-8') as f:
                 data = json.load(f)
             if not isinstance(data, list):
                 raise ValueError("Default JSON loader only supports list format.")
        elif file_extension == ".csv":
             data = pd.read_csv(file_path).to_dict('records')
        elif file_extension == ".txt":
             logger.info(".txt file detected. Using Longform Biographies loader.")
             loader_func = _load_longform_biographies
        else:
             raise ValueError(f"Unsupported file extension: {file_extension}")
        
        if loader_func is None:
             if data and not any(key in data[0] for key in ['question', 'query']):
                 logger.warning(f"First item of loaded data does not have 'question' or 'query' key.")
             return data
        
    try:
        data = loader_func(file_path)

        if not data:
            logger.warning(f"Data load result is empty: {file_path}")
            return [] 
        elif not isinstance(data[0], dict):
            raise ValueError("Loaded data item is not in dictionary format.")
        elif not any(key in data[0] for key in ['question', 'query']):
            logger.warning(f"First item of loaded data does not have 'question' or 'query' key: {list(data[0].keys())}")        
        return data

    except Exception as e:
        logger.error(f"Error occurred while loading '{dataset_name}' dataset ({file_path}): {e}", exc_info=True)
        raise 

if __name__ == '__main__':
    logger.info("--- Data Loader Test Start ---")
    test_dir = "_temp_test_data_loader"
    os.makedirs(test_dir, exist_ok=True)

    try:
        test_file_bio_txt = os.path.join(test_dir, "temp_bio_entities.txt")
        test_data_bio_txt = "Leonardo da Vinci\nMarie Curie"
        with open(test_file_bio_txt, 'w', encoding='utf-8') as f:
            f.write(test_data_bio_txt)
        
        loaded_bio = load_dataset("longform_bio", test_file_bio_txt)
        assert len(loaded_bio) == 2 and 'query' in loaded_bio[0]
        assert loaded_bio[0]['query'] == 'Tell me a bio of Leonardo da Vinci'
        assert 'topic' in loaded_bio[0] and loaded_bio[0]['topic'] == 'Leonardo da Vinci'
        logger.info("[Success] Longform Biographies (.txt) Loader Test")

        test_file_pqa = os.path.join(test_dir, "temp_precisewikiqa_prompts.jsonl")
        test_data_pqa = [{'question': 'Who are politicians born in Boston?', 'answer': 'John F. Kennedy', 'id': 'pqa1'}]
        save_jsonl(test_data_pqa, test_file_pqa) 
        loaded_pqa = load_dataset("hallulens_precisewikiqa", test_file_pqa)
        assert len(loaded_pqa) == 1 and loaded_pqa[0]['query'] == 'Who are politicians born in Boston?'
        assert 'answers' in loaded_pqa[0] and loaded_pqa[0]['answers'] == ['John F. Kennedy']
        logger.info("[Success] HalluLens PreciseWikiQA Loader Test")
        test_file_tqa = os.path.join(test_dir, "temp_TruthfulQA.csv")
        test_data_tqa = [
            {'Type': 'Misconceptions', 'Category': 'Health', 'Question': 'Does sugar make kids hyper?', 'Correct Answers': 'No...', 'Incorrect Answers': 'Yes...'},
            {'Type': 'Myths', 'Category': 'Animals', 'Question': 'Do goldfish have a 3-second memory?', 'Correct Answers': 'No...', 'Incorrect Answers': 'Yes...'}
        ]
        with open(test_file_tqa, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_data_tqa[0].keys())
            writer.writeheader()
            writer.writerows(test_data_tqa)
        loaded_tqa = load_dataset("truthfulqa", test_file_tqa)
        assert len(loaded_tqa) == 2 and loaded_tqa[0]['query'] == 'Does sugar make kids hyper?'
        assert 'correct_answers_truthfulqa' in loaded_tqa[0]
        logger.info("[Success] TruthfulQA Loader Test")

    except AssertionError as e:
        logger.error(f"Data Loader Test Failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error occurred during Data Loader Test: {e}", exc_info=True)
    finally:
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            logger.info(f"Deleted temporary test directory: {test_dir}")