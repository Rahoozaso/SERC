import re
from typing import List, Dict, Tuple
import nltk
import logging # Added for logging

# --- NLTK Sentence Tokenizer Initialization ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    try:
        nltk.download('punkt')
        print("'punkt' downloaded successfully.")
    except Exception as e:
        logging.warning(f"Failed to download NLTK 'punkt'. Falling back to simple split. Error: {e}")
        # Set flag if NLTK download fails
        _nltk_punkt_available = False
    else:
        _nltk_punkt_available = True
else:
    _nltk_punkt_available = True

def programmatic_split_into_sentences(text: str) -> List[str]:
    """
    Splits text into a list of sentences.
    Attempts to use NLTK, falling back to simple punctuation-based splitting on failure.
    """
    if not text:
        return []

    if _nltk_punkt_available:
        try:
            sentences = nltk.sent_tokenize(text.strip())
            # Remove empty strings and strip whitespace from NLTK results
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logging.warning(f"NLTK sent_tokenize failed: {e}. Falling back to simple split.")
            # Use simple split on NLTK failure (safety fallback)

    # Simple split if NLTK is unavailable or failed
    # Split based on !, ?, . followed by whitespace (can be extended)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Remove empty strings and strip whitespace from results
    return [s.strip() for s in sentences if s.strip()]

def programmatic_parse_fact_list(list_str: str) -> List[str]:
    """
    Parses a string of facts returned by the model into a list of strings.
    Handles 'None', newline-separated lists, numbered lists, and bulleted lists.
    """
    if list_str is None or list_str.strip().lower() == "none" or not list_str.strip():
        return []
    
    regex = re.compile(r'^\s*(?:(?:\d+\.)|\-)\s+(.*)', re.MULTILINE)
    facts = regex.findall(list_str)

    if facts:
        return [f.strip() for f in facts if f.strip()]
    else:
        return [line.strip() for line in list_str.splitlines() if line.strip()]
    
def programmatic_replace(baseline: str, bad_sentence: str, good_sentence: str) -> str:
    """
    Replaces the first occurrence of 'bad_sentence' with 'good_sentence'
    within the baseline text exactly once.
    """
    if not baseline or not bad_sentence:
        return baseline
    # Use count=1 to replace only the first occurrence
    # Returns the baseline unchanged if bad_sentence is not found
    return baseline.replace(bad_sentence, good_sentence, 1)

def programmatic_group_facts_by_tag(fact_tags: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Groups a {fact_id: tag} dictionary into a {tag: [fact_id_1, ...]} dictionary.
    Cleans the tag strings before grouping (lowercase, strip whitespace/punctuation).
    """
    groups: Dict[str, List[str]] = {}
    if not fact_tags:
        return groups

    for fi, tag in fact_tags.items():
        # Tag cleaning: lowercase, strip whitespace, strip trailing punctuation
        # If tag is empty or None, classify as 'misc' (miscellaneous, translated from "기타")
        cleaned_tag = tag.strip().strip('.!?').lower() if tag and tag.strip() else "misc"

        if cleaned_tag not in groups:
            groups[cleaned_tag] = []
        groups[cleaned_tag].append(fi) # Add fact ID (e.g., "f1")
    return groups

def programmatic_chunk_groups(fact_groups: Dict[str, List[str]], max_size: int) -> Dict[str, List[str]]:
    """
    If the list of fact IDs in a group exceeds max_size,
    this function splits the group into multiple smaller chunks.
    E.g.: {"Work": [f1-f7]} (max_size=5) -> {"Work_1": [f1-f5], "Work_2": [f6, f7]}
    """
    if max_size <= 0:
        logging.warning("max_size must be positive. Returning original groups.")
        return fact_groups

    chunked_groups: Dict[str, List[str]] = {}
    for tag, fact_ids_list in fact_groups.items():
        if len(fact_ids_list) <= max_size:
            # If group size is within the limit, add it as is
            chunked_groups[tag] = fact_ids_list
        else:
            # If group is too large, split it into chunks of max_size
            chunk_num = 1
            for i in range(0, len(fact_ids_list), max_size):
                chunk = fact_ids_list[i : i + max_size]
                # Create new group names (e.g., "Work_1", "Work_2")
                chunked_groups[f"{tag}_{chunk_num}"] = chunk
                chunk_num += 1
    return chunked_groups