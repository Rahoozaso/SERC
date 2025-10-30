import re
from typing import List, Dict, Tuple
import nltk
import logging # 로깅 추가

# --- NLTK 문장 토크나이저 초기화 ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    try:
        nltk.download('punkt')
        print("'punkt' downloaded successfully.")
    except Exception as e:
        logging.warning(f"Failed to download NLTK 'punkt'. Falling back to simple split. Error: {e}")
        # NLTK 다운로드 실패 시 플래그 설정
        _nltk_punkt_available = False
    else:
        _nltk_punkt_available = True
else:
    _nltk_punkt_available = True

def programmatic_split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 리스트로 분할합니다.
    NLTK 사용을 시도하고, 실패 시 간단한 구두점 기반 분할로 대체합니다.
    """
    if not text:
        return []

    if _nltk_punkt_available:
        try:
            sentences = nltk.sent_tokenize(text.strip())
            # NLTK 결과에서 빈 문자열 제거 및 공백 정리
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logging.warning(f"NLTK sent_tokenize failed: {e}. Falling back to simple split.")
            # NLTK 실패 시 간단한 분할 사용 (안전 장치)

    # NLTK 사용 불가 또는 실패 시 간단한 분할
    # !, ?, . 뒤에 공백이 오는 경우를 기준으로 분할 (더 많은 경우 추가 가능)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # 결과에서 빈 문자열 제거 및 공백 정리
    return [s.strip() for s in sentences if s.strip()]

def programmatic_parse_fact_list(list_str: str) -> List[str]:
    """
    모델이 반환한 줄바꿈(\n)으로 구분된 사실 목록 문자열을
    문자열 리스트로 파싱합니다. '없음' 응답은 빈 리스트를 반환합니다.
    """
    if list_str is None or list_str.strip().lower() == "없음" or not list_str.strip():
        return []
    # 각 줄의 앞뒤 공백 제거, 빈 줄은 제외
    return [line.strip() for line in list_str.splitlines() if line.strip()]

def programmatic_replace(baseline: str, bad_sentence: str, good_sentence: str) -> str:
    """
    텍스트(baseline) 내에서 첫 번째로 발견되는 'bad_sentence'를 'good_sentence'로
    정확히 한 번만 치환합니다.
    """
    if not baseline or not bad_sentence:
        return baseline
    # 정확히 1번만 치환하도록 count=1 사용
    # bad_sentence가 baseline에 없으면 변경 없이 반환됨
    return baseline.replace(bad_sentence, good_sentence, 1)

def programmatic_group_facts_by_tag(fact_tags: Dict[str, str]) -> Dict[str, List[str]]:
    """
    {fact_id: tag} 딕셔너리를 {tag: [fact_id_1, fact_id_2, ...]} 딕셔너리로 그룹화합니다.
    태그 문자열을 정제하여 사용합니다 (소문자 변환, 앞뒤 공백/구두점 제거).
    """
    groups: Dict[str, List[str]] = {}
    if not fact_tags:
        return groups

    for fi, tag in fact_tags.items():
        # 태그 정제: 소문자 변환, 양 끝 공백 제거, 끝 구두점(.) 제거
        # 태그가 없거나 비어있으면 '기타' 그룹으로 분류
        cleaned_tag = tag.strip().strip('.!?').lower() if tag and tag.strip() else "기타"

        if cleaned_tag not in groups:
            groups[cleaned_tag] = []
        groups[cleaned_tag].append(fi) # 사실 ID (e.g., "f1") 추가
    return groups

def programmatic_chunk_groups(fact_groups: Dict[str, List[str]], max_size: int) -> Dict[str, List[str]]:
    """
    그룹 내 사실 ID 리스트의 길이가 max_size를 초과하면,
    그룹을 여러 개의 작은 그룹으로 분할 (Chunking) 합니다.
    예: {"작품": [f1~f7]} (max_size=5) -> {"작품_1": [f1~f5], "작품_2": [f6, f7]}
    """
    if max_size <= 0:
        logging.warning("max_size must be positive. Returning original groups.")
        return fact_groups

    chunked_groups: Dict[str, List[str]] = {}
    for tag, fact_ids_list in fact_groups.items():
        if len(fact_ids_list) <= max_size:
            # 그룹 크기가 최대 크기 이하이면 그대로 추가
            chunked_groups[tag] = fact_ids_list
        else:
            # 그룹이 너무 크면, max_size 단위로 쪼개기
            chunk_num = 1
            for i in range(0, len(fact_ids_list), max_size):
                chunk = fact_ids_list[i : i + max_size]
                # 새 그룹 이름 생성 (e.g., "작품_1", "작품_2")
                chunked_groups[f"{tag}_{chunk_num}"] = chunk
                chunk_num += 1
    return chunked_groups