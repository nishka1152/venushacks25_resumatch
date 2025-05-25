import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

import re
from num2words import num2words
from word2number import w2n

def convert_words_to_digits(text: str) -> str:
    """
    Converts number words (e.g., 'four', 'twenty one') to digits (e.g., 4, 21) in a sentence.
    
    Args:
        text (str): The input sentence.

    Returns:
        str: Sentence with number words replaced by digits.
    """
    # Regex to match one or more word characters possibly representing numbers
    pattern = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|' \
              r'ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|' \
              r'seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|' \
              r'sixty|seventy|eighty|ninety|hundred|thousand|million)+' \
              r'(?:\s(?:and\s)?(?:zero|one|two|three|four|five|six|seven|eight|nine|' \
              r'ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|' \
              r'seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|' \
              r'sixty|seventy|eighty|ninety|hundred|thousand|million))*\b'

    def replacer(match):
        try:
            return str(w2n.word_to_num(match.group()))
        except ValueError:
            return match.group()  # fallback if it's not a valid number phrase

    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)

def convert_digits_to_words(text: str) -> str:
    """
    Converts all standalone digits in a string to their text equivalents.
    
    Args:
        text (str): The input sentence or phrase.

    Returns:
        str: The sentence with digits replaced by words.
    """
    def replacer(match):
        num = int(match.group())
        return num2words(num)

    return re.sub(r'\b\d+\b', replacer, text)

def phrase_matches_skill(phrase: str, skill_keyword: str, threshold: float = 0.3) -> tuple[bool, float]:
    """
    Checks if a resume phrase semantically implies a given skill.

    Args:
        phrase (str): The resume line or phrase to check.
        skill_keyword (str): The skill word (e.g., 'collaboration', 'leadership').
        threshold (float): Similarity score threshold to consider it a match.

    Returns:
        (bool, float): Tuple where bool is match status, and float is similarity score.
    """
    embeddings = model.encode([skill_keyword, phrase], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return score >= threshold, round(score, 3)

#line = "Collaborated with cross-functional teams on product design"
#skill = "collaboration"

#is_match, similarity = phrase_matches_skill(line, skill)


def section_checking(resume_section: list[str], jd_section: list[str], threshold: float = 0.3) -> float:
    """
    Calculates how many JD items are semantically matched in the resume section.

    Args:
        resume_section (list[str]): List of phrases from the resume (skills/tools).
        jd_section (list[str]): List of required skills/items from the JD.
        threshold (float): Cosine similarity threshold to consider a match.

    Returns:
        float: Proportion of JD items that are matched (0.0–1.0)
    """
    if not jd_section:
        return 1.0  # If no JD requirements, assume full match
    
    match_count = 0

    for jd_skill in jd_section:
        for resume_item in resume_section:
            is_match, _ = phrase_matches_skill(resume_item, jd_skill, threshold)
            if is_match:
                match_count += 1
                break  # Stop after first match per JD skill

    return round(match_count / len(jd_section), 2)


def section_checking(resume_section: list[str], jd_section: list[str], threshold: float = 0.3) -> float:
    """
    Calculates the proportion of JD items semantically matched in the resume section.

    Args:
        resume_section (list[str]): List of strings from the resume (skills, tools, languages).
        jd_section (list[str]): List of required items from job description.
        threshold (float): Cosine similarity threshold for semantic match.

    Returns:
        float: Match score between 0.0 and 1.0 (e.g., 0.75 means 75% matched).
    """
    if not jd_section:
        return 1.0  # If the JD has no required skills, treat as full match

    match_count = 0

    for jd_skill in jd_section:
        for resume_item in resume_section:
            is_match, _ = phrase_matches_skill(resume_item, jd_skill, threshold)
            if is_match:
                match_count += 1
                break  # Only need one match per JD item

    return round(match_count / len(jd_section), 2)

def degree_checking(resume_education_section: list[str], jd_degree: str) -> bool:
    """
    Checks if the resume education section satisfies the required degree keyword.

    Args:
        resume_education_section (list[str]): Lines from resume related to education.
        jd_degree (str): Required degree keyword (e.g., 'bachelor', 'masters', 'phd').

    Returns:
        bool: True if a matching degree is found in the resume.
    """
    degree_keywords = {
        "bachelor": ["bachelor", "bachelors", "b.sc", "bs", "ba", "btech", "b.e"],
        "master": ["master", "masters", "m.sc", "ms", "ma", "mtech", "m.e", "mba"],
        "phd": ["phd", "ph.d", "doctorate", "doctoral"],
        "associate": ["associate", "associates"],
        "diploma": ["diploma", "certificate program"]
    }

    # Normalize JD degree
    jd_degree = jd_degree.lower().strip()

    # Check if it exists in the keyword map
    if jd_degree not in degree_keywords:
        return False  # not a supported degree keyword

    # Flatten resume lines to lowercase
    resume_text = " ".join(resume_education_section).lower()

    # Look for any of the degree variants
    for variant in degree_keywords[jd_degree]:
        if variant in resume_text:
            return True

    return False






        

    



    
    


        




            


