import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
from jd_parser import extract_job_features
from resume_parser_ml_model import section_generator
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

 #jd_text = input() ??


from datetime import datetime
from dateparser import parse

def experience_checking_combined(resume_lines: list[str], jd_experience_section: float) -> tuple[bool, float]:
    """
    Combines phrase-based and date range-based experience detection.

    Args:
        resume_lines (list[str]): Resume lines (entire resume or experience section).
        jd_experience_section (float): Minimum required experience in years.

    Returns:
        (bool, float): Whether candidate meets the requirement, and total years of experience found.
    """

    total_months = 0

    # -------- 1. Phrase-based extraction (e.g. "3 years", "six months") -------- #
    for line in resume_lines:
        # Convert word-numbers to digits
        line = convert_words_to_digits(line.lower())

        # Match things like "3 years", "1.5 yrs", "12 months", etc.
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*(years?|yrs?|months?)', line)
        for value, unit in matches:
            num = float(value)
            if 'month' in unit:
                num /= 12
            total_months += num * 12

    # -------- 2. Date range-based extraction (e.g. "Jan 2020 to Feb 2022") -------- #
    pattern = re.compile(r'([A-Za-z]{3,9}\s*\d{4})\s*(?:-|to|–)\s*([A-Za-z]{3,9}\s*\d{4}|present)', re.IGNORECASE)

    for line in resume_lines:
        for match in pattern.findall(line):
            start_str, end_str = match
            start_date = parse(start_str)
            end_date = parse(end_str) if 'present' not in end_str.lower() else datetime.today()

            if start_date and end_date and end_date > start_date:
                months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                total_months += months

    total_years = round(total_months / 12, 2)
    return total_years >= jd_experience_section, total_years


def final_score_checking(resume: dict, jd: dict) -> tuple[float, float | None]:
    """
    Computes the final resume score by comparing resume sections with JD requirements.

    Args:
        resume (dict): Resume sections generated by section_generator().
        jd (dict): Extracted JD fields from extract_job_features().

    Returns:
        (score, experience_time): Final match score (0.0–1.0) and years of experience (if found).
    """
    total_score = 0
    num_sections = 0
    experience_time = None

    # Section-by-section match
    for section in jd:
        if section in ('skills_required', 'soft_skills', 'languages'):
            if resume.get('Skills') and jd[section]:
                total_score += section_checking(resume['Skills'], jd[section])
                num_sections += 1

        elif section == 'min_experience' and jd[section]:
            match, experience_time = experience_checking_combined(resume.get('Experience', []), jd[section])
            if not match:
                return 0.0, experience_time  # Hard fail: min experience not met

        elif section == 'degree' and jd[section]:
            match = degree_checking(resume.get('Education', []), jd[section])
            if not match:
                return 0.0, experience_time  # Hard fail: degree not met

        elif section == 'top_keywords':
            all_resume_lines = [line for section_lines in resume.values() for line in section_lines]
            total_score += section_checking(all_resume_lines, jd[section])
            num_sections += 1

    # Optionally include projects, certifications
    for extra in ('Projects', 'Certifications'):
        if extra in resume and resume[extra]:
            total_score += section_checking(resume[extra], jd.get('skills_required', []))
            num_sections += 1

    final_score = round(total_score / num_sections, 2) if num_sections else 0.0

    return (final_score, experience_time)


    
        


