import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import re
from num2words import num2words
from word2number import w2n
from datetime import datetime
from dateparser import parse
from resume_lines_and_labels import dataframe_builder

model = SentenceTransformer('all-MiniLM-L6-v2')

def section_generator(resume_lines):
    df = dataframe_builder(resume_lines)
    section_dict = {
        "Education": [],
        "Experience": [],
        "Skills": [],
        "Certifications": [],
        "Projects": [],
        "Other": []
    }
    for _, row in df.iterrows():
        label = row["label"]
        text = row["text"]
        if label in section_dict:
            section_dict[label].append(text)
        else:
            section_dict["Other"].append(text)
    return section_dict

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

def expand_keywords_with_synonyms(keywords: list[str]) -> list[str]:
    expanded = set()
    for word in keywords:
        expanded.add(word)
        expanded.update(get_synonyms(word))
    return list(expanded)

def convert_words_to_digits(text: str) -> str:
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
            return match.group()
    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)

def convert_digits_to_words(text: str) -> str:
    def replacer(match):
        num = int(match.group())
        return num2words(num)
    return re.sub(r'\b\d+\b', replacer, text)

def phrase_matches_skill(phrase: str, skill_keyword: str, threshold: float = 0.3) -> tuple[bool, float]:
    embeddings = model.encode([skill_keyword, phrase], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return score >= threshold, round(score, 3)

def section_checking(resume_section: list[str], jd_section: list[str], threshold: float = 0.3) -> float:
    if not jd_section:
        return 1.0
    expanded_jd_section = expand_keywords_with_synonyms(jd_section)
    match_count = 0
    for jd_skill in expanded_jd_section:
        for resume_item in resume_section:
            is_match, _ = phrase_matches_skill(resume_item, jd_skill, threshold)
            if is_match:
                match_count += 1
                break
    return round(match_count / len(jd_section), 2)

def degree_checking(resume_education_section: list[str], jd_degree: str) -> bool:
    degree_keywords = {
        "bachelor": ["bachelor", "bachelors", "b.sc", "bs", "ba", "btech", "b.e"],
        "master": ["master", "masters", "m.sc", "ms", "ma", "mtech", "m.e", "mba"],
        "phd": ["phd", "ph.d", "doctorate", "doctoral"],
        "associate": ["associate", "associates"],
        "diploma": ["diploma", "certificate program"]
    }
    jd_degree = jd_degree.lower().strip()
    if jd_degree not in degree_keywords:
        return False
    resume_text = " ".join(resume_education_section).lower()
    for variant in degree_keywords[jd_degree]:
        if variant in resume_text:
            return True
    return False

def experience_checking_combined(resume_lines: list[str], jd_experience_section: float) -> tuple[bool, float]:
    total_months = 0
    for line in resume_lines:
        line = convert_words_to_digits(line.lower())
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*(years?|yrs?|months?)', line)
        for value, unit in matches:
            num = float(value)
            if 'month' in unit:
                num /= 12
            total_months += num * 12
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

def final_score_checking(resume: dict, jd: dict) -> dict:
    from sentence_transformers.util import cos_sim

    def batch_score(resume_lines, jd_keywords, threshold=0.3):
        if not resume_lines or not jd_keywords:
            return 0.0
        resume_embeds = model.encode(resume_lines, convert_to_tensor=True)
        jd_embeds = model.encode(jd_keywords, convert_to_tensor=True)
        sim_matrix = cos_sim(jd_embeds, resume_embeds)  # shape [len(jd), len(resume)]
        match_count = 0
        for row in sim_matrix:
            if any(score >= threshold for score in row):
                match_count += 1
        return round(match_count / len(jd_keywords), 2)

    total_score = 0
    num_sections = 0
    experience_time = None

    # Expand all keyword groups
    def expand(jd_list): return expand_keywords_with_synonyms(jd_list)

    tech_score = batch_score(resume.get("Skills", []), expand(jd.get("skills_required", [])))
    soft_score = batch_score(resume.get("Skills", []), expand(jd.get("soft_skills", [])))
    lang_score = batch_score(resume.get("Skills", []), expand(jd.get("languages", [])))

    total_score += tech_score + soft_score + lang_score
    num_sections += 3

    exp_match = True
    deg_match = True

    if jd.get("min_experience"):
        exp_match, experience_time = experience_checking_combined(resume.get("Experience", []), jd["min_experience"])
        if not exp_match:
            return {
                "total_score": 0.0,
                "tech_skill_match": tech_score,
                "soft_skill_match": soft_score,
                "language_match": lang_score,
                "degree_match": False,
                "experience_match": False,
                "experience_years": experience_time
            }

    if jd.get("degree"):
        deg_match = degree_checking(resume.get("Education", []), jd["degree"])
        if not deg_match:
            return {
                "total_score": 0.0,
                "tech_skill_match": tech_score,
                "soft_skill_match": soft_score,
                "language_match": lang_score,
                "degree_match": False,
                "experience_match": True,
                "experience_years": experience_time
            }

    # Top keyword matching
    keyword_score = 0
    if jd.get("top_keywords"):
        all_resume_lines = [line for section_lines in resume.values() for line in section_lines]
        top_kw = [kw for kw, _ in jd["top_keywords"]]
        keyword_score = batch_score(all_resume_lines, expand(top_kw))
        total_score += keyword_score
        num_sections += 1

    final_score = round((total_score / num_sections), 2) if num_sections else 0.0

    return {
        "total_score": final_score,
        "tech_skill_match": tech_score,
        "soft_skill_match": soft_score,
        "language_match": lang_score,
        "degree_match": True,
        "experience_match": True,
        "experience_years": experience_time
    }
