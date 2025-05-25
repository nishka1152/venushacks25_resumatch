import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

# --- Static Lists ---
tech_skills = {
    "python", "java", "sql", "c++", "c", "javascript", "django", "flask", "html", "css",
    "rest", "git", "linux", "tensorflow", "tableau", "power bi", "react", "node", "docker", "aws", "azure"
}
soft_skills = {
    "communication", "leadership", "teamwork", "collaboration", "problem-solving", "adaptability", "analytical"
}
languages = {
    "python", "java", "c++", "javascript", "c", "go", "rust", "typescript", "r", "scala", "sql"
}
degree_patterns = [
    r"(bachelor[’']?s?|b\.?sc\.?)",
    r"(master[’']?s?|m\.?sc\.?)",
    r"(ph\.?d|phd)",
]

# --- Model for embeddings (loaded once globally) ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_job_features(jd_text: str) -> dict:
    # Tokenization
    sentences = sent_tokenize(jd_text)
    words = [word.lower() for word in word_tokenize(jd_text) if word.isalpha()]

    # TF-IDF keywords
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([jd_text])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    top_n = 10
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    top_keywords = [(feature_array[i], round(tfidf_scores[i], 3)) for i in top_indices]

    # Sentence embeddings
    _ = embedding_model.encode(sentences)  # Ready if you use it later

    # Job title extraction
    match = re.search(
        r"we are (looking for|seeking|hiring) (an?|the)? ?([a-z ]+?)( with| who| that|$)",
        jd_text.lower()
    )
    job_title = match.group(3).strip() if match else None

    # Build extracted dict
    extracted = {
        "job_title": job_title,
        "skills_required": list(set(words) & tech_skills),
        "languages": list(set(words) & languages),
        "soft_skills": list(set(words) & soft_skills),
        "min_experience": None,
        "degree": None,
        "top_keywords": top_keywords
    }

    # Years of experience
    exp_match = re.search(r"(\d+)\+?\s+(years|yrs)", jd_text.lower())
    if exp_match:
        extracted["min_experience"] = int(exp_match.group(1))

    # Degree match
    for pattern in degree_patterns:
        degree_match = re.search(pattern, jd_text.lower())
        if degree_match:
            extracted["degree"] = degree_match.group(1)
            break

    return extracted
