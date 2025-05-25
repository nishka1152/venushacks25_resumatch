import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Download NLTK data
nltk.download('punkt')

# Sample job description
job_description = """
We are seeking a software engineer with 3+ years of experience in Python, Django, and REST APIs.
Strong communication and problem-solving skills are required. A bachelor’s degree in computer science or a related field is preferred.
AWS Certification is a plus.
"""

# Predefined lists
tech_skills = {"python", "java", "sql", "c++", "c", "javascript", "django", "flask", "html", "css", "rest", "git", "linux", "tensorflow"}
soft_skills = {"communication", "leadership", "teamwork", "collaboration", "problem-solving", "adaptability"}
languages = {"python", "java", "c++", "javascript", "c", "go", "rust", "typescript", "r", "scala", "sql"}

degree_patterns = [
    r"(bachelor[’']?s?|b\.?sc\.?)",
    r"(master[’']?s?|m\.?sc\.?)",
    r"(ph\.?d|phd)",
]

certifications_keywords = {"aws", "pmp", "azure", "gcp", "compTIA", "scrum", "cisco"}

# Tokenization
sentences = sent_tokenize(job_description)
words = [word.lower() for word in word_tokenize(job_description) if word.isalpha()]

# TF-IDF Keyword Extraction
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform([job_description])
feature_array = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()[0]
top_n = 10
top_indices = tfidf_scores.argsort()[-top_n:][::-1]
top_keywords = [(feature_array[i], round(tfidf_scores[i], 3)) for i in top_indices]

# Sentence Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences)

# Extraction Logic
extracted = {
    "job_title": re.search(r"(we are (looking for|seeking|hiring) (an?|the)? ?([a-z ]+))", job_description.lower()),
    "skills_required": list(set(words) & tech_skills),
    "languages": list(set(words) & languages),
    "soft_skills": list(set(words) & soft_skills),
    "min_experience": None,
    "degree": None,
    "certifications": [word for word in words if word.lower() in certifications_keywords],
    "top_keywords": top_keywords,
    "num_sentences": len(sentences),
    "sentence_embedding_shape": sentence_embeddings[0].shape
}

# Job title cleanup
if extracted["job_title"]:
    extracted["job_title"] = extracted["job_title"].group(4).strip()

# Experience extraction
exp_match = re.search(r"(\d+)\+?\s+(years|yrs)", job_description.lower())
if exp_match:
    extracted["min_experience"] = int(exp_match.group(1))

# Degree extraction
for pattern in degree_patterns:
    match = re.search(pattern, job_description.lower())
    if match:
        extracted["degree"] = match.group(1)
        break

# Output
print("\n--- Extracted Job Description Features ---")
for key, value in extracted.items():
    print(f"{key}: {value}")
