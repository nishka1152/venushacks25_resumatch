from flask import Flask, request, jsonify, send_file
from jd_parser import extract_job_features
from resume_parser import pdf_to_python_list
from resume_lines_and_labels import dataframe_builder
from resume_parser_ml_model import vectorizer, model  # classifier

app = Flask(__name__)

# Helper: Extract resume_dict from labeled lines
def build_resume_dict_from_lines(lines):
    df = dataframe_builder(lines)

    skill_lines = [row['text'] for _, row in df.iterrows() if row['label'] == "Skills"]
    experience_lines = [row['text'] for _, row in df.iterrows() if row['label'] == "Experience"]
    education_lines = [row['text'] for _, row in df.iterrows() if row['label'] == "Education"]

    all_text = " ".join(lines).lower()

    return {
        "skills": [word.lower() for line in skill_lines for word in line.split()],
        "languages": [lang for lang in ['python', 'java', 'sql', 'c++', 'javascript'] if lang in all_text],
        "soft_skills": [s for s in ['communication', 'teamwork', 'leadership', 'problem-solving'] if s in all_text],
        "experience_years": 2,  # need to fix (how do we get the experience?)
        "education": education_lines[0].lower() if education_lines else "",
        "keywords": lines
    }

# Helper: JD-Resume score function
def score_resume_against_jd(jd_dict, resume_dict):
    score = 0
    weights = {
        "skills": 0.4,
        "languages": 0.1,
        "soft_skills": 0.2,
        "experience": 0.2,
        "degree": 0.1
    }

    matched_skills = set(jd_dict["skills_required"]) & set(resume_dict["skills"])
    skill_score = len(matched_skills) / len(jd_dict["skills_required"]) if jd_dict["skills_required"] else 0

    matched_soft_skills = set(jd_dict["soft_skills"]) & set(resume_dict["soft_skills"])
    soft_skill_score = len(matched_soft_skills) / len(jd_dict["soft_skills"]) if jd_dict["soft_skills"] else 0

    matched_langs = set(jd_dict["languages"]) & set(resume_dict["languages"])
    lang_score = len(matched_langs) / len(jd_dict["languages"]) if jd_dict["languages"] else 0

    exp_score = min(resume_dict["experience_years"] / jd_dict["min_experience"], 1.0) if jd_dict["min_experience"] and resume_dict["experience_years"] else 0

    degree_match = 1.0 if jd_dict["degree"] and resume_dict["education"] and jd_dict["degree"].split()[0] in resume_dict["education"] else 0

    score += (
        weights["skills"] * skill_score +
        weights["soft_skills"] * soft_skill_score +
        weights["languages"] * lang_score +
        weights["experience"] * exp_score +
        weights["degree"] * degree_match
    )

    return {
        "total_score": round(score * 100),
        "tech_skill_match": round(skill_score * 100),
        "soft_skill_match": round(soft_skill_score * 100),
        "language_match": round(lang_score * 100),
        "experience_score": round(exp_score * 100),
        "degree_match": round(degree_match * 100)
    }

# Route to serve frontend HTML
@app.route('/')
def index():
    return send_file('front_end.html')

# API Route: /api/match
@app.route('/api/match', methods=['POST'])
def match():
    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({'error': 'Missing input'}), 400

    resume_file = request.files['resume']
    jd_text = request.form['job_description']

    try:
        jd_dict = extract_job_features(jd_text)
        resume_lines = pdf_to_python_list(resume_file)
        resume_dict = build_resume_dict_from_lines(resume_lines)

        score = score_resume_against_jd(jd_dict, resume_dict)

        return jsonify({"score": score})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
