from flask import Flask, request, jsonify, send_file
from jd_parser import extract_job_features
from resume_parser import pdf_to_python_list
from jd_resume_strength import section_generator, final_score_checking
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('front_end.html')

@app.route('/api/match', methods=['POST'])
def match():
    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({'error': 'Missing input'}), 400

    resume_file = request.files['resume']
    jd_text = request.form['job_description']

    try:
        # 1. Extract JD features
        jd_dict = extract_job_features(jd_text)

        # 2. Parse resume PDF into lines
        resume_lines = pdf_to_python_list(resume_file)

        # 3. Build structured resume section dict
        resume_section_dict = section_generator(resume_lines)

        # 4. Compute final score
        score = final_score_checking(resume_section_dict, jd_dict)

        # 5. Return all score fields for front end

    
        score = final_score_checking(resume_section_dict, jd_dict)

        return jsonify({
            "score": {
                "total_score": round(score["total_score"] * 100),
                "tech_skill_match": round(score["tech_skill_match"] * 100),
                "soft_skill_match": round(score["soft_skill_match"] * 100),
                "language_match": round(score["language_match"] * 100),
                "degree_match": 100 if score["degree_match"] else 0,
                "experience_score": 100 if score["experience_match"] else 0,
                "experience_years": score["experience_years"]
            }
        })

    except Exception as e:
        print("ERROR DETAILS")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
