# venushacks25
# 💼 ResuMatch

**AI-powered resume screening for employers.**  
Upload a resume and job description — get instant section breakdowns, skill matches, and bullet-level impact scores.


🧠 What It Does
Extracts clean text from PDF resumes
Classifies lines into resume sections (Education, Experience, Projects, etc.)
Scores resume bullet points using a trained ML model
Analyzes job descriptions and extracts skill requirements
Matches resume content to the JD using keywords and WordNet-based synonyms
Renders a complete recruiter-friendly analysis

Features
Section detection with logistic regression
Bullet scoring via gradient boosted regression
JD skill extraction and semantic keyword match
WordNet-enhanced synonym matching for deeper compatibility
Full web interface using Flask + HTML

Built With
Python
Flask
HTML + Jinja2
PyMuPDF (fitz)
scikit-learn
NLTK + WordNet
SentenceTransformers (all-MiniLM-L6-v2)
Pandas
