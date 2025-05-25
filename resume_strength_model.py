import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error

training_data = [
    {"text": "Led a team of 5 to develop a CRM tool used by 1,000+ clients", "score": 95},
    {"text": "Reduced cloud costs by 40% using AWS Lambda and EC2 autoscaling", "score": 92},
    {"text": "Built full-stack web app with React, Node.js, and PostgreSQL", "score": 90},
    {"text": "Created data pipeline using Python, Airflow, and Snowflake", "score": 93},
    {"text": "Managed Agile sprints and coordinated cross-functional teams", "score": 88},
    {"text": "Developed machine learning model that improved churn prediction by 15%", "score": 94},
    {"text": "Published research on NLP at EMNLP 2023", "score": 96},
    {"text": "Shipped MVP to 100+ beta users within 2 weeks", "score": 91},
    {"text": "Designed REST API used across 3 microservices", "score": 89},
    {"text": "Optimized SQL queries to reduce latency by 60%", "score": 90},
    
    {"text": "Created dashboards using Tableau to track KPIs", "score": 83},
    {"text": "Analyzed user behavior data and presented insights to product team", "score": 88},
    {"text": "Built a chatbot using Python and Flask for customer support", "score": 87},
    {"text": "Mentored junior developers and reviewed their pull requests", "score": 85},
    {"text": "Migrated legacy system from on-premise to AWS cloud", "score": 91},
    {"text": "Developed an A/B testing framework for marketing team", "score": 89},
    {"text": "Improved web accessibility scores using semantic HTML and ARIA", "score": 79},
    {"text": "Trained internal teams on Git workflows and CI/CD best practices", "score": 82},
    {"text": "Implemented user authentication with JWT and OAuth2", "score": 80},
    {"text": "Audited and improved application security using OWASP standards", "score": 90},
    {"text": "Built internal Chrome extension to automate email responses", "score": 86},
    {"text": "Conducted usability testing for mobile application redesign", "score": 76},
    {"text": "Spearheaded hackathon project that won best innovation award", "score": 92},
    {"text": "Refactored React codebase to reduce technical debt", "score": 75},
    {"text": "Developed training materials for onboarding software engineers", "score": 78},
    {"text": "Created SQL reports for weekly business metrics", "score": 72},
    {"text": "Maintained and monitored Elasticsearch cluster", "score": 73},
    {"text": "Redesigned UI using Figma, resulting in 20% increase in task completion", "score": 88},
    {"text": "Built ETL pipeline for social media analytics dashboard", "score": 87},
    {"text": "Collaborated with UX team to improve form error handling", "score": 79},
    {"text": "Maintained internal HR dashboard built with Django", "score": 78},
    {"text": "Assisted senior engineers with feature testing and bug tracking", "score": 75},
    {"text": "Wrote unit tests using PyTest for backend APIs", "score": 80},
    {"text": "Participated in code reviews and team retrospectives", "score": 72},
    {"text": "Contributed to open-source library with 300+ GitHub stars", "score": 83},
    {"text": "Improved CSS responsiveness on mobile views", "score": 74},
    {"text": "Created documentation for setup and onboarding process", "score": 77},
    {"text": "Built personal portfolio using Next.js and deployed on Vercel", "score": 70},
    {"text": "Integrated Stripe payment gateway in ecommerce app", "score": 84},
    {"text": "Created visualizations in Tableau for business insights", "score": 81},

    {"text": "Worked on updating user interface elements", "score": 65},
    {"text": "Helped organize weekly team meetings", "score": 58},
    {"text": "Used Microsoft Excel to analyze sales data", "score": 60},
    {"text": "Attended daily standups and took meeting notes", "score": 55},
    {"text": "Completed coursework in Machine Learning and Data Science", "score": 68},
    {"text": "Built small Python scripts for automation", "score": 66},
    {"text": "Assisted with regression testing for mobile app", "score": 64},
    {"text": "Participated in summer internship program", "score": 59},
    {"text": "Created simple login page using HTML and CSS", "score": 62},
    {"text": "Summarized articles for a blog series", "score": 57},

    {"text": "Made a resume", "score": 30},
    {"text": "Did work related to websites", "score": 40},
    {"text": "Attended an event", "score": 35},
    {"text": "Was part of a team", "score": 42},
    {"text": "Worked on school project", "score": 39},
    {"text": "Wrote some code", "score": 33},
    {"text": "Created things for a class", "score": 37},
    {"text": "Helped out", "score": 28},
    {"text": "Used some software", "score": 36},
    {"text": "Did design stuff", "score": 34},
    {"text": "Participated in club activities", "score": 38},
    {"text": "Helped organize group", "score": 44},
    {"text": "Worked at a company", "score": 41},
    {"text": "Made a website once", "score": 29},
    {"text": "Worked on a bunch of tasks", "score": 32},
    {"text": "Handled things", "score": 26},
    {"text": "Fixed bugs (I think)", "score": 25},
    {"text": "Did internship", "score": 43},
    {"text": "Filled out forms and managed Excel sheets", "score": 47},
    {"text": "Typed up notes", "score": 31},
    
]


training_dataset = pd.DataFrame(training_data)
X = training_dataset['text']
y = training_dataset['score']
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)
print(vectorizer.get_feature_names_out())
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

test_line = ["Created dashboards using Tableau to track KPIs"]
X_new = vectorizer.transform(test_line)
predicted_score = model.predict(X_new)[0]

print(f"Predicted Resume Strength Score: {round(predicted_score)}/100")

# pushing
