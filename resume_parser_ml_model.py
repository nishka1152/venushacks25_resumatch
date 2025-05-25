from resume_lines_and_labels import dataframe_builder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sample_data = ['Bachelor of Science in Computer Science, UC Irvine', 
               'Graduated with a GPA of 3.8/4.0', "Dean's List", 
               'Honors', 
               'Summa Cum Laude', 
               'Magna Cum Laude', 
               'Relevant Coursework: Data Structures, Algorithms, Machine Learning', 
               'Software Engineering Intern at Google, Summer 2023',
                'Developed internal tools using Python and Flask', 
                'Collaborated with cross-functional teams to improve deployment pipelines', 
                'Languages: Python, Java, JavaScript', 
                'Frameworks: React, Flask, Node.js', 
                'Tools: Git, Docker, Postman', 
                'Built a resume parser using NLP and machine learning', 
                'Created a full-stack task manager app using MERN stack', 
                'Implemented sentiment analysis tool for Reddit comments',
                'Can code in C++',
                'Oversaw daily activity and outing planning for 100 clients',
                'AWS Certified Solutions Architect – Associate', 
                'Google IT Support Professional Certificate', 
                'References available upon request', 
                'Portfolio: www.myname.dev', 
                'Love cooking', 
                'Like rock climbing']

<<<<<<< HEAD
def section_generator():
    sample_data = ['Bachelor of Science in Computer Science, UC Irvine', 
                   'BS in Early Childhood Development (1999)', 'BA in Elementary Education',
                'Graduated with a GPA of 3.8/4.0', "Dean's List", 
                'Honors', 
                'Summa Cum Laude', 
                'Magna Cum Laude', 
                'Relevant Coursework: Data Structures, Algorithms, Machine Learning', 
                'Software Engineering Intern at Google, Summer 2023',
                    'Developed internal tools using Python and Flask', 
                    'Collaborated with cross-functional teams to improve deployment pipelines', 
                    'Languages: Python, Java, JavaScript', 
                    'Frameworks: React, Flask, Node.js', 
                    'Tools: Git, Docker, Postman', 
                    'Built a resume parser using NLP and machine learning', 
                    'Created a full-stack task manager app using MERN stack', 
                    'Implemented sentiment analysis tool for Reddit comments',
                    'Can code in C++',
                    'Oversaw daily activity and outing planning for 100 clients',
                    'AWS Certified Solutions Architect – Associate', 
                    'Google IT Support Professional Certificate', 
                    'References available upon request', 
                    'Portfolio: www.myname.dev', 
                    'Love cooking', 
                    'Like rock climbing']

    test_data = dataframe_builder(sample_data)
=======
test_data = dataframe_builder(sample_data)
>>>>>>> 57351e2c0303c84dc19124e8749150de0fe8ed95



texts = list(test_data["text"])
labels = list(test_data["label"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
print(vectorizer.get_feature_names_out())

model = LogisticRegression()
model.fit(X, labels)

new_line = "Assisted families of special needs clients with researching financial assistance and healthcare"
X_new = vectorizer.transform([new_line])

<<<<<<< HEAD
    return sections
    
print(section_generator()["Other"])



    
    
=======
pred = model.predict(X_new)[0]
print("Predicted Section:", pred)
>>>>>>> 57351e2c0303c84dc19124e8749150de0fe8ed95
