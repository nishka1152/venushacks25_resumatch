from resume_lines_and_labels import dataframe_builder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from resume_parser import pdf_to_python_list

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

test_data = dataframe_builder(sample_data)



texts = list(test_data["text"])
labels = list(test_data["label"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
#print(vectorizer.get_feature_names_out())

model = LogisticRegression()
model.fit(X, labels)
pdf_path_test = '/Users/nishkajain/Desktop/functionalsample.pdf'
resume_lines = pdf_to_python_list(pdf_path_test)
print(resume_lines)
#for i in resume_lines:
    
    #print(i)
    