from action_verbs_builder import action_verb_list
action_verbs = action_verb_list()
action_verbs.extend(["intern", "developed", "worked", "engineer", "team", "collaborated", 
                       "managed", "project", "designed", "led", "lead", "contributed", "analyst", "facilitated", "oversaw", "planned"])
section_keywords = {
    "Education": [
        "gpa", "bachelor", "master", "phd", "university", "college", "degree",
        "coursework", "graduated", "academic", "school", "magna", "summa", "cum laude"
    ],
    "Experience": action_verbs,
    "Skills": [
        "skills", "languages", "tools", "frameworks", "technologies", "proficient",
        "familiar", "knowledge of", "expertise", "experience with", "fluent", "code"
    ],
    "Projects": [
        "project", "built", "created", "designed", "developed", "implemented",
        "github", "demo", "application", "tool"
    ],
    "Certifications": [
        "certified", "certification", "certificate", "aws", "coursera", "udemy",
        "google", "microsoft", "verified", "verification"
    ],
    "Other": [
        "reference", "portfolio", "available upon", "personal website", "linkedin",
        "contact", "hobbies", "interests"
    ]
}

 
def dataframe_builder(training_lines: list):
    def label_line(line: str):
        line_lower = line.lower()
        for section, keywords in section_keywords.items():
            for keyword in keywords:
                if keyword in line_lower:
                    return section
        return "Other"  # if no keyword matches

    labeled_data = [{"text": line, "label": label_line(line)} for line in training_lines]

    import pandas as pd
    df = pd.DataFrame(labeled_data)
    return df

