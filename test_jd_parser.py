from jd_parser import extract_job_features

job_description = """

"""

jd_data = extract_job_features(job_description)

for k, v in jd_data.items():
    print(f"{k}: {v}")
