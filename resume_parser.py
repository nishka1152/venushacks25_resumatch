import fitz  # PyMuPDF
import re

def pdf_to_python_list(file_obj) -> list:
    doc = fitz.open(stream=file_obj.read(), filetype='pdf')
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    text = re.sub(r'\r', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text.strip())

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    return lines
