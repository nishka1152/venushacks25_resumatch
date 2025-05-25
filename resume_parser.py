import fitz  # PyMuPDF
import re

def pdf_to_python_list(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    text = re.sub(r'\r', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text.strip())

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    formatted = 'raw_lines = [\n' + ',\n'.join(f'    "{line}"' for line in lines) + '\n]'
    return formatted
