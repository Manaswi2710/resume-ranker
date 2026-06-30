from sentence_transformers import SentenceTransformer, util
import pdfplumber
import os


model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    return text.strip()

def extract_all_resumes(pdf_paths):
    resumes = {}
    for path in pdf_paths:
        filename = os.path.basename(path)
        candidate_name = filename.replace(".pdf", "")
        text = extract_text_from_pdf(path)
        resumes[candidate_name] = text
    return resumes

def rank_resumes(jd_text, resumes):
    candidate_names = list(resumes.keys())
    resume_texts = list(resumes.values())

   
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)

   
    scores = util.cos_sim(jd_embedding, resume_embeddings)[0]
    

    results = list(zip(candidate_names, scores.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)

    ranked = []
    for rank, (name, score) in enumerate(results, start=1):
        ranked.append({
            "rank": rank,
            "candidate": name,
            "score": round(float(score) * 100, 2)
        })

    return ranked