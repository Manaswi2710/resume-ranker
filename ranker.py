import pdfplumber
import os
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a lightweight English tokenizer (no model download needed)
nlp = spacy.blank("en")


# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "

    return text.strip()


# -------------------------------
# Extract all resumes
# -------------------------------
def extract_all_resumes(pdf_paths):
    resumes = {}

    for path in pdf_paths:
        filename = os.path.basename(path)
        candidate_name = filename.replace(".pdf", "")

        text = extract_text_from_pdf(path)
        resumes[candidate_name] = text

    return resumes


# -------------------------------
# Preprocess text
# -------------------------------
def preprocess(text):

    doc = nlp(text.lower())

    tokens = [
        token.text
        for token in doc
        if token.is_alpha
    ]

    return " ".join(tokens)


# -------------------------------
# Rank resumes
# -------------------------------
def rank_resumes(jd_text, resumes):

    candidate_names = list(resumes.keys())

    processed_jd = preprocess(jd_text)

    processed_resumes = [
        preprocess(resume)
        for resume in resumes.values()
    ]

    corpus = [processed_jd] + processed_resumes

    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    similarity_scores = cosine_similarity(
        jd_vector,
        resume_vectors
    )[0]

    results = list(zip(candidate_names, similarity_scores))

    results.sort(key=lambda x: x[1], reverse=True)

    ranked = []

    for rank, (name, score) in enumerate(results, start=1):
        ranked.append({
            "rank": rank,
            "candidate": name,
            "score": round(score * 100, 2)
        })

    return ranked