# app.py
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_description_vector], resume_vectors).flatten()

def plot_scores(results):
    plt.figure(figsize=(10, 5))
    plt.bar(results["Resume"], results["Score"], color='skyblue')
    plt.xlabel("Resumes")
    plt.ylabel("Score")
    plt.title("Resume Ranking based on Job Description")
    plt.xticks(rotation=45)
    st.pyplot(plt)

st.title("📄 AI Resume Screening & Candidate Ranking System")

st.header("🧾 Job Description")
job_description = st.text_area("Paste the job description here:")

st.header("📂 Upload Resumes (PDF only)")
uploaded_files = st.file_uploader("Upload multiple resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.subheader("📊 Resume Ranking")

    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    filenames = [file.name for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)

    results = pd.DataFrame({"Resume": filenames, "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    st.dataframe(results.reset_index(drop=True), use_container_width=True)
    plot_scores(results)
