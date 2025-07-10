import streamlit as st # type: ignore #local deploment as well as streamlit cloud create your web application
from PyPDF2 import PdfReader # type: ignore #this helps to read and interact with PDF(Resume are in PDF formate)
#PdfReader- helps to read the PDF file
import pandas as pd # type: ignore #to interact with dataframe(Rows and columns)
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore #Sklearn- sci-kit learn
#feature_extraction- it si a sub folder in Sci-kit learn
import matplotlib.pyplot as plt #


from sklearn.metrics.pairwise import cosine_similarity # type: ignore #cosine_similarity- this is actually used for juding the similaritiesbtw two documents 
file=r"D:\Skill development Notes\Projects\project\Project1\Resume.pdf"
#r over here is used to show that the data is raw what does that mean to read the / as normal character as / is also used n escape sequense in python



# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file) #pdf object is created 
    text = ""
    for page in pdf.pages: #using previous object we are calling the function call as pages
        text += page.extract_text() #here operator helps to add the extracted text into text variable
                                    #text=text+page.extract.text()l
    return text #returning the text




# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes(multiple resumes in the form of list)
    documents [job_description] + resumes # type: ignore
    vectorizer = TfidfVectorizer().fit_transform(documents) # type: ignore #TFid- Term Frequency Inverse doc frequency
    vectors = vectorizer.toarray() #vectors act as a list
                                   #list[resume1,resume2,resume3]

    # Calculate cosine similarity
    job_description_vector = vectors[0] #over here we are indexing the list
    resume_vectors = vectors [1:] #over here we are slicing the list
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    #cosine_similarity is used to compare the job dis(doc) with the resume(doc)
    return cosine_similarities #closer to 1 than ore accurate and vice-versa
    #Similarly the ATS System works which helps to rank the resume in the pre interview rounds.

def adjust_ranking(scores, feedback):
    adjustment_factor = feedback / 5.0  # Normalize feedback between 0 and 1
    adjusted_scores = scores * adjustment_factor
    return adjusted_scores



# Plot bar chart of resume scores
def plot_scores(results):
    plt.bar(results["Resume"], results["Score"])
    plt.xlabel("Resumes")
    plt.ylabel("Scores")
    plt.title("Resume Scores")
    st.pyplot(plt)


# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System") #title of my app
# Job description input
st.header("Job Description") #it will take the job discription from the user
job_description= st.text_area ("Enter the job description")
# File uploader
st.header("Upload Resumes") #helps to upload the resumes 
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files and job_description: # ths instruction is used only if we upload the pdf and job discription
    st.header("Ranking Resumes")
    
    resumes = [] # blank list
    for file in uploaded_files:
        text = extract_text_from_pdf(file) # fuction calling
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes) #ranked as the comparison 

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores }) # used to arrange the resume in proper format 
    results = results.sort_values(by="Score", ascending=False) #used for sorting the best scored resume
    
    st.write(results) #used to display the results in the form of table
    #This is how the AI Resume Screening and Candidate Ranking System works.
    #This is the basic version of the project.
    #This can be further improved by adding more features and functionalities.
    #This is a very useful tool for HR professionals and recruiters.
    #This tool can help them to save time and effort in the recruitment process.
    #This tool can also help them to find the best candidates for the job.
    #This tool can also help them to avoid bias in the recruitment process.
    #This tool can also help them to improve the quality of hires.