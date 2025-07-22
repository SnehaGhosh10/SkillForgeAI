import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="SkillForgeAI", page_icon="🚀", layout="centered")

# ===============================
# Load environment
# ===============================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY not found in .env")
    st.stop()

# ===============================
# Initialize Groq LLM
# ===============================
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

# ===============================
# Embeddings & Vector Store
# ===============================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = None

# ===============================
# Prompt Templates
# ===============================
roadmap_prompt = PromptTemplate(
    input_variables=["career_goal"],
    template="""
You are a career mentor. Create a structured micro-course to become a {career_goal} including:
- Key skills
- Modules with sub-modules
- Free resources
- Mini projects
- 8-week roadmap
"""
)

quiz_prompt = PromptTemplate(
    input_variables=["career_goal"],
    template="Generate a 5-question quiz with answers to test knowledge on {career_goal}."
)

flashcard_prompt = PromptTemplate(
    input_variables=["career_goal"],
    template="Generate 10 flashcards (Q: question, A: answer) to revise {career_goal}."
)

interview_prompt = PromptTemplate(
    input_variables=["career_goal"],
    template="Generate 5 technical and 2 behavioral mock interview questions for {career_goal}."
)

# ===============================
# Streamlit UI
# ===============================
st.markdown("<h1 style='color:#00BFFF'>🚀 SkillForgeAI: GenAI Career Micro-Course Generator</h1>", unsafe_allow_html=True)
st.caption("⚡ Powered by Groq (fast GenAI) + FAISS retrieval for your final year project.")

uploaded_file = st.file_uploader("📄 Upload PDF notes/resume for personalized retrieval (optional):", type="pdf")
career_goal = st.text_input("🎯 Enter your target career role:", "AI Product Manager").strip()

# ===============================
# PDF Upload and Indexing
# ===============================
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, embeddings)
    st.success("✅ Uploaded and indexed with FAISS for personalized retrieval.")

# ===============================
# Micro-Course Roadmap Generation
# ===============================
if st.button("✨ Generate Micro-Course Roadmap"):
    with st.spinner("Generating roadmap..."):
        prompt = roadmap_prompt.format(career_goal=career_goal)
        response = llm.invoke(prompt)
        st.markdown("## 📘 Micro-Course Roadmap")
        st.success(response.content)

# ===============================
# Quiz Generation
# ===============================
if st.button("📝 Generate Quiz"):
    with st.spinner("Generating quiz..."):
        prompt = quiz_prompt.format(career_goal=career_goal)
        response = llm.invoke(prompt)
        st.markdown("## 📝 Quiz")
        st.info(response.content)

# ===============================
# Flashcards Generation
# ===============================
if st.button("📚 Generate Flashcards"):
    with st.spinner("Generating flashcards..."):
        prompt = flashcard_prompt.format(career_goal=career_goal)
        response = llm.invoke(prompt)
        st.markdown("## 📚 Flashcards")
        st.success(response.content)

# ===============================
# Mock Interview Generation
# ===============================
if st.button("🎤 Generate Mock Interview Questions"):
    with st.spinner("Generating mock interview questions..."):
        prompt = interview_prompt.format(career_goal=career_goal)
        response = llm.invoke(prompt)
        st.markdown("## 🎤 Mock Interview Questions")
        st.warning(response.content)

# ===============================
# Mentor Chatbot
# ===============================
st.markdown("<h2 style='color:#00BFFF'>💬 SkillForgeAI Mentor Chatbot</h2>", unsafe_allow_html=True)
user_input = st.text_input("💡 Ask your mentor about your learning path, skills, or interview prep here:").strip()

if user_input:
    with st.spinner("Thinking..."):
        if vector_store:
            docs = vector_store.similarity_search(user_input, k=3)
            context = "\n".join([doc.page_content for doc in docs])
        else:
            context = ""
        prompt = f"You are a helpful AI career mentor. Use the context below if relevant.\nContext: {context}\nQuestion: {user_input}\nAnswer clearly, practically, and concisely."
        response = llm.invoke(prompt)
        st.markdown("### 🤖 Mentor's Response")
        st.info(response.content)
