import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# ===============================
# Environment and Groq LLM Setup
# ===============================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY not found. Please add it in your .env file.")
    st.stop()

llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")
memory = ConversationBufferMemory(memory_key="chat_history")

# ===============================
# Embeddings and FAISS Setup
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
st.caption("Fast Groq + FAISS powered GenAI learning pipeline for your final year project 🚀")

uploaded_file = st.file_uploader("📄 Upload PDF notes/resume for personalized retrieval (optional):", type="pdf")
career_goal = st.text_input("🎯 Enter your target career role:", "AI Product Manager")

# ===============================
# Upload and Index PDF
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
# Generate Micro-Course Roadmap
# ===============================
if st.button("✨ Generate Micro-Course Roadmap"):
    chain = LLMChain(llm=llm, prompt=roadmap_prompt, memory=memory)
    result = chain.run(career_goal=career_goal)
    st.markdown("## 📘 Micro-Course Roadmap")
    st.success(result)

# ===============================
# Generate Quiz
# ===============================
if st.button("📝 Generate Quiz"):
    chain = LLMChain(llm=llm, prompt=quiz_prompt, memory=memory)
    result = chain.run(career_goal=career_goal)
    st.markdown("## 📝 Quiz")
    st.info(result)

# ===============================
# Generate Flashcards
# ===============================
if st.button("📚 Generate Flashcards"):
    chain = LLMChain(llm=llm, prompt=flashcard_prompt, memory=memory)
    result = chain.run(career_goal=career_goal)
    st.markdown("## 📚 Flashcards")
    st.success(result)

# ===============================
# Generate Mock Interview Questions
# ===============================
if st.button("🎤 Generate Mock Interview Questions"):
    chain = LLMChain(llm=llm, prompt=interview_prompt, memory=memory)
    result = chain.run(career_goal=career_goal)
    st.markdown("## 🎤 Mock Interview Questions")
    st.warning(result)

# ===============================
# Mentor Chatbot
# ===============================
st.markdown("<h2 style='color:#00BFFF'>💬 SkillForgeAI Mentor Chatbot</h2>", unsafe_allow_html=True)
user_input = st.text_input("💡 Ask your mentor about your learning path, skills, or interview prep here:")

if user_input:
    if vector_store:
        docs = vector_store.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in docs])
    else:
        context = ""
    prompt = f"You are a helpful AI career mentor. Use the context below if relevant.\nContext: {context}\nQuestion: {user_input}\nAnswer clearly, practically, and concisely."
    response = llm.invoke(prompt)
    st.markdown("### 🤖 Mentor's Response")
    st.info(response.content)
