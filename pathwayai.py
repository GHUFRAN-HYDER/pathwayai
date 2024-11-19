import logging
import sqlite3
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import bcrypt
import os
import asyncio
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()
# Enhanced logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Connect to MongoDB
def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client["pathwayai_users"]  # Create or connect to the database
    return db

db = get_mongo_client()
users_collection = db["users"]

def signup_user(email, password):
    try:
        # Check if the user already exists
        if users_collection.find_one({"email": email}):
            logger.debug(f"Signup failed: Email {email} already exists.")
            return False  # Email already exists

        # Hash the password and store user
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = {
            "email": email,
            "password": hashed_password.decode('utf-8'),
            "created_at": datetime.utcnow()
        }
        users_collection.insert_one(user)
        logger.debug(f"Signup successful: Email {email} added to database.")
        return True
    except Exception as e:
        logger.error(f"Error during signup: {str(e)}")
        return False


# Function to authenticate user
def authenticate_user(email, password):
    try:
        # Fetch the user by email
        user = users_collection.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
            logger.debug(f"Authentication successful for {email}.")
            return True
        logger.debug(f"Authentication failed for {email}.")
        return False
    except Exception as e:
        logger.error(f"Error during authentication: {str(e)}")
        return False


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Function to load PDFs and log debug info
def load_and_debug_pdf():

    try:
        pdf_directory = "knowledgebase"
        raw_docs = []
        pdf_files = [f for f in os.listdir(pdf_directory)]
        for pdf_file in pdf_files:
            file_path = os.path.join(pdf_directory, pdf_file)
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split()
            raw_docs.extend(docs)
            logger.info(f"Successfully loaded: {pdf_file}")
        return raw_docs
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise

# Function to split documents
def split_and_debug_documents(raw_docs):

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len, is_separator_regex=False
        )
        return text_splitter.split_documents(raw_docs)
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise

@st.cache_resource(show_spinner=False)
def initialize_resources():

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    index_path = "faiss_index.bin"

    if os.path.exists(index_path):
        logger.info("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        raw_docs = load_and_debug_pdf()
        docs = split_and_debug_documents(raw_docs)
        logger.info("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = """You are a EB-2 Immigration  information assistant. Follow these rules strictly:
    2. DO NOT make assumptions or provide general visa information
    3. Before responding, verify that the specific visa type or information being asked about appears in the context
    Context for visa-related questions: {context}"""

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return retriever, rag_chain

retriever, rag_chain = initialize_resources()

def main():
    st.title("PathwayAI")

    # User Authentication Section
    if not st.session_state.authenticated:
        option = st.selectbox("Select an option:", ["Login", "Sign Up"])
        if option == "Login":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if authenticate_user(email, password):
                    st.session_state.authenticated = True
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        elif option == "Sign Up":
            new_email = st.text_input("New Email")
            new_password = st.text_input("New Password", type="password")
            if st.button("Sign Up"):
                if signup_user(new_email, new_password):
                    st.success("Signup successful! You can now log in.")
                else:
                    st.error("Email already exists. Please choose another.")
        return
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question about visa information..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]]
            st.session_state.current_message = ""

            async def handle_query():
                message_placeholder = st.empty()
                async for chunk in rag_chain.astream({"input": prompt, "chat_history": chat_history}):
                    if "answer" in chunk:
                        st.session_state.current_message += chunk["answer"]
                        message_placeholder.markdown(st.session_state.current_message + "▌")
                message_placeholder.markdown(st.session_state.current_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": st.session_state.current_message}
                )
            asyncio.run(handle_query())
            sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
            selected = st.feedback("thumbs")
            if selected is not None:
               st.markdown(f"You selected: {sentiment_mapping[selected]}")
if __name__ == "__main__":
    main()