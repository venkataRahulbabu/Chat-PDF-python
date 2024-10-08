import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the question is not based on the context or is a general comment, handle it appropriately.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def is_general_comment(user_question):
    general_comments = ["well done", "thank you", "thanks", "good job", "great", "nice", "awesome"]
    return any(comment in user_question.lower() for comment in general_comments)

def user_input(user_question):
    if is_general_comment(user_question):
        return "You're welcome! I'm here to help you with any questions you have."

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Chat with PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.error("Please upload at least one PDF file.")

    # Custom CSS for chat styling
    st.markdown("""
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
            }
            .user-message, .model-response {
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
                max-width: 60%;
            }
            .user-message {
                align-self: flex-end;
                background-color: #DCF8C6;
                font-weight: 600;
            }
            .model-response {
                align-self: flex-start;
                background-color: #F1F0F0;
                font-weight: normal;
            }
        </style>
    """, unsafe_allow_html=True)

    for i, (question, answer) in enumerate(st.session_state.conversation):
        st.markdown(f'<div class="chat-container"><div class="user-message">YOU: {question}</div><div class="model-response">MODEL: {answer}</div></div>', unsafe_allow_html=True)

    new_question = st.text_input("Your new question:", key=f"question_{len(st.session_state.conversation)}")

    if st.button("Submit Query"):
        if new_question:
            answer = user_input(new_question)
            st.session_state.conversation.append((new_question, answer))
            st.rerun()

if __name__ == "__main__":
    main()
