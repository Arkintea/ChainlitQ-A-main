import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to process uploaded file
def process_file(uploaded_file):
    """Handles file processing based on the file type."""
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")
    return text

# Initialize Streamlit UI
st.title("Document Chatbot 🚀 ")

# Initialize session state for chain and chat history
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File uploader
uploaded_files = st.file_uploader("Upload text or PDF files", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing files..."):
        combined_text = ""
        for uploaded_file in uploaded_files:
            combined_text += process_file(uploaded_file) + "\n"

        # Split the text into chunks
        texts = text_splitter.split_text(combined_text)
        
        # Create metadata for each chunk
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
        
        # Create a Chroma vector store
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
        
        # Initialize memory and chain
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )
        
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            memory=memory,
            return_source_documents=False,  # Set to False to omit sources
        )
        
        st.success("Processing done. You can now ask questions!")

# Layout management
chat_placeholder = st.empty()
input_placeholder = st.empty()

# Display previous chat history
with chat_placeholder.container():
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**User:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")

# User input for questions (fixed at the bottom)
with input_placeholder.container():
    user_input = st.text_input("Ask a question about the documents:")

    if user_input and st.session_state.chain:
        with st.spinner("Generating response..."):
            chain = st.session_state.chain
            res = chain({"question": user_input})
            answer = res["answer"]
            
            # Update chat history
            st.session_state.chat_history.append({
                'user': user_input,
                'bot': answer
            })

            # Clear the input box
            st.empty()

            # Display the answer
            st.write(f"**Answer:** {answer}")

            # Display updated chat history
            chat_placeholder.empty()
            with chat_placeholder.container():
                st.subheader("Updated Chat History")
                for chat in st.session_state.chat_history:
                    st.write(f"**User:** {chat['user']}")
                    st.write(f"**Bot:** {chat['bot']}")


