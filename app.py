import os
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

import chainlit as cl
from dotenv import load_dotenv
from PyPDF2 import PdfReader  # PDF handling

print("all_ok")

load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text file or PDF to begin!",
            accept=["text/plain", "application/pdf"],  # Accepting PDF files
            max_size_mb=50,
            timeout=180,
        ).send()

    file = files[0]
    
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()
    
    # Determine file type and extract text accordingly
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file.path)
    else:
        with open(file.path, "r", encoding="utf-8") as f:
            text = f.read()

    # Split the text into chunks
    texts = text_splitter.split_text(text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    
    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    message_history = ChatMessageHistory()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)
    
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    
    # Get the response from the chain
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    # Omit the sources
    await cl.Message(content=answer).send()
