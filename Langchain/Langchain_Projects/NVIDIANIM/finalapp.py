import streamlit as st 
import os 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain.chains import create_retrieval_chain 
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## Load the Nvidia API Key 
os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')
llm=ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader('./us_census')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,embedding=st.session_state.embeddings)

st.title("Nvidia NIM Demo")
prompt=ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only. 
Please provide the most accurate response based on the question 
<context>
{context}
<context>
Question:{input}
"""
)



prompt1=st.text_input("Enter Your Question From Documents")

if st.button("Document Embedding"):
    vector_embeddings()
    st.write("FAISS Vector Store DB Is Ready Using NvidiaEmbedding")

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrievel_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrievel_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------")

