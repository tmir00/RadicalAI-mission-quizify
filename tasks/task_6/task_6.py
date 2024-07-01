import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator


if __name__ == "__main__":
    st.header("Quizzify")

    # Configuration for EmbeddingClient
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "sample-mission-422802",
        "location": "us-central1"
    }
    
    screen = st.empty() # Screen 1, ingest documents
    with screen.container():
        # Initalize DocumentProcessor and Ingest Documents
        processor = DocumentProcessor()
        processor.ingest_documents()
        # 2) Initalize the EmbeddingClient with embed config
        client = EmbeddingClient(**embed_config)
        # Initialize the ChromaCollectionCreator
        chroma_creator = ChromaCollectionCreator(processor, client)

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            # Use streamlit widgets to capture the user's input
            topic_input = st.text_input("Enter Topic")
            # for the quiz topic and the desired number of questions
            num_questions = st.slider("Number of Questions", min_value=1, max_value=50, value=10)
            
            document = None
            
            submitted = st.form_submit_button("Generate a Quiz!")
            if submitted:
                # Use the create_chroma_collection() method to create a Chroma collection from the processed documents
                db = chroma_creator.create_chroma_collection()
                   
                # document = chroma_creator.query_chroma_collection(topic_input) 
                
    if document:
        screen.empty()
        with st.container():
            st.header("Query Chroma for Topic, top Document: ")
            st.write(document)