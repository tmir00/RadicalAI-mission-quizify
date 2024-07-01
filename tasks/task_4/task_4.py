# embedding_client.py

from langchain_google_vertexai import VertexAIEmbeddings
import streamlit as st

class EmbeddingClient:
    """
    The EmbeddingClient class initializes an embedding client with specific configurations
    for model name, project, and location. This setup will allow the class to utilize Google Cloud's VertexAIEmbeddings for processing text queries.

    Parameters:
    - model_name: A string representing the name of the model to use for embeddings.
    - project: The Google Cloud project ID where the embedding model is hosted.
    - location: The location of the Google Cloud project, such as 'us-central1'.
    """
    
    def __init__(self, model_name, project, location):
        # Initialize the VertexAIEmbeddings client with the given parameters
        self.client = VertexAIEmbeddings(
            model_name = model_name,
            project = project,
            location = location
        )
        
    def embed_query(self, query):
        """
        Uses the embedding client to retrieve embeddings for the given query.

        :param query: The text query to embed.
        :return: The embeddings for the query or None if the operation fails.
        """
        vectors = self.client.embed_query(query)
        return vectors
    
    def embed_documents(self, documents):
        """
        Retrieve embeddings for multiple documents.

        :param documents: A list of text documents to embed.
        :return: A list of embeddings for the given documents.
        """
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            print("Method embed_documents not defined for the client.")
            return None

if __name__ == "__main__":
    model_name = "textembedding-gecko@003"
    project = "sample-mission-422802"
    location = "us-central1"

    embedding_client = EmbeddingClient(model_name, project, location)
    vectors = embedding_client.embed_query("Hello World!")
    if vectors:
        st.write(vectors)
        print(vectors)
        print("Successfully used the embedding client!")
    else:
        st.error("Failed to retrieve embeddings!")
