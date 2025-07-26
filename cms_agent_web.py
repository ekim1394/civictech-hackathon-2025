#!/usr/bin/env python3

import streamlit as st
import boto3
import json
import os
import dotenv
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

class CMSAgent:
    def __init__(self):
        """Initialize the CMS Agent with AWS clients."""
        # Create Bedrock Runtime and S3 Vectors clients
        self.bedrock = boto3.client("bedrock-runtime", region_name='us-east-1')
        self.s3vectors = boto3.client("s3vectors", region_name='us-east-1')
        self.vector_bucket_name = "ekim-civictech-hackathon-2025"
        self.vector_index_name = "cms-titan"
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for input text using Amazon Titan Text Embeddings V2."""
        # Truncate text if it's too long (Titan has a context limit)
        if len(text) > 8000:
            text = text[:8000]
            
        response = self.bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": text})
        )

        # Extract embedding from response
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]
    
    def query_vector_store(self, query: str, top_k: int = 5) -> List[Dict[Any, Any]]:
        """Query the vector store with the given query text."""
        # logger.info(f"Querying vector store with query: {query}")
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        
        # Query vector index
        response = self.s3vectors.query_vectors(
            vectorBucketName=self.vector_bucket_name,
            indexName=self.vector_index_name,
            queryVector={"float32": query_embedding}, 
            topK=top_k, 
            returnDistance=True,
            returnMetadata=True
        )
        # logger.info(f"Response from vector store: {response}")
        return response["vectors"]
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate a response to the user's query about CMS dockets."""
        # Query the vector store to get relevant context
        results = self.query_vector_store(query)
        
        # Extract the text content and metadata from results
        contexts = []
        sources = []
        
        for result in results:
            logger.info(f"Result: {result}")
            if "metadata" in result and "source_text" in result["metadata"]:
                contexts.append(result["metadata"]["source_text"])
                
                # Extract source information
                source_info = {}
                if "docket_id" in result["metadata"]:
                    source_info["docket_id"] = result["metadata"]["docket_id"]
                if "title" in result["metadata"]:
                    source_info["title"] = result["metadata"]["title"]
                if "document_type" in result["metadata"]:
                    source_info["document_type"] = result["metadata"]["document_type"]
                if "posted_date" in result["metadata"]:
                    source_info["posted_date"] = result["metadata"]["posted_date"]
                logger.info(f"Source info: {source_info}")
                sources.append(source_info)
        
        # Use Bedrock Claude model to generate a response based on the context
        prompt = self._create_prompt(query, contexts)
        
        response = self.bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        # Extract and return the response
        response_body = json.loads(response["body"].read())
        answer = response_body["content"][0]["text"]
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources
        }
    
    def _create_prompt(self, query: str, contexts: List[str]) -> str:
        """Create a prompt for the LLM using the query and retrieved contexts."""
        context_text = "\n\n".join(contexts)
        
        prompt = f"""You are a helpful assistant that answers questions about CMS (Centers for Medicare & Medicaid Services) dockets, regulations, Medicare, and Medicaid programs.
        
        Below is information retrieved from CMS documents that may help answer the user's question.

        RETRIEVED CONTEXT:
        {context_text}

        USER QUESTION: {query}

        Please answer the question based on the retrieved context. Be particularly thorough when addressing questions about:
        - Common issues with Medicare
        - Changes to Medicaid eligibility
        - Recent CMS final rules and their key points
        - CMS regulation of telehealth services
        - Medicare and Medicaid coverage policies

        If the context doesn't contain enough information to answer the question fully, acknowledge what you know and what you don't know. Be specific and cite information from the context when possible. 

        If the question is about Medicare or Medicaid but the context doesn't provide sufficient information, you can still provide general information about these programs while noting that your answer isn't based on the most recent regulations.

        If the question is not related to CMS dockets, regulations, Medicare, or Medicaid, politely explain that you're focused on helping with CMS-related inquiries.
        """
        return prompt


# Streamlit UI
def main():
    st.set_page_config(
        page_title="CMS Docket Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize the agent
    if 'agent' not in st.session_state:
        st.session_state.agent = CMSAgent()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Header
    st.title("CMS Docket Assistant")
    st.markdown("""
    Ask questions about CMS (Centers for Medicare & Medicaid Services) dockets and regulations.
    This assistant uses AI to search through CMS documents and provide relevant information.
    """)
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant helps you find information about CMS dockets and regulations.
        
        It uses:
        - Vector embeddings to search through CMS documents
        - Amazon Bedrock for AI processing
        - Streamlit for the web interface
        
        **Example questions:**
        - What are common issues with Medicare?
        - Explain the latest changes to Medicaid eligibility
        - What are the key points in the recent CMS final rule?
        - How does CMS regulate telehealth services?
        """)
        
        st.header("Settings")
        top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If this is an assistant message with sources, display them in an expander
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}**")
                        st.markdown(f"**Title:** {source.get('title', 'Untitled')}")
                        st.markdown(f"**Docket ID:** {source.get('docket_id', 'No docket ID')}")
                        if "document_type" in source:
                            st.markdown(f"**Document Type:** {source['document_type']}")
                        if "posted_date" in source:
                            st.markdown(f"**Posted Date:** {source['posted_date']}")
                        st.divider()
    
    # User input
    if prompt := st.chat_input("Ask a question about CMS dockets and regulations"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with a spinner while processing
        with st.chat_message("assistant"):
            with st.spinner("Searching CMS documents..."):
                try:
                    # Generate response
                    result = st.session_state.agent.generate_response(prompt)
                    
                    # Display the answer
                    st.markdown(result["answer"])
                    
                    # Store sources for display in expander
                    sources = result["sources"]
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": sources
                    })
                    
                    # Display sources in an expander
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}**")
                            st.markdown(f"**Title:** {source.get('title', 'Untitled')}")
                            st.markdown(f"**Docket ID:** {source.get('docket_id', 'No docket ID')}")
                            if "document_type" in source:
                                st.markdown(f"**Document Type:** {source['document_type']}")
                            if "posted_date" in source:
                                st.markdown(f"**Posted Date:** {source['posted_date']}")
                            st.divider()
                            
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main() 