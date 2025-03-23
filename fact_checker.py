import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import streamlit as st
from openai import OpenAI
import requests
from duckduckgo_search import DDGS
import numpy as np
import re
from FlagEmbedding import BGEM3FlagModel

class FactChecker:
    def __init__(self, api_base: str, model: str, temperature: float, max_tokens: int):
        """
        Initialize the fact checker with configuration parameters.
        
        Args:
            api_base: The base URL for the LLM API
            model: The model to use for fact checking
            temperature: Temperature parameter for LLM
            max_tokens: Maximum tokens for LLM response
        """
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.openai_api_key = "EMPTY"  # Placeholder for local setup
        
        # Initialize the OpenAI client with local settings
        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.api_base,
        )
        
        # Initialize the embedding model
        try:
            # self.embedding_model = BGEM3FlagModel('BAAI/bge-m3',  
            #            use_fp16=True)
            self.embedding_model = BGEM3FlagModel('/home/user1/wyf/model/bge-m3/')
        except Exception as e:
            st.error(f"Error loading BGE-M3 model: {str(e)}")
            self.embedding_model = None

    def extract_claim(self, text: str) -> str:
        """
        Extract core claims from the input text using LLM.
        
        Args:
            text: The input text to extract claims from
            
        Returns:
            extracted claim
        """
        system_prompt = """
        You are a precise claim extraction assistant. Analyze the provided news and summarize the central idea of it.
        Format the central idea as a worthy-check statement, which is a claim that can be verified independently.
        output format:
        calim: <claim>
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract the key factual claims from this text: {text}"}
                ],
                temperature=0.0,  # Use low temperature for consistent claim extraction
                max_tokens=500
            )
            
            claims_text = response.choices[0].message.content
            
            # Parse the numbered list into separate claims
            claims = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)', claims_text, re.DOTALL)
            
            # Clean up the claims
            claims = [claim.strip() for claim in claims if claim.strip()]
            
            # If no numbered claims were found, split by newlines
            if not claims and claims_text.strip():
                claims = [line.strip() for line in claims_text.strip().split('\n') if line.strip()]
            
            return claims[0]
            
        except Exception as e:
            st.error(f"Error extracting claims: {str(e)}")
            return text  # Return the original text as a fallback

    def search_evidence(self, claim: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for evidence using DuckDuckGo.
        
        Args:
            claim: The claim to search for evidence
            num_results: Number of search results to return
            
        Returns:
            List of evidence documents with title, url, and snippet
        """
        try:
            ddgs = DDGS(proxy="socks5://127.0.0.1:20170", timeout=60)
            results = list(ddgs.text(claim, max_results=num_results))
            
            evidence_docs = []
            for result in results:
                evidence_docs.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
            
            return evidence_docs
        except Exception as e:
            st.error(f"Error searching for evidence: {str(e)}")
            return []

    def get_evidence_chunks(self, evidence_docs: List[Dict[str, str]], claim: str, chunk_size: int = 200, chunk_overlap: int = 50, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Extract and rank evidence chunks related to the claim using BGE-M3.
        
        Args:
            evidence_docs: List of evidence documents
            claim: The claim to match with evidence
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            top_k: Number of top chunks to return
            
        Returns:
            List of ranked evidence chunks with similarity scores
        """
        if not self.embedding_model:
            return [{
                'text': "Evidence ranking unavailable - BGE-M3 model could not be loaded.",
                'source': "System",
                'similarity': 0.0
            }]
        
        try:
            # Create text chunks from evidence documents
            all_chunks = []
            
            for doc in evidence_docs:
                # Add title as a separate chunk
                all_chunks.append({
                    'text': doc['title'],
                    'source': doc['url'],
                })
                
                # Process the snippet into overlapping chunks
                snippet = doc['snippet']
                if len(snippet) <= chunk_size:
                    # If snippet is shorter than chunk_size, use it as is
                    all_chunks.append({
                        'text': snippet,
                        'source': doc['url'],
                    })
                else:
                    # Create overlapping chunks
                    for i in range(0, len(snippet), chunk_size - chunk_overlap):
                        chunk_text = snippet[i:i + chunk_size]
                        if len(chunk_text) >= chunk_size // 2:  # Only keep chunks of reasonable size
                            all_chunks.append({
                                'text': chunk_text,
                                'source': doc['url'],
                            })
            
            # Compute embeddings for claim
            claim_embedding = self.embedding_model.encode(claim)['dense_vecs']
            
            # Compute embeddings for chunks
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)['dense_vecs']
            
            # Calculate similarities
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = np.dot(claim_embedding, chunk_embedding) / (
                    np.linalg.norm(claim_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append(float(similarity))
            
            # Add similarities to chunks
            for i, similarity in enumerate(similarities):
                all_chunks[i]['similarity'] = similarity
            
            # Sort chunks by similarity (descending)
            ranked_chunks = sorted(all_chunks, key=lambda x: x['similarity'], reverse=True)
            
            # Return top k chunks
            return ranked_chunks[:top_k]
            
        except Exception as e:
            st.error(f"Error ranking evidence: {str(e)}")
            return [{
                'text': f"Error ranking evidence: {str(e)}",
                'source': "System",
                'similarity': 0.0
            }]
    def evaluate_claim(self, claim: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Evaluate the truthfulness of a claim based on evidence using LLM.
        
        Args:
            claim: The claim to evaluate
            evidence_chunks: The evidence chunks to use for evaluation
            
        Returns:
            Dictionary with verdict and reasoning
        """
        system_prompt = """
        You are a precise fact-checking assistant. Analyze the claim provided and determine its accuracy based on the evidence provided.
        
        Follow these steps:
        1. Consider each piece of evidence carefully
        2. Evaluate how the evidence supports or contradicts the claim
        3. Provide a clear verdict: TRUE, FALSE, or PARTIALLY TRUE
        4. Explain your reasoning with specific references to the evidence
        
        Format your response as:
        
        VERDICT: [TRUE/FALSE/PARTIALLY TRUE]
        
        REASONING: [Your detailed explanation citing specific evidence]
        
        Remain neutral, objective, and focus solely on what the evidence shows. Don't speculate beyond the provided evidence.
        """
        
        # Prepare evidence text for the prompt
        evidence_text = "\n\n".join([
            f"EVIDENCE {i+1} (Relevance: {chunk['similarity']:.2f}):\n{chunk['text']}\nSource: {chunk['source']}"
            for i, chunk in enumerate(evidence_chunks)
        ])
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_text}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_text = response.choices[0].message.content
            
            # Extract verdict and reasoning
            verdict_match = re.search(r'VERDICT:\s*(TRUE|FALSE|PARTIALLY TRUE)', result_text, re.IGNORECASE)
            verdict = verdict_match.group(1) if verdict_match else "UNVERIFIABLE"
            
            reasoning_match = re.search(r'REASONING:\s*(.*)', result_text, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else result_text
            
            return {
                "verdict": verdict,
                "reasoning": reasoning
            }
            
        except Exception as e:
            st.error(f"Error evaluating claim: {str(e)}")
            return {
                "verdict": "ERROR",
                "reasoning": f"An error occurred during evaluation: {str(e)}"
            }

    def check_fact(self, text: str) -> Dict[str, Any]:
        """
        Main function to check the factuality of a statement.
        
        Args:
            text: The statement to fact-check
            
        Returns:
            Dictionary with all results of the fact-checking process
        """
        # 1. Extract core claim
        claim = self.extract_claim(text)
        
        result = {
            "original_text": text,
            "claim": claim,
            "results": []
        }
        # 2. Search for evidence
        evidence_docs = self.search_evidence(claim)
        
        # 3. Get relevant evidence chunks
        evidence_chunks = self.get_evidence_chunks(evidence_docs, claim)
        
        # 4. Evaluate claim based on evidence
        evaluation = self.evaluate_claim(claim, evidence_chunks)
        
        # Add results for this claim
        result={
            "claim": claim,
            "evidence_docs": evidence_docs,
            "evidence_chunks": evidence_chunks,
            "verdict": evaluation["verdict"],
            "reasoning": evaluation["reasoning"]
        }
        
        return result


# Function to be imported in the main Streamlit app
def check_fact(claim: str, api_base: str, model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """
    Public interface for fact checking to be used by the Streamlit app.
    
    Args:
        claim: The statement to fact-check
        api_base: The base URL for the LLM API
        model: The model to use for fact checking
        temperature: Temperature parameter for LLM
        max_tokens: Maximum tokens for LLM response
        
    Returns:
        Dictionary with all results of the fact-checking process
    """
    checker = FactChecker(api_base, model, temperature, max_tokens)
    return checker.check_fact(claim)