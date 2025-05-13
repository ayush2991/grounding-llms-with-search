import streamlit as st
import requests
import pandas as pd
import logging
import google.genai as genai
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple, Optional, Protocol
import os
import json
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class SentenceTokenizer(Protocol):
    """Protocol for sentence tokenizers"""
    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences"""
        ...

class SimpleTokenizer:
    """Simple space-based sentence tokenizer"""
    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences using spaces"""
        # Clean up any extra whitespace first
        text = ' '.join(text.split())
        sentences = text.split('. ')
        # Add periods back and handle the last sentence
        return [s + '.' if not s.endswith('.') else s for s in sentences]

@st.cache_resource(show_spinner=False)
def get_tokenizer() -> SentenceTokenizer:
    """Get the sentence tokenizer instance"""
    return SimpleTokenizer()

@st.cache_resource()
def get_google_client() -> genai.Client | None:
    """
    Initialize and return a Google Gemini API client.
    
    The function will try to get the API key from:
    1. Session state (user input in sidebar)
    2. Streamlit secrets
    
    Returns:
        Initialized Google Gemini client if successful, None if error
        
    Note:
        Uses cache_resource to avoid creating multiple clients
    """
    try:
        # First try to get the API key from session state (user input)
        gemini_api_key = st.session_state.get("gemini_api_key", "").strip()
        
        # If not in session state, try to get from secrets
        if not gemini_api_key:
            try:
                gemini_api_key = st.secrets.get("gemini_api_key", "").strip()
            except Exception as e:
                logger.error(f"Error accessing secrets: {e}")
                gemini_api_key = ""
        
        if not gemini_api_key:
            logger.error("No Gemini API key found")
            st.error("Please enter your Gemini API key in the sidebar or set it in secrets.")
            return None
            
        client = genai.Client(api_key=gemini_api_key)
        # Verify the client works by making a simple request
        client.models.list()
        return client
        
    except Exception as e:
        logger.exception("Failed to create Google client")
        st.error("Failed to initialize Gemini API client. Please check your API key.")
        return None

@st.cache_resource(show_spinner=True, ttl=7 * 24 * 60 * 60)
def get_sentence_transformer() -> SentenceTransformer:
    """Initialize and cache the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=True)
def get_gemini_response(_client: genai.Client, query: str, max_retries: int = 3) -> str | None:
    """
    Get a response from the Gemini API for the given query.
    
    The function will retry on failure up to max_retries times.
    Response is cached for 7 days to avoid unnecessary API calls.
    
    Args:
        _client: The Gemini API client instance
        query: The query to send to Gemini
        max_retries: Maximum number of retries on failure, defaults to 3
        
    Returns:
        The response text if successful, None if all retries fail
        
    Raises:
        No exceptions are raised, errors are logged and None is returned
    """
    logger.info(f"Getting Gemini response for: {query}")
    
    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[query]
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                st.warning(f"Retrying... ({attempt + 2}/{max_retries})")
            else:
                st.error("Max retries reached. Please try again later.")
                return None

@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=True)
def search_brave(query: str, num_results: int = 10) -> dict | None:
    """
    Search the Brave Search API with the given query.
    
    Handles API authentication using either session state or secrets.
    Results are cached for 7 days to avoid unnecessary API calls.
    
    Args:
        query: The search query string
        num_results: Number of results to return (1-20)
        
    Returns:
        JSON response from Brave Search API if successful, None if error
        
    Note:
        The function will attempt to get the API key from:
        1. Session state (user input in sidebar)
        2. Streamlit secrets
    """
    if not isinstance(query, str) or not query.strip():
        logger.error("Invalid or empty query")
        st.error("Please enter a valid search query")
        return None
    
    if not isinstance(num_results, int) or not 1 <= num_results <= 20:
        logger.error(f"Invalid num_results: {num_results}")
        st.error("Number of results must be between 1 and 20")
        return None
        
    logger.info(f"Searching for: {query}")
    
    # First try to get the API key from session state (user input)
    api_key = st.session_state.get("brave_search_api_key")
    # If not in session state, try to get from secrets
    if not api_key:
        try:
            api_key = st.secrets["brave_search_api_key"]
        except KeyError:
            st.error("Please enter your Brave Search API key in the sidebar or set it in secrets.")
            return None
        
    headers = {
        'X-Subscription-Token': api_key,
        'Accept': 'application/json',
    }
        
    params = {
        'q': query,
        'count': num_results,
        'result_filter': 'discussions, faq, infobox, news, query, summarizer, web',
    }
    try:
        response = requests.get(
            'https://api.search.brave.com/res/v1/web/search',
            headers=headers,
            params=params
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Brave Search Request failed: {e}")
        return None
    
    if response.status_code != 200: 
        st.error(f"Search failed with error: {response.status_code}")
        logger.error(f"Search failed with error: {response.status_code}")
        logger.error(f"Response: {response.text}")
        return None
    return response.json()

# @st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=True)
def fetch_webpage_content(url: str, max_retries: int = 3) -> str | None:
    """
    Fetch the content of a webpage and extract readable text.
    
    Args:
        url: The URL to fetch content from
        max_retries: Maximum number of retries on failure, defaults to 3
        
    Returns:
        The readable text content of the webpage if successful, None if error
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            # Handle specific HTTP errors
            if response.status_code == 403:
                logger.warning(f"Access forbidden (403) for {url} - Website may be blocking automated access")
                return ""
            elif response.status_code == 404:
                logger.warning(f"Page not found (404) for {url}")
                return ""
            elif response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} error for {url}")
                if attempt < max_retries - 1:
                    continue
                return ""
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
                element.decompose()
                
            # Get text and clean it up
            text = soup.get_text(separator=' ', strip=True)
            
            # Remove excessive whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Truncate very long texts to avoid overwhelming the UI
            if len(text) > 10000:
                text = text[:10000] + "... [content truncated]"
                
            return text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return ""
        except Exception as e:
            logger.error(f"Failed to process content from {url}: {e}")
            return ""

@st.cache_data(show_spinner=True)
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using the configured tokenizer"""
    tokenizer = get_tokenizer()
    return tokenizer.tokenize(text)

@st.cache_data(show_spinner=True)
def compute_embeddings(sentences: List[str], _model: SentenceTransformer) -> np.ndarray:
    """Compute and cache sentence embeddings"""
    return _model.encode(sentences, convert_to_tensor=True).cpu().numpy()

@st.cache_data(show_spinner=True)
def compute_sentence_similarities(
    llm_sentences: List[str],
    article_sentences: List[str],
    _model: SentenceTransformer
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cosine similarities between LLM response sentences and article sentences
    
    Returns:
        Tuple of (similarities matrix, llm_embeddings, article_embeddings)
    """
    llm_embeddings = compute_embeddings(llm_sentences, _model)
    article_embeddings = compute_embeddings(article_sentences, _model)
    similarities = cosine_similarity(llm_embeddings, article_embeddings)
    return similarities, llm_embeddings, article_embeddings

# @st.cache_data(ttl=7 * 24 * 60 * 60)
def search_response_to_dataframe(search_response: dict | None) -> pd.DataFrame | None:
    """
    Convert Brave Search API response to a pandas DataFrame.
    
    Args:
        search_response: JSON response from Brave Search API
        
    Returns:
        DataFrame with Title, URL, Content and sentence-level similarity data
    """
    try:
        if not isinstance(search_response, dict):
            logger.error(f"Invalid response type: {type(search_response)}")
            st.error("Invalid search response format")
            return None
            
        if "web" not in search_response:
            logger.error("Missing 'web' key in response")
            st.error("Invalid search response structure")
            return None
            
        results = search_response["web"].get("results", [])
        if not results:
            logger.warning("No results found in response")
            st.warning("No search results found")
            return None
            
        required_columns = {"title", "url"}
        if not all(col in results[0] for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            st.error("Invalid search result format")
            return None
            
        df = pd.DataFrame(results)
        df = df[["title", "url"]].rename(columns={
            "title": "Title",
            "url": "URL"
        })
        
        # Fetch content for each URL
        with st.spinner("Fetching article contents..."):
            df["Content"] = df["URL"].apply(fetch_webpage_content)
            
        # Add sentence tokenization
        df["Sentences"] = df["Content"].apply(split_into_sentences)
        df["Num_Sentences"] = df["Sentences"].apply(len)
            
        return df
        
    except Exception as e:
        logger.exception("Error converting search results to DataFrame")
        st.error(f"Error processing search results: {str(e)}")
        return None

st.set_page_config(page_title="Grounding LLMs with Search", page_icon=":mag_right:", layout="wide")

# Load resources
google_client = get_google_client()
if google_client is None:
    st.error("Failed to create Google client. Please check your API key.")
    st.stop()


st.title("Grounding LLMs with Search")
st.write("This app demonstrates how to find web citations for LLM responses.")

with st.sidebar:
    st.header("Parameters")
    st.text_area("Query", "tell me about cobalt mining in 100 words", key="query")
    st.checkbox("Debug mode", False, key="debug_mode", help="Show raw search and LLM response.")
    run_button = st.button("Run", key="run", help="Run the app. Scores LLM response against search results.")
    if st.button("Clear Cache", help="Clear all cached data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown("-----")
    st.subheader("Brave Search")
    st.text_input("Brave Search API Key", type="password", key="brave_search_api_key", help="Brave Search API key.")
    st.slider("Number of Search Results", 1, 20, 10, key="num_results")
    st.markdown("-----")
    st.subheader("Google Gemini")
    st.text_input("Gemini API Key", type="password", key="gemini_api_key", help="Gemini API key.")

if run_button:
    # Load the sentence transformer model
    sentence_model = get_sentence_transformer()
    
    with st.spinner("Fetching search results..."):    
        search_results = search_brave(st.session_state.query, st.session_state.num_results)
    if not search_results:
        st.error("Failed to get search results.")
        st.stop()
    search_response_df = search_response_to_dataframe(search_results)
    if search_response_df is None or search_response_df.empty:
        st.error("Failed to parse search results into dataframe.")
        st.stop()

    with st.spinner("Fetching LLM response..."):
        response = get_gemini_response(google_client, st.session_state.query)
    if response:
        st.session_state.llm_response = response
        # Split LLM response into sentences
        llm_sentences = split_into_sentences(response)
        st.session_state.llm_sentences = llm_sentences
    else:
        st.error("Failed to get response from LLM.")
        st.stop()

    # Compute similarities for each article
    similarities_data = []
    for idx, row in search_response_df.iterrows():
        similarities, _, _ = compute_sentence_similarities(
            llm_sentences,
            row["Sentences"],
            sentence_model
        )
        similarities_data.append({
            "URL": row["URL"],
            "Max_Similarity": float(np.max(similarities)),
            "Avg_Similarity": float(np.mean(similarities)),
            "Similarity_Matrix": similarities
        })
    
    # Add similarity scores to DataFrame
    similarities_df = pd.DataFrame(similarities_data)
    search_response_df = search_response_df.merge(
        similarities_df[["URL", "Max_Similarity", "Avg_Similarity"]],
        on="URL"
    )
    search_response_df = search_response_df.sort_values(
        by=["Max_Similarity", "Avg_Similarity"],
        ascending=False
    )

    if st.session_state.debug_mode:
        st.success("Successfully fetched {} search results.".format(len(search_response_df)))
        with st.expander("Search Results with Similarities", expanded=False):
            st.dataframe(
                search_response_df[[
                    "Title", "URL", "Content",
                    "Max_Similarity", "Avg_Similarity",
                    "Num_Sentences"
                ]],
                use_container_width=True
            )

        st.success("Successfully fetched LLM response.")
        with st.expander("LLM Response", expanded=False):
            st.markdown(st.session_state.llm_response)
            st.write("Number of sentences:", len(llm_sentences))
            st.write("Sentences:")
            for i, sent in enumerate(llm_sentences):
                st.write(f"{i+1}. {sent}")

