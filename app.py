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
logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class SentenceTokenizer(Protocol):
    """Protocol for sentence tokenizers"""

    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences"""
        ...


class SimpleTokenizer:
    """Simple space-based sentence tokenizer"""

    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences using spaces"""
        logger.debug("Tokenizing text into sentences")
        try:
            # Clean up any extra whitespace first
            text = " ".join(text.split())
            sentences = text.split(". ")
            # Add periods back and handle the last sentence
            result = [s + "." if not s.endswith(".") else s for s in sentences]
            logger.debug(f"Successfully tokenized text into {len(result)} sentences")
            return result
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise


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
            st.error(
                "Please enter your Gemini API key in the sidebar or set it in secrets."
            )
            return None

        client = genai.Client(api_key=gemini_api_key)
        # Verify the client works by making a simple request
        client.models.list()
        return client

    except Exception as e:
        logger.exception("Failed to create Google client")
        st.error("Failed to initialize Gemini API client. Please check your API key.")
        return None


@st.cache_resource(show_spinner=False, ttl=7 * 24 * 60 * 60)
def get_sentence_transformer() -> SentenceTransformer:
    """Initialize and cache the sentence transformer model"""
    logger.info("Initializing sentence transformer model (all-MiniLM-L6-v2)")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Successfully initialized sentence transformer model")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize sentence transformer model: {e}")
        raise


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def get_gemini_response(
    query: str, 
    api_key: str,
    max_retries: int = 3
) -> str | None:
    """
    Get a response from the Gemini API for the given query.

    The function will retry on failure up to max_retries times.
    Response is cached for 7 days to avoid unnecessary API calls.

    Args:
        query: The query to send to Gemini
        api_key: The Gemini API key
        max_retries: Maximum number of retries on failure, defaults to 3

    Returns:
        The response text if successful, None if all retries fail
    """
    logger.info(f"Getting Gemini response for: {query}")

    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=[query]
            )
            return response.text
        except Exception as e:
            logger.error(
                f"Gemini API request failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                st.warning(f"Retrying... ({attempt + 2}/{max_retries})")
            else:
                st.error("Max retries reached. Please try again later.")
                return None


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def search_brave(
    query: str, 
    api_key: str,
    num_results: int = 10
) -> dict | None:
    """
    Search the Brave Search API with the given query.

    Args:
        query: The search query string
        api_key: The Brave Search API key
        num_results: Number of results to return (1-20)

    Returns:
        JSON response from Brave Search API if successful, None if error
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

    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }

    params = {
        "q": query,
        "count": num_results,
        "result_filter": "discussions, faq, infobox, news, query, summarizer, web",
    }
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
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


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def fetch_webpage_content(url: str, max_retries: int = 1) -> str | None:
    """
    Fetch the content of a webpage and extract readable text.
    Implements conditional requests and content hashing for efficient caching.

    Args:
        url: The URL to fetch content from
        max_retries: Maximum number of retries on failure, defaults to 1

    Returns:
        The readable text content of the webpage if successful, None if error
    """
    logger.info(f"Fetching content from URL: {url}")
    
    # Generate cache key from URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_key = f"webpage_content_{url_hash}"
    
    # Try to get cached response headers
    cached_etag = st.session_state.get(f"{cache_key}_etag")
    cached_last_modified = st.session_state.get(f"{cache_key}_last_modified")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }
    
    # Add conditional request headers if we have cached values
    if cached_etag:
        headers["If-None-Match"] = cached_etag
    if cached_last_modified:
        headers["If-Modified-Since"] = cached_last_modified

    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt + 1}/{max_retries} to fetch {url}")
            response = requests.get(url, headers=headers, timeout=10)

            # Handle 304 Not Modified
            if response.status_code == 304:
                logger.debug(f"Content not modified for {url}, using cached version")
                return st.session_state.get(cache_key, "")

            # Handle other HTTP errors
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

            # Store response headers for future conditional requests
            if "ETag" in response.headers:
                st.session_state[f"{cache_key}_etag"] = response.headers["ETag"]
            if "Last-Modified" in response.headers:
                st.session_state[f"{cache_key}_last_modified"] = response.headers["Last-Modified"]

            # Parse HTML with BeautifulSoup
            logger.debug(f"Successfully fetched {url}, parsing content")
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script, style, and other non-content elements
            for element in soup(["script", "style", "meta", "link", "noscript"]):
                element.decompose()

            # Get text and clean it up
            text = soup.get_text(separator=" ", strip=True)
            
            # Remove excessive whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Truncate very long texts to avoid overwhelming the UI
            original_length = len(text)
            if original_length > 10000:
                text = text[:10000] + "... [content truncated]"
                logger.info(f"Truncated content from {original_length} to 10000 characters for {url}")
            else:
                logger.debug(f"Extracted {len(text)} characters of content from {url}")

            # Cache the processed content
            st.session_state[cache_key] = text
            return text

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return ""
        except Exception as e:
            logger.error(f"Failed to process content from {url}: {e}")
            return ""


@st.cache_data(show_spinner=False)
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using the configured tokenizer"""
    tokenizer = get_tokenizer()
    return tokenizer.tokenize(text)


@st.cache_data(show_spinner=False)
def compute_embeddings(sentences: List[str], _model: SentenceTransformer) -> np.ndarray:
    """Compute and cache sentence embeddings"""
    logger.debug(f"Computing embeddings for {len(sentences)} sentences")
    try:
        embeddings = _model.encode(sentences, convert_to_tensor=True).cpu().numpy()
        logger.debug(f"Successfully computed embeddings with shape {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error computing embeddings: {e}")
        raise


@st.cache_data(show_spinner=False)
def compute_sentence_similarities(
    llm_sentences: List[str], article_sentences: List[str], _model: SentenceTransformer
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cosine similarities between LLM response sentences and article sentences

    Returns:
        Tuple of (similarities matrix, llm_embeddings, article_embeddings)
    """
    logger.info(f"Computing similarities between {len(llm_sentences)} LLM sentences and {len(article_sentences)} article sentences")
    try:
        llm_embeddings = compute_embeddings(llm_sentences, _model)
        article_embeddings = compute_embeddings(article_sentences, _model)
        similarities = cosine_similarity(llm_embeddings, article_embeddings)
        logger.info(f"Successfully computed similarities matrix with shape {similarities.shape}")
        return similarities, llm_embeddings, article_embeddings
    except Exception as e:
        logger.error(f"Error computing sentence similarities: {e}")
        raise


@st.cache_data(show_spinner=False)
def get_top_similar_sentences(
    llm_sentence: str,
    article_sentences: List[str],
    article_url: str,
    _model,
    top_k: int = 3,
) -> List[Dict]:
    """Get top-k similar sentences for a single LLM sentence"""
    logger.debug(f"Finding top {top_k} similar sentences from article {article_url}")
    try:
        llm_embedding = compute_embeddings([llm_sentence], _model)
        article_embeddings = compute_embeddings(article_sentences, _model)
        similarities = cosine_similarity(llm_embedding, article_embeddings)[0]

        # Get indices of top-k similar sentences
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Create result list
        results = []
        for idx in top_indices:
            results.append({
                "sentence": article_sentences[idx],
                "similarity": float(similarities[idx]),
                "url": article_url,
            })
        
        logger.debug(f"Found {len(results)} similar sentences with scores: {[r['similarity'] for r in results]}")
        return results
    except Exception as e:
        logger.error(f"Error finding similar sentences: {e}")
        raise


@st.cache_data(show_spinner=False)
def compute_all_similarities(
    llm_sentences: List[str],
    search_response_df: pd.DataFrame,
    _model,
    top_k: int = 3,
    initial_candidates: int = 10
) -> Dict[str, List[Dict]]:
    """
    Two-stage similarity computation:
    1. Fast word overlap scoring to get initial candidates
    2. Compute embedding similarities only for top candidates

    Args:
        llm_sentences: List of sentences from LLM response
        search_response_df: DataFrame with search results
        _model: SentenceTransformer model
        top_k: Number of final similar sentences to return
        initial_candidates: Number of candidates to select in first stage

    Returns:
        Dictionary mapping each LLM sentence to its top-k similar sentences
    """
    logger.info(f"Computing similarities for {len(llm_sentences)} LLM sentences across {len(search_response_df)} articles")
    results = {}
    try:
        for i, llm_sentence in enumerate(llm_sentences, 1):
            logger.debug(f"Processing LLM sentence {i}/{len(llm_sentences)}")
            
            # Stage 1: Get initial candidates using word overlap
            logger.info(f"Stage 1: Computing word overlap scores for LLM sentence {i}")
            start_time = datetime.now()
            candidates = []
            total_sentences = 0
            for _, row in search_response_df.iterrows():
                overlap_results = get_top_sentences_by_word_overlap(
                    llm_sentence, row["Sentences"], row["URL"], initial_candidates
                )
                candidates.extend(overlap_results)
                total_sentences += len(row["Sentences"])
            stage1_time = datetime.now() - start_time
            logger.info(f"Stage 1 complete: Processed {total_sentences} sentences in {stage1_time.total_seconds():.2f}s")
            
            # Sort and get top candidates overall
            candidates.sort(key=lambda x: x['overlap_score'], reverse=True)
            top_candidates = candidates[:initial_candidates]
            
            # Stage 2: Compute embedding similarities only for top candidates
            logger.info(f"Stage 2: Computing embedding similarities for {len(top_candidates) if top_candidates else 0} candidates")
            start_time = datetime.now()
            if top_candidates:
                # Extract sentences and compute embeddings
                candidate_sentences = [c['sentence'] for c in top_candidates]
                similarities = cosine_similarity(
                    compute_embeddings([llm_sentence], _model),
                    compute_embeddings(candidate_sentences, _model)
                )[0]
                
                # Add similarity scores to candidates
                for idx, candidate in enumerate(top_candidates):
                    candidate['similarity'] = float(similarities[idx])
                    logger.debug(f"Candidate {idx+1}/{len(top_candidates)}: similarity={candidate['similarity']:.3f}, overlap_score={candidate['overlap_score']:.3f}")
                
                # Sort by embedding similarity and get top-k
                top_candidates.sort(key=lambda x: x['similarity'], reverse=True)
                results[llm_sentence] = top_candidates[:top_k]
                stage2_time = datetime.now() - start_time
                logger.info(f"Stage 2 complete: Processed {len(top_candidates)} candidates in {stage2_time.total_seconds():.2f}s")
            else:
                results[llm_sentence] = []
                logger.info("Stage 2 skipped: No candidates from stage 1")
            
            max_similarity = max(s["similarity"] for s in results[llm_sentence]) if results[llm_sentence] else 0
            logger.info(f"LLM sentence {i}: max similarity = {max_similarity:.3f}")
            logger.info("----------------------------------------")

        return results
    except Exception as e:
        logger.error(f"Error computing similarities across all sentences: {e}")
        raise


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def search_response_to_dataframe(search_response: dict | None) -> pd.DataFrame | None:
    """
    Convert Brave Search API response to a pandas DataFrame.
    Uses parallel processing to fetch webpage content efficiently.

    Args:
        search_response: JSON response from Brave Search API

    Returns:
        DataFrame with Title, URL, Content and sentence-level similarity data
    """
    logger.info("Converting search response to DataFrame")
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

        logger.info(f"Processing {len(results)} search results")

        # Create DataFrame with required columns first
        df = pd.DataFrame(results)[["title", "url"]].rename(
            columns={"title": "Title", "url": "URL"}
        )

        # Fetch content for each URL in parallel
        with st.spinner("Fetching article contents..."):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Start all fetch tasks
                future_to_url = {
                    executor.submit(fetch_webpage_content, url): url 
                    for url in df["URL"]
                }
                
                # Create a dict to store results
                content_dict = {}
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        content_dict[url] = content
                        logger.debug(f"Successfully fetched content from {url}")
                    except Exception as e:
                        logger.error(f"Failed to fetch content from {url}: {e}")
                        content_dict[url] = ""

                # Add content to dataframe
                df["Content"] = df["URL"].map(content_dict)

        # Drop rows with empty content
        df = df.dropna(subset=["Content"]).copy()

        if len(df) == 0:
            logger.warning("No valid content found in search results")
            st.warning("Could not fetch content from any search results")
            return None

        # Add sentence tokenization
        df["Sentences"] = df["Content"].apply(split_into_sentences)
        df["Num_Sentences"] = df["Sentences"].apply(len)
        
        total_sentences = df["Num_Sentences"].sum()
        logger.info(f"Successfully processed {len(df)} articles with {total_sentences} total sentences")
        return df

    except Exception as e:
        logger.exception("Error converting search results to DataFrame")
        st.error(f"Error processing search results: {str(e)}")
        return None


@st.cache_data(show_spinner=False)
def compute_word_overlap_score(sentence1: str, sentence2: str) -> float:
    """
    Compute a simple word overlap score between two sentences.
    
    Args:
        sentence1: First sentence
        sentence2: Second sentence
        
    Returns:
        Float score between 0 and 1
    """
    # Normalize and tokenize sentences
    words1 = set(word.lower() for word in sentence1.split())
    words2 = set(word.lower() for word in sentence2.split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return 0.0
        
    # Compute Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0


@st.cache_data(show_spinner=False)
def get_top_sentences_by_word_overlap(
    llm_sentence: str,
    article_sentences: List[str],
    article_url: str,
    top_k: int = 10
) -> List[Dict]:
    """
    Get top-k sentences based on word overlap score.
    
    Args:
        llm_sentence: The sentence to compare against
        article_sentences: List of sentences from the article
        article_url: URL of the article
        top_k: Number of top sentences to return
        
    Returns:
        List of dicts with sentence, score, and URL
    """
    scores = []
    for sentence in article_sentences:
        score = compute_word_overlap_score(llm_sentence, sentence)
        scores.append({
            'sentence': sentence,
            'overlap_score': score,
            'url': article_url
        })
    
    # Sort by score and get top-k
    scores.sort(key=lambda x: x['overlap_score'], reverse=True)
    return scores[:top_k]


st.set_page_config(
    page_title="Grounding LLMs with Web Search", page_icon=":mag_right:", layout="wide"
)

with st.spinner("Loading resources..."):
    google_client = get_google_client()
if google_client is None:
    st.error("Failed to create Google client. Please check your API key.")
    st.stop()


st.title("Grounding LLMs with Web Search")
st.write(
    "This app demonstrates how to find web citations for LLM responses. Enter a query on the left and press 'Run' to see the results."
)
st.selectbox(
        "Query",
        [
            "what is the economic impact of artificial intelligence on developing nations?",
            "explain the concept of quantum entanglement in layman's terms.",
            "compare and contrast the political systems of the United Kingdom and Canada.",
            "what are the ethical considerations surrounding gene editing technology?",
            "describe the role of the hippocampus in memory formation.",
            "analyze the causes and consequences of the deforestation in the Amazon basin.",
            "what are the key differences between a Roth IRA and a traditional IRA?",
            "explain the principles behind blockchain technology and its potential applications beyond cryptocurrency.",
            "discuss the impact of social media on political polarization.",
            "what are the major challenges facing the implementation of renewable energy sources on a global scale?"
        ],
        key="query",
        accept_new_options=True,
        placeholder="Select a query to run or enter a new one.",
        index=None,
    )

run_button = st.button(
    "Run",
    key="run",
    help="Click to run the search and LLM response.",
    disabled=not st.session_state.query,)

with st.sidebar:
    st.header("Parameters")
    corroboration_score_threshold = st.slider(
        "Corroboration Score Threshold",
        0.0,
        1.0,
        0.7,
        step=0.01,
        key="corroboration_score_threshold",
        help="Minimum similarity score to consider a sentence as corroborated.",
    )
    st.slider("Number of Search Results", 1, 20, 10, key="num_results")
        
    st.checkbox("Debug mode", True, key="debug_mode", help="Show raw search and LLM response.")
    clear_cache_button = st.button(
        "Clear Cache",
        key="clear_cache",
        help="Clear the cache to refresh data.",
        disabled=not st.session_state.query,
    )
    if clear_cache_button:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared successfully.")
        st.session_state.clear()
    with st.expander("API Keys", expanded=False):
        st.subheader("Brave Search")
        st.text_input(
            "Brave Search API Key",
            type="password",
            key="brave_search_api_key",
            help="Brave Search API key.",
        )
        st.subheader("Google Gemini")
        st.text_input(
            "Gemini API Key", type="password", key="gemini_api_key", help="Gemini API key."
        )

if run_button:
    # Load API keys
    brave_api_key = st.session_state.get("brave_search_api_key", "").strip()
    gemini_api_key = st.session_state.get("gemini_api_key", "").strip()
    
    # Try to get API keys from secrets if not in session state
    if not brave_api_key:
        try:
            brave_api_key = st.secrets.get("brave_search_api_key", "").strip()
        except Exception as e:
            logger.error(f"Error accessing secrets: {e}")
            brave_api_key = ""
            
    if not gemini_api_key:
        try:
            gemini_api_key = st.secrets.get("gemini_api_key", "").strip()
        except Exception as e:
            logger.error(f"Error accessing secrets: {e}")
            gemini_api_key = ""
    
    # Validate API keys
    if not brave_api_key:
        st.error("Please enter your Brave Search API key in the sidebar or set it in secrets.")
        st.stop()
    if not gemini_api_key:
        st.error("Please enter your Gemini API key in the sidebar or set it in secrets.")
        st.stop()
    
    # Load the sentence transformer model
    sentence_model = get_sentence_transformer()

    with st.spinner("Fetching search results..."):    
        search_results = search_brave(
            st.session_state.query,
            brave_api_key,
            st.session_state.num_results
        )
    if not search_results:
        st.error("Failed to get search results.")
        st.stop()
    search_response_df = search_response_to_dataframe(search_results)
    if search_response_df is None or search_response_df.empty:
        st.error("Failed to parse search results into dataframe.")
        st.stop()

    with st.spinner("Fetching LLM response..."):
        response = get_gemini_response(st.session_state.query, gemini_api_key)
    if response:
        st.session_state.llm_response = response
        # Split LLM response into sentences
        llm_sentences = split_into_sentences(response)
        st.session_state.llm_sentences = llm_sentences
    else:
        st.error("Failed to get response from LLM.")
        st.stop()

    if st.session_state.debug_mode:
        st.success(
            "Successfully fetched {} search results.".format(len(search_response_df))
        )
        with st.expander("Inspect Web Search Results", expanded=False):
            st.dataframe(
                search_response_df[
                    [
                        "Title",
                        "URL",
                        "Content",
                    ]
                ],
                use_container_width=True,
            )

        st.success("Successfully fetched LLM response.")
        with st.expander("Inspect LLM Response", expanded=False):
            st.markdown(st.session_state.llm_response)
        st.markdown("-----")

    # Remove markdown formatting from LLM response
    llm_sentences = [
        sentence.replace("`", "").replace("*", "").replace("_", "")
        for sentence in llm_sentences
    ]

    # Remove sentences that are too short
    llm_sentences = [
        sentence for sentence in llm_sentences if len(sentence.split()) > 5
    ]

    with st.spinner("Computing scores..."):
        # Compute and store top similar sentences for each LLM sentence
        top_similarities = compute_all_similarities(
            llm_sentences, search_response_df, sentence_model
        )
    st.session_state.top_similarities = top_similarities
    for i, sent in enumerate(llm_sentences):
        similar_sentences = top_similarities[sent]
        corroboration_score = max(s['similarity'] for s in similar_sentences) if similar_sentences else 0
        background_color = '#d4edda' if corroboration_score >= corroboration_score_threshold else '#f8d7da'
        
        
        st.markdown(
                f'<div style="padding: 10px; background-color: {background_color}; border-radius: 5px;">{sent}</div>',
                unsafe_allow_html=True
            )
        with st.expander(
            f"(corroboration_score = {corroboration_score:.2f})",
            expanded=False,
        ):
           
            
            if similar_sentences:
                # Create DataFrame with explicit column ordering
                df = pd.DataFrame(similar_sentences)
                # Ensure all required columns exist before selecting them
                columns_to_show = []
                if "similarity" in df.columns:
                    columns_to_show.append("similarity")
                if "sentence" in df.columns:
                    columns_to_show.append("sentence")
                if "url" in df.columns:
                    columns_to_show.append("url")
                elif "URL" in df.columns:
                    columns_to_show.append("URL")
                
                if columns_to_show:
                    # Apply background color based on similarity score
                    def color_similarity(row):
                        color = '#d4edda' if row['similarity'] >= 0.8 else '#f8d7da'
                        return [f'background-color: {color}'] * len(columns_to_show)
                    
                    # Apply the styling to the entire row based on similarity score
                    styled_df = df[columns_to_show].style.apply(color_similarity, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.error("No valid columns found in similarity results")
