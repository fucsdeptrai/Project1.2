import streamlit as st
import torch
from urllib.parse import urlparse, parse_qs

# LangChain and related imports
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Transformers and YouTube API imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# Session state initialization
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

# --- Core Functions ---

@st.cache_resource
def load_embeddings():
    """Loads the sentence embedding model."""
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def load_llm():
    """Loads the large language model with quantization."""
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=model_pipeline)

def get_video_id(url: str) -> str | None:
    """
    Extracts the video ID from various YouTube URL formats.
    """
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]

    # Fallback for the specific googleusercontent format in the original code
    if 'googleusercontent.com/youtube.com/' in url:
        try:
            # This parsing is based on the original user's code structure
            video_id_part = url.split('/')[-1]
            return video_id_part.split('?')[0].split('&')[0]
        except IndexError:
            return None # Invalid format

    return None

def get_subtitles(url: str, language: list =['vi', 'en']) -> str | None:
    """
    Fetches and formats the transcript for a given YouTube video URL.
    """
    video_id = get_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL. Could not extract video ID.")
        return None

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            # Find a transcript in the specified languages
            transcript = transcript_list.find_transcript(language)
        except NoTranscriptFound:
            # If not found, try to find an auto-generated one
            transcript = transcript_list.find_generated_transcript(language)

        fetched_transcript = transcript.fetch()
        formatter = TextFormatter()
        return formatter.format_transcript(fetched_transcript)

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        st.error(f"Could not retrieve transcript: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching subtitles: {e}")
        return None

def process_subtitle(video_url: str, language: list):
    """
    Processes subtitles to create a RAG chain for Q&A.
    """
    subtitles_text = get_subtitles(video_url, language)
    if not subtitles_text:
        return None, 0
    
    # 1. Create Document objects from the subtitle text
    # The SemanticChunker works on a list of Document objects.
    docs = [Document(page_content=subtitles_text)]
    
    # 2. Initialize the Semantic Chunker
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500, # Adjust if needed
        add_start_index=True
    )
    
    # 3. Split documents into semantic chunks
    subs = semantic_splitter.split_documents(docs)
    
    # 4. Create a Chroma vector database and retriever
    vector_db = Chroma.from_documents(documents=subs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever()
    
    # 5. Define the RAG prompt and chain
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )
    
    return rag_chain, len(subs)

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")
    st.title("YouTube Video RAG Assistant 💬")

    st.markdown("""
    **AI application to help you ask questions directly about the content of a YouTube video using its subtitles.**

    **Simple to use:**
    1.  **Enter URL** → Paste a YouTube video URL and click "Process Video".
    2.  **Ask a Question** → Type a question about the video's content and get an answer instantly.
    ---
    """)

    # Load models only once
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models... This may take a moment."):
            st.session_state.embeddings = load_embeddings()
            st.session_state.llm = load_llm()
            st.session_state.models_loaded = True
        st.success("Models are ready!")
        st.rerun()

    # Input for YouTube URL
    youtube_url = st.text_input("Enter a YouTube video URL:", placeholder="https://www.youtube.com/watch?v=...")
    langs = st.multiselect(
        "Select subtitle languages (defaults to Vietnamese and English):",
        options=["vi", "en"], # Added more options
    )

    if youtube_url and st.button("Process Video"):
        with st.spinner("Processing video subtitles..."):
            st.session_state.rag_chain, num_chunks = process_subtitle(youtube_url, langs)
            if st.session_state.rag_chain:
                st.success(f"Processing complete! The video was divided into {num_chunks} chunks.")

    # Q&A section
    if st.session_state.rag_chain:
        question = st.text_input("Ask a question about the video:")
        if question:
            with st.spinner("Thinking..."):
                output = st.session_state.rag_chain.invoke(question)
                # The Vicuna model prompt sometimes includes the question. This cleans the output.
                answer = output.split('Answer:')[-1].strip()
                st.write("**Answer:**")
                st.write(answer)

if __name__ == "__main__":
    main()
