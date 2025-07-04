import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from src.chat.rag import process_subtitle
from src.base.llm_model import get_llm
from src.chat.rag import process_subtitle

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
def load_llm_model():
    """Loads the large language model."""
    return get_llm()

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
            st.session_state.llm = get_llm()
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
            st.session_state.rag_chain, num_chunks = process_subtitle(youtube_url, langs, history_folder="chat_history", max_history_length=5)
            if st.session_state.rag_chain:
                st.success(f"Processing complete! The video was divided into {num_chunks} chunks.")

    # Q&A section
    session_id = st.session_state.get("session_id", "default-session")
    
    if st.session_state.rag_chain:
        question = st.text_input("Ask a question about the video:")
        if question:
            with st.spinner("Thinking..."):
                output = st.session_state.rag_chain.invoke(
                   {"question": question},
                    config={"configurable": {"session_id": session_id}}
                )
                # The Vicuna model prompt sometimes includes the question. This cleans the output.
                answer = output.split('Answer:')[-1].strip()
                st.write("**Answer:**")
                st.write(answer)

if __name__ == "__main__":
    main()
