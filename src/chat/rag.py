
from tools.get_subtiltes import get_subtitles
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from tools.get_subtiltes import get_subtitles
from src.chat.history import create_session_factory
from tools.output_parser import format_docs

import streamlit as st

def process_subtitle(video_url: str, language: list, history_folder: str, max_history_length: int):
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
    question_to_string = RunnableLambda(lambda x: x["question"])
    retrieval_pipeline = question_to_string | retriever | format_docs
    
    rag_chain = (
        {"context": retrieval_pipeline, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        create_session_factory(base_dir=history_folder, 
                               max_history_length=max_history_length),
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return chain_with_history, len(subs)