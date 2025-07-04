import streamlit as st
import uuid
from src.chat.rag import process_subtitle
from src.base.llm_model import get_llm
from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- Session State Setup ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "video_info" not in st.session_state:
    st.session_state.video_info = ""
if "session_id" not in st.session_state:
    st.session_state.session_id = "session-" + str(uuid.uuid4())

# --- Caching Models ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def load_llm_model():
    return get_llm()

# --- Utilities ---
def add_message(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

def clear_chat():
    st.session_state.chat_history = []

def display_chat():
    if not st.session_state.chat_history:
        add_message("assistant", "Xin chào! Dán link YouTube và đặt câu hỏi về nội dung video nhé.")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Main App ---
def main():
    st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide", initial_sidebar_state="expanded")
    st.title("YouTube RAG Assistant")

    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Cài đặt & Điều khiển")

        if not st.session_state.models_loaded:
            with st.spinner("Đang tải mô hình..."):
                st.session_state.embeddings = load_embeddings()
                st.session_state.llm = load_llm_model()
                st.session_state.models_loaded = True
            st.success("Mô hình đã sẵn sàng!")
            st.rerun()
        else:
            st.success("Mô hình đã tải")

        st.markdown("---")

        st.subheader("Nhập URL Video")
        youtube_url = st.text_input("Dán link YouTube", placeholder="https://www.youtube.com/watch?v=...")
        langs = st.multiselect("Chọn phụ đề", ["vi", "en"], default=["vi", "en"])

        if st.button("Xử lý Video", use_container_width=True, disabled=not youtube_url):
            with st.spinner("Đang xử lý phụ đề..."):
                chain, num_chunks = process_subtitle(youtube_url, langs, history_folder="chat_history", max_history_length=20)
                if chain:
                    st.session_state.rag_chain = chain
                    st.session_state.video_processed = True
                    st.session_state.video_info = youtube_url
                    clear_chat()
                    add_message("assistant", f"Video được xử lý! Có {num_chunks} đoạn phụ đề.\n\nBắt đầu đặt câu hỏi phía dưới 👇")
                    st.rerun()

        st.markdown("---")
        if st.session_state.video_processed:
            st.success("Video đã được phân tích")
            st.caption(st.session_state.video_info)
        else:
            st.info("Chưa có video được xử lý")

        st.subheader("Quản lý Chat")
        if st.button("Xóa lịch sử", use_container_width=True):
            clear_chat()
            st.rerun()

        st.subheader("Hướng dẫn")
        st.markdown("""
        **Cách sử dụng:**
        1. Dán URL YouTube ở trên và nhấn **Xử lý Video**.
        2. Đặt câu hỏi bên dưới.
        3. Nhận câu trả lời dựa trên nội dung video.
        """)

    # --- Main Chat Panel ---
    st.markdown("### Chat")
    display_chat()

    # Input box
    if user_input := st.chat_input("Đặt câu hỏi về video...", disabled=not st.session_state.video_processed):
        add_message("user", user_input)
        st.rerun()

    # Generate assistant reply
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        question = st.session_state.chat_history[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                try:
                    output = st.session_state.rag_chain.invoke(
                        {"question": question},
                        config={"configurable": {"session_id": st.session_state.session_id}},
                    )
                    if "Answer:" in output:
                        answer = output.split("Answer:")[1].strip()
                    else:
                        answer = output.strip()
                    st.markdown(answer)
                    add_message("assistant", answer)
                except Exception as e:
                    err_msg = f"Lỗi khi xử lý: {str(e)}"
                    st.error(err_msg)
                    add_message("assistant", err_msg)

if __name__ == "__main__":
    main()
