from langchain_google_genai import ChatGoogleGenerativeAI
# from src.base.get_token import get_token
import os
import getpass


def get_llm(model: str = "gemini-2.0-flash", temperature: float = 0.0, max_tokens: int = None, **kwargs):
    """
    Initialize Gemini LLM via Google Generative AI using LangChain.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key:")
        print("Waiting API key from frontend...")
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("GOOGLE_API_KEY"),
        **kwargs
    )
    return llm