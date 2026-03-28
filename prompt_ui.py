"""Simple Streamlit UI for chatting with Gemini via LangChain."""

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError


def load_env_file(path: str = ".env") -> None:
    """Load key=value pairs from a local .env file if present."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

st.title("Research Tool")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing GOOGLE_API_KEY. Add it to your .env file and restart the app.")
    st.stop()

model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def invoke_with_fallback(prompt: str) -> tuple[str, str]:
    """Invoke Gemini and fall back to known model names if needed."""
    fallback_candidates = [
        model_name,
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]
    models_to_try = list(dict.fromkeys(fallback_candidates))
    last_error = None

    for candidate in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(model=candidate, google_api_key=api_key)
            response = llm.invoke(prompt)
            return response.content, candidate
        except ChatGoogleGenerativeAIError as exc:
            last_error = exc
            # Continue only for model-not-found style failures.
            if "NOT_FOUND" not in str(exc):
                raise
        except (ValueError, RuntimeError, TypeError) as exc:
            last_error = exc

    raise RuntimeError(f"No working Gemini model found. Last error: {last_error}")

user_input = st.text_input("Enter your prompt:")

if st.button("Summarise"):
    if not user_input:
        st.warning("Please enter your prompt first.")
    else:
        try:
            result, used_model = invoke_with_fallback(user_input)
            st.caption(f"Model used: {used_model}")
            st.write(result)
        except (ChatGoogleGenerativeAIError, ValueError, RuntimeError, TypeError) as exc:
            st.error(f"Request failed: {exc}")
