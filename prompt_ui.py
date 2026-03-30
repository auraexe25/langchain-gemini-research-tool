"""Streamlit UI for Research Tool with Dynamic Prompting and Model Fallback."""

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from prompt_generator import build_summary_prompt

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
            # Continue for both NOT_FOUND and RESOURCE_EXHAUSTED (quota limit) errors
            if "NOT_FOUND" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                raise
        except (ValueError, RuntimeError, TypeError) as exc:
            last_error = exc

    raise RuntimeError(f"No working Gemini model found. Last error: {last_error}")


# --- Dynamic UI Inputs ---
paper_input = st.selectbox(
    "Select Research Paper Name", 
    ["Attention Is All You Need", 
     "BERT: Pre-training of Deep Bidirectional Transformers", 
     "GPT-3: Language Models are Few-Shot Learners", 
     "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select Explanation Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length", 
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# --- Execution Logic ---
if st.button("Summarise"):
    with st.spinner("Generating summary..."):
        try:
            # 1. Inject UI values into the reusable template.
            formatted_prompt = build_summary_prompt(
                paper_input=paper_input,
                style_input=style_input,
                length_input=length_input
            )
            
            # 2. Pass the formatted string to your fallback function
            result, used_model = invoke_with_fallback(formatted_prompt)
            
            # 3. Display results
            st.success("Summary generated successfully!")
            st.caption(f"Model used: {used_model}")
            st.write(result)
            
        except (ChatGoogleGenerativeAIError, ValueError, RuntimeError, TypeError) as exc:
            st.error(f"Request failed: {exc}")