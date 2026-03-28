# Gemini-Powered AI Research Tool 🚀

A streamlined web application built with Streamlit and LangChain that leverages Google's Gemini models to assist with research tasks, such as summarizing complex academic papers.

## ✨ Features
* **Interactive UI:** A clean, easy-to-use web interface built with Streamlit.
* **LangChain Integration:** Utilizes LangChain's powerful wrappers to manage prompts and model invocations.
* **Robust Model Fallback:** Implements custom error-handling logic to automatically switch between Gemini models (e.g., falling back from `gemini-2.0-flash` to `gemini-1.5-flash`) if quota limits are reached or a model is temporarily unavailable.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Framework:** Streamlit
* **LLM Orchestration:** LangChain (`langchain-google-genai`)
* **Models:** Google Gemini API

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed on your machine. You will also need a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
```

2. **Create a virtual environment (recommended):**
```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. **Install the dependencies:**
```bash
   pip install streamlit langchain-google-genai python-dotenv
```

4. **Set up your environment variables:**
   Create a `.env` file in the root directory and add your Google API key:
```
   GOOGLE_API_KEY="your_actual_api_key_here"
```

### Running the App
Start the Streamlit server by running the following command in your terminal:
```bash
streamlit run prompt_ui.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

## 👨‍💻 Author
Veena Sahu
