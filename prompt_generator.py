"""Prompt generation utilities for the research summary app."""

import json
from pathlib import Path

from langchain_core.prompts import PromptTemplate


TEMPLATE_PATH = Path(__file__).with_name("template.json")


def _load_summary_template_text() -> str:
    """Read summary template text from template.json."""
    with TEMPLATE_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    template_text = data.get("summary_template")
    if not isinstance(template_text, str) or not template_text.strip():
        raise ValueError("template.json is missing a non-empty 'summary_template' string.")

    return template_text


SUMMARY_TEMPLATE = PromptTemplate(
    template=_load_summary_template_text(),
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
)


def build_summary_prompt(paper_input: str, style_input: str, length_input: str) -> str:
    """Return a formatted prompt for summarizing a research paper."""
    return SUMMARY_TEMPLATE.format(
        paper_input=paper_input,
        style_input=style_input,
        length_input=length_input,
    )

