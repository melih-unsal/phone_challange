"""Layer 2 — country / linguistic-origin detection from the transcript."""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import LLM_BASE_URL, LLM_MODEL_NAME
from src.prompts import COUNTRY_HUMAN, COUNTRY_SYSTEM


def build_country_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", COUNTRY_SYSTEM), ("human", COUNTRY_HUMAN)]
    )
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=0,
        max_tokens=400,
        base_url=LLM_BASE_URL,
    )
    return prompt | llm | JsonOutputParser()


def detect_country(chain, transcription: str) -> dict:
    try:
        info = chain.invoke({"transcription": transcription})
    except Exception:
        info = {}
    return {
        "country": info.get("country", "") or "",
        "language_code": info.get("language_code", "") or "",
        "reasoning": info.get("reasoning", "") or "",
    }
