"""Layer 3 — Candidate 1.

Self-consistent LLM extraction over the transcript. The chain runs
NUM_LLM_VOTES times in parallel, and we collapse the results with a
per-key majority vote so that LLM stochasticity is averaged out.
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import KEYS, LLM_BASE_URL, LLM_MODEL_NAME, NUM_LLM_VOTES
from src.prompts import EXTRACT_HUMAN, EXTRACT_SYSTEM
from src.utils import majority_vote


def build_extract_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", EXTRACT_SYSTEM), ("human", EXTRACT_HUMAN)]
    )
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=0.7,
        max_tokens=1200,
        base_url=LLM_BASE_URL,
    )
    return prompt | llm | JsonOutputParser()


def extract_candidate_1(chain, transcription: str, country: str, language_code: str) -> dict:
    """Run the extraction chain NUM_LLM_VOTES times in parallel and majority-vote.

    Returns a dict with the four caller-info keys plus a `_runs` list of the
    raw run dicts (useful for the reconciler / debugging).
    """
    chain_input = {
        "keys": KEYS,
        "country": country,
        "language_code": language_code,
        "transcription": transcription,
    }
    try:
        runs = chain.batch([chain_input] * NUM_LLM_VOTES)
    except Exception:
        runs = []
        for _ in range(NUM_LLM_VOTES):
            try:
                runs.append(chain.invoke(chain_input))
            except Exception:
                pass

    voted = majority_vote(runs, KEYS) if runs else {k: "" for k in KEYS}
    voted["_runs"] = runs
    return voted
