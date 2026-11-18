"""Evaluation data: questions with reference answers, and hand-labelled cases for the judges.

EVAL_EXAMPLES     - a small "golden" dataset. Each reference answer was written by a human
                    from the files in data/. Only the correctness judge uses it.
CALIBRATION_CASES - answers whose correct verdict is known. Running the judges on them
                    shows whether a judge agrees with a human before you trust its scores.
"""

from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass(frozen=True)
class EvalExample:
    question: str
    reference_answer: str


EVAL_EXAMPLES = [
    EvalExample(
        "How many paid vacation days do full-time employees get per year?",
        "27 paid vacation days per year; employees with more than five years at Lumora get 3 additional loyalty days.",
    ),
    EvalExample(
        "How many unused vacation days can be carried over, and when do they expire?",
        "Up to 5 unused days can be carried over; they expire on March 31st of the following year.",
    ),
    EvalExample(
        "How many days per week can employees work remotely?",
        "Up to 3 days per week. Tuesdays are Anchor Days on which everyone is expected in the office.",
    ),
    EvalExample(
        "How long does the Glowbeam X2 battery last at 50 % brightness?",
        "About 11 hours at 50 % brightness (about 6 hours at full brightness).",
    ),
    EvalExample(
        "How much does the Glowbeam X2 Studio edition cost?",
        "159 euros.",
    ),
    EvalExample(
        "What should I do when TunnelKite shows error TK-407?",
        "The device certificate has expired: open AppDock and run 'Renew Device Certificate'.",
    ),
    EvalExample(
        "What is the hotel limit per night on business trips?",
        "140 euros per night in most cities and 190 euros per night in London, Paris and Zurich.",
    ),
    EvalExample(
        "Where is the fire assembly point in Leipzig?",
        "Parking lot P2.",
    ),
    EvalExample(
        # Not in the documents: a correct answer says that the information is not available.
        "What is Lumora's policy on bringing dogs to the office?",
        "The documents contain no information about this; the assistant should say that it does not know.",
    ),
]


@dataclass(frozen=True)
class CalibrationCase:
    judge: str                  # correctness | relevance | groundedness | retrieval_relevance
    question: str
    answer: str
    expected: bool              # the verdict a human gives
    reference_answer: str = ""
    documents: list[Document] = field(default_factory=list)


_VACATION_DOC = Document(
    page_content="Every full-time employee at Lumora Systems receives 27 paid vacation days per calendar year.",
    metadata={"source": "hr/vacation_policy.md"},
)
_PRICING_DOC = Document(
    page_content="Glowbeam X2 standard edition: 129 euros. Studio edition with a clamp mount: 159 euros.",
    metadata={"source": "products/pricing.md"},
)
_QUESTION = "How many paid vacation days do full-time employees get per year?"

CALIBRATION_CASES = [
    CalibrationCase("correctness", _QUESTION, "Full-time employees get 27 paid vacation days a year.", True,
                    reference_answer="27 paid vacation days per year."),
    CalibrationCase("correctness", _QUESTION, "Full-time employees get 30 paid vacation days a year.", False,
                    reference_answer="27 paid vacation days per year."),
    CalibrationCase("relevance", _QUESTION, "You get 27 paid vacation days per year.", True),
    CalibrationCase("relevance", _QUESTION, "The Glowbeam X2 lamp costs 129 euros.", False),
    CalibrationCase("groundedness", _QUESTION, "Full-time employees receive 27 paid vacation days.", True,
                    documents=[_VACATION_DOC]),
    CalibrationCase("groundedness", _QUESTION,
                    "Full-time employees receive 27 paid vacation days and a yearly bonus of 2,000 euros.", False,
                    documents=[_VACATION_DOC]),
    CalibrationCase("retrieval_relevance", _QUESTION, "", True, documents=[_VACATION_DOC]),
    CalibrationCase("retrieval_relevance", _QUESTION, "", False, documents=[_PRICING_DOC]),
]
