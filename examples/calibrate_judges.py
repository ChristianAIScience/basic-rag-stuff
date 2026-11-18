"""Check the judges against hand-labelled cases before trusting their scores.

Run:  uv run python -m examples.calibrate_judges      (needs a real LLM_API_KEY, no database)

Each case has a known correct verdict (one good and one bad answer per judge). A judge that
disagrees with these obvious cases needs a better prompt or a stronger JUDGE_MODEL.
"""

from rag_lab.config import load_settings
from rag_lab.evaluation.dataset import CALIBRATION_CASES
from rag_lab.evaluation.judges import run_judge
from rag_lab.models import get_judge_model


def main() -> None:
    settings = load_settings()
    llm = get_judge_model(settings)
    print(f"Judge model: {settings.judge_model}\n")

    agreements = 0
    for case in CALIBRATION_CASES:
        result = run_judge(case.judge, llm, case.question, case.answer, case.documents, case.reference_answer)
        agrees = result.passed == case.expected
        agreements += agrees
        shown = case.answer or f"documents: {case.documents[0].metadata['source']}"
        print(f"{'agree   ' if agrees else 'DISAGREE'}  {case.judge:<20} expected={case.expected!s:<5} "
              f"judge={result.passed!s:<5} {shown[:60]}")
        if not agrees:
            print(f"          reasoning: {result.explanation[:300]}")

    print(f"\nAgreement with the human labels: {agreements}/{len(CALIBRATION_CASES)}")


if __name__ == "__main__":
    main()
