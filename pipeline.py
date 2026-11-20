"""영화 리뷰 한 건에서 정보를 추출하고 같은 원문의 감성을 판정한다."""

import argparse

from src.information_extractor import RegexInformationExtractor
from src.korean_sentiment import OktTokenizer, SentimentAnalyzer, load_assignment_lexicon


SAMPLE_REVIEW = "2024년 3월 15일 영화제에서 본 영화. 관람료는 12,000원. 정말 명작!"


def analyze_review(text: str) -> dict:
    # 원문에서 날짜·금액 등을 추출한다. 민감정보는 추출기의 기본 정책에 따라 마스킹한다.
    extracted = RegexInformationExtractor().extract(text)

    # 같은 원문을 Okt로 분석하고 확장 감성사전과 부정·강조 규칙으로 점수를 계산한다.
    tokenizer = OktTokenizer()
    analyzer = SentimentAnalyzer(load_assignment_lexicon(tokenizer))
    score = analyzer.analyze(text, tokenizer)["score"]

    # 05 평가 노트북과 같은 이진 판정 기준을 사용한다.
    return {
        "information": {kind: values for kind, values in extracted.items() if values},
        "score": score,
        "label": "긍정" if score > 0 else "부정",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=SAMPLE_REVIEW, help="영화 리뷰 한 건 (생략 시 예문 사용)")
    review = parser.parse_args().text
    result = analyze_review(review)

    print("추출 정보")
    for kind, values in result["information"].items():
        print(f"  {kind}: {', '.join(values)}")
    if not result["information"]:
        print("  없음")
    print(f"감성 판정: {result['label']} ({result['score']:+g}점)")


if __name__ == "__main__":
    main()
