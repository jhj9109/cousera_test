"""과제용 unittest: 9개 유형의 추출·정규화·마스킹을 검증한다.

샘플 카드 번호는 공개 테스트 번호이며 API Key 예시는 무효한 가상 문자열이다.
"""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.information_extractor import InformationNormalizer, PrivacyMasker, RegexInformationExtractor


# (입력, 정규화된 값, 최종 마스킹 결과)
CASES = {
    "email": [
        ("user@domain.com", "user@domain.com", "u***@domain.com"),
        ("user.name@sub.domain.co.kr", "user.name@sub.domain.co.kr", "u***@sub.domain.co.kr"),
        ("user+tag@DOMAIN.COM", "user+tag@domain.com", "u***@domain.com"),
    ],
    "phone": [
        ("010-1234-5678", "010-1234-5678", "010-****-5678"),
        ("02-123-4567", "02-123-4567", "02-***-4567"),
        ("031-1234-5678", "031-1234-5678", "031-****-5678"),
        ("01012345678", "010-1234-5678", "010-****-5678"),
    ],
    "date": [
        ("2024년 1월 15일", "2024-01-15", "2024-01-15"),
        ("2024/01/15", "2024-01-15", "2024-01-15"),
        ("2024-01-15", "2024-01-15", "2024-01-15"),
        ("2024.1.5", "2024-01-05", "2024-01-05"),
    ],
    "money": [
        ("10,000원", "10000", "10000"),
        ("1억 2천만원", "120000000", "120000000"),
        ("$100", "100", "100"),
        ("$1,234.50", "1234.50", "1234.50"),
        ("1경 2조 3억 4천만원", "10002000340000000", "10002000340000000"),
    ],
    "url": [
        ("http://example.com", "http://example.com", "http://example.com"),
        ("https://www.example.com/path?query=value", "https://www.example.com/path?query=value", "https://www.example.com/path?query=value"),
        ("HTTPS://WWW.Example.COM:8080/Path#Section", "https://www.example.com:8080/Path#Section", "https://www.example.com:8080/Path#Section"),
    ],
    "rrn": [
        ("900101-1234567", "900101-1234567", "******-1******"),
        ("030101 3123456", "030101-3123456", "******-3******"),
        ("9001011234567", "900101-1234567", "******-1******"),
    ],
    "card": [
        ("4111-1111-1111-1111", "4111111111111111", "****-****-****-1111"),
        ("5555 5555 5555 4444", "5555555555554444", "****-****-****-4444"),
        ("4111111111111111", "4111111111111111", "****-****-****-1111"),
    ],
    "address": [
        ("서울특별시 강남구 테헤란로 123", "서울특별시 강남구 테헤란로 123", "서울특별시 강남구 [상세주소 마스킹]"),
        ("경기도 성남시 분당구 판교역로 235", "경기도 성남시 분당구 판교역로 235", "경기도 성남시 분당구 [상세주소 마스킹]"),
        ("부산광역시 해운대구 우동 123-45", "부산광역시 해운대구 우동 123-45", "부산광역시 해운대구 [상세주소 마스킹]"),
    ],
    "api_key": [
        ("sk-abcdefghijklmnop1234", "sk-abcdefghijklmnop1234", "sk-********"),
        ("ghp_abcdefghijklmnopqrstuvwxyz123456", "ghp_abcdefghijklmnopqrstuvwxyz123456", "ghp_********"),
        ("AKIAAAAAAAAAAAAAAAAA", "AKIAAAAAAAAAAAAAAAAA", "AKIA********"),
        ("github_pat_abcdefghijklmnopqrstuvwx1234", "github_pat_abcdefghijklmnopqrstuvwx1234", "github_pat_********"),
    ],
}


class TestInformationNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = InformationNormalizer()

    def test_normalization_by_type(self):
        for kind, cases in CASES.items():
            for raw, expected, _ in cases:
                with self.subTest(kind=kind, raw_type=kind):
                    self.assertEqual(getattr(self.normalizer, f"normalize_{kind}")(raw), expected)

    def test_invalid_date(self):
        for raw in ("2024-02-30", "2024-13-01", "2023-02-29"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    self.normalizer.normalize_date(raw)

    def test_invalid_card_luhn(self):
        with self.assertRaises(ValueError):
            self.normalizer.normalize_card("4111-1111-1111-1112")

    def test_invalid_rrn_birth_date(self):
        with self.assertRaises(ValueError):
            self.normalizer.normalize_rrn("991332-1234567")

    def test_compound_korean_money(self):
        self.assertEqual(
            self.normalizer.normalize_money("1억 2천3백만원"),
            "123000000"
        )
    def test_money_unit_groups_and_invalid_order(self):
        for raw, expected in [("2천3백만원", "23000000"), ("1억 2천3백4십5만6천7백8십9원", "123456789"), ("0원", "0")]:
            with self.subTest(raw=raw):
                self.assertEqual(self.normalizer.normalize_money(raw), expected)
        for raw in ("1만2억원", "1백2천원", "1억2억원"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.normalizer.normalize_money(raw)

class TestRegexInformationExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = RegexInformationExtractor()

    def test_extraction_by_type(self):
        for kind, cases in CASES.items():
            for raw, _, masked in cases:
                with self.subTest(kind=kind, example=raw if kind not in {"rrn", "card", "api_key"} else "synthetic"):
                    actual = self.extractor.extract(f"데이터: {raw}")
                    self.assertIn(masked, actual[kind])

    def test_full_pipeline(self):
        text = (
            "이메일 user@domain.com, 전화 010-1234-5678, "
            "날짜 2024년 1월 15일, 금액 1억 2천만원, "
            "사이트 https://www.example.com/path, "
            "식별번호 900101-1234567, 카드 4111-1111-1111-1111, "
            "주소 서울특별시 강남구 테헤란로 123, "
            "토큰 sk-abcdefghijklmnop1234"
        )
        expected = {
            "email": ["u***@domain.com"],
            "phone": ["010-****-5678"],
            "date": ["2024-01-15"],
            "money": ["120000000"],
            "url": ["https://www.example.com/path"],
            "rrn": ["******-1******"],
            "card": ["****-****-****-1111"],
            "address": ["서울특별시 강남구 [상세주소 마스킹]"],
            "api_key": ["sk-********"],
        }
        self.assertEqual(self.extractor.extract(text), expected)
        # 원문 숫자 식별자, API Key 등은 최종 출력에 포함되면 안 됨
        output = str(self.extractor.extract(text))
        for secret in ("900101-1234567", "4111111111111111", "sk-abcdefghijklmnop1234"):
            self.assertNotIn(secret, output)

    def test_invalid_candidates_excluded(self):
        result = self.extractor.extract("날짜 2024-02-30, 카드 4111-1111-1111-1112")
        self.assertEqual(result["date"], [])
        self.assertEqual(result["card"], [])

    def test_multiple_matches(self):
        result = self.extractor.extract("연락처 010-1234-5678, 02-123-4567")
        self.assertEqual(result["phone"], ["010-****-5678", "02-***-4567"])

    def test_empty_input(self):
        self.assertEqual(
            self.extractor.extract(""),
            {kind: [] for kind in self.extractor.PATTERNS},
        )

    def test_url_credentials_redacted(self):
        url = "https://example.com/path?api_key=abcdefghijklmnop&query=hello"
        result = self.extractor.extract(url)
        self.assertEqual(result["url"], [
            "https://example.com/path?api_key=********&query=hello"
        ])

    def test_api_key_inside_url_is_masked(self):
        text = (
            "https://example.com/callback"
            "?state=sk-abcdefghijklmnop1234"
        )

        result = self.extractor.extract(text)

        self.assertNotIn(
            "sk-abcdefghijklmnop1234",
            result["url"][0]
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)