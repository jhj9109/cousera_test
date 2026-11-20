"""정규표현식 기반 정보 추출기 (학습 과제용).

- 원래 5개 유형: email, phone, date, money, url
- 보너스 4개 유형: rrn(주민번호), card, address, api_key
- 정규화 -> 민감정보 마스킹 -> 반환
- 카드/주소/API Key는 목적에 맞게 인식 범위를 제한하며 만능 검증기가 아님
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit, urlunsplit


EMAIL_PATTERN = re.compile(r"""
    (?<![\w.+-])                 # 이메일 중간부터 시작하지 않음
    [A-Za-z0-9_%+-]+            # 로컬 첫 구간
    (?:\.[A-Za-z0-9_%+-]+)*     # .구간 반복
    @                           # @
    [A-Za-z0-9-]+               # 도메인 라벨
    (?:\.[A-Za-z0-9-]+)+        # 점으로 구분된 도메인
    (?![\w-])                  # 단어 중간에서 종료하지 않음
""", re.VERBOSE)

# 공급자 접두사가 있는 후보만 인식하며 임의 비밀값 전체를 탐지하지 않는다.
API_KEY_PATTERN = re.compile(r"""
            (?<![A-Za-z0-9_])
            (?:
                sk-(?:proj-)?[A-Za-z0-9_-]{16,}
              | sk_(?:test|live)_[A-Za-z0-9]{16,}
              | gh[pousr]_[A-Za-z0-9]{20,}
              | github_pat_[A-Za-z0-9_]{20,}
              | AKIA[0-9A-Z]{16}
            )
            (?![A-Za-z0-9_-])
        """, re.VERBOSE)


class InformationNormalizer:
    """추출된 문자열을 의미 있는, 일관된 문자열로 정규화한다."""

    # 작은 단위를 합산한 뒤 만/억/조/경 단위로 묶는다.
    SMALL_UNITS = {"천": 1000, "백": 100, "십": 10}
    LARGE_UNITS = {"만": 10**4, "억": 10**8, "조": 10**12, "경": 10**16}
    _MONEY_TOKEN = re.compile(r"[0-9]+|[천백십만억조경]")

    def normalize_email(self, value: str) -> str:
        """로컬 부분은 보존하고 도메인만 소문자화."""
        local, domain = value.rsplit("@", 1)
        return f"{local}@{domain.lower()}"

    def normalize_phone(self, value: str) -> str:
        """한국 번호를 국번 길이에 맞춘 하이픈 형식으로 통일."""
        digits = re.sub(r"\D", "", value)
        if digits.startswith("02") and len(digits) in (9, 10):
            middle = digits[2:-4]
            return f"02-{middle}-{digits[-4:]}"
        if len(digits) in (10, 11) and digits.startswith("0"):
            return f"{digits[:3]}-{digits[3:-4]}-{digits[-4:]}"
        raise ValueError("지원하지 않는 전화번호 형식")

    def normalize_date(self, value: str) -> str:
        """실제 달력 날짜인지 확인한 뒤 YYYY-MM-DD로 반환."""
        numbers = re.findall(r"\d+", value)
        if len(numbers) != 3:
            raise ValueError("날짜 구성요소는 연/월/일 3개여야 함")
        year, month, day = map(int, numbers)
        return date(year, month, day).isoformat()

    def normalize_money(self, value: str) -> str:
        """통화 기호 없이 값만 반환 (예: 1억 2천만원 -> '120000000')."""
        if value.startswith("$"):
            try:
                return str(Decimal(value[1:].strip().replace(",", "")))
            except InvalidOperation as exc:
                raise ValueError("유효하지 않은 달러 금액") from exc
        return str(self._parse_korean_money(value))

    def _parse_korean_money(self, value: str) -> int:
        """한국어 숫자 단위를 배율로 환산하여 합산."""
        body = re.sub(r"[\s,]", "", value.removesuffix("원"))
        tokens = self._MONEY_TOKEN.findall(body)
        if not tokens or "".join(tokens) != body:
            raise ValueError("지원하지 않는 금액 표현")
        total = section = 0
        number = None
        last_small, last_large = 10000, 10**20
        for token in tokens:
            if token.isdigit():
                number = int(token)
            elif token in self.SMALL_UNITS:
                unit = self.SMALL_UNITS[token]
                if unit >= last_small:
                    raise ValueError("작은 금액 단위는 내림차순이어야 함")
                section += (1 if number is None else number) * unit
                number, last_small = None, unit
            else:
                unit = self.LARGE_UNITS[token]
                if unit >= last_large:
                    raise ValueError("큰 금액 단위는 내림차순이어야 함")
                subtotal = section + (0 if number is None else number)
                if not subtotal:
                    raise ValueError("큰 단위 앞에 금액이 필요함")
                total += subtotal * unit
                section, number, last_small, last_large = 0, None, 10000, unit
        return total + section + (0 if number is None else number)

    def normalize_url(self, value: str) -> str:
        """스킴과 호스트 소문자화; 경로/쿼리/프래그먼트 유지."""
        parts = urlsplit(value.rstrip(".,;!"))
        if not parts.hostname or parts.scheme.lower() not in {"http", "https"}:
            raise ValueError("지원하지 않는 URL")
        # 과제의 정규표현식은 userinfo, IPv6를 다루지 않음
        authority = parts.hostname.lower()
        if parts.port is not None:
            authority += f":{parts.port}"
        return urlunsplit((
            parts.scheme.lower(),  # https
            authority,             # www.example.com:8080
            parts.path,            # /Path
            parts.query,           # query=Value
            parts.fragment,        # Section
        ))

    def normalize_rrn(self, value: str) -> str:
        """주민등록번호(형식 후보): 6자리-7자리로 통일, 생년월일 기본 검증."""
        digits = re.sub(r"[ -]", "", value)
        if not re.fullmatch(r"\d{6}[1-8]\d{6}", digits):
            raise ValueError("주민등록번호 형식 불일치")
        century = 1900 if digits[6] in "1256" else 2000
        date(century + int(digits[:2]), int(digits[2:4]), int(digits[4:6]))
        # 실제 번호인지 확인하는 것이 아니라 생년월일과 외형만 확인
        return digits[:6] + "-" + digits[6:]

    def normalize_card(self, value: str) -> str:
        """16자리 카드번호 후보를 연속 16자리 숫자로 변환하고 Luhn 검증."""
        digits = re.sub(r"[ -]", "", value)
        if not re.fullmatch(r"\d{16}", digits) or not self._luhn_valid(digits):
            raise ValueError("유효하지 않은 카드번호 후보")
        return digits

    @staticmethod
    def _luhn_valid(digits: str) -> bool:
        """주요 카드번호가 사용하는 Luhn 검사합 (발급 여부 검증은 아님)."""
        total = 0
        for index, digit in enumerate(reversed(digits)):
            n = int(digit)
            if index % 2:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    def normalize_address(self, value: str) -> str:
        """지원 범위의 한국 주소에서 공백을 한 칸으로 정리."""
        return " ".join(value.split())

    def normalize_api_key(self, value: str) -> str:
        """토큰의 대소문자는 변경하지 않고 앞뒤 공백만 정리."""
        return value.strip()


class PrivacyMasker:
    """민감한 정규화 결과를 사람이 식별하기 어렵도록 마스킹한다."""
    API_PREFIXS = [
        "github_pat_", "sk-proj-", "sk_test_", "sk_live_",
        "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "sk-", "AKIA",
    ]

    def mask_email(self, value: str) -> str:
        local, domain = value.rsplit("@", 1)
        return f"{local[:1]}***@{domain}"

    def mask_phone(self, value: str) -> str:
        head, middle, tail = value.split("-")
        return f"{head}-{'*' * len(middle)}-{tail}"

    def mask_rrn(self, value: str) -> str:
        # 생년월일 6자리와 뒷번호 대부분을 모두 숨긴다.
        return f"******-{value[7]}******"

    def mask_card(self, value: str) -> str:
        return f"****-****-****-{value[-4:]}"

    def mask_address(self, value: str) -> str:
        # 매칭된 주소는 '(시/도) (시/군/구) ... (도로/동) (건물번호)' 형태
        # 상세 위치인 마지막 두 구간은 제거
        parts = value.split()
        return " ".join(parts[:-2]) + " [상세주소 마스킹]"

    def mask_api_key(self, value: str) -> str:
        for prefix in self.API_PREFIXS:
            if value.startswith(prefix):
                return prefix + "********"
        # 모르는 형식은 토큰 전체를 차단
        return "********"

    def mask_url(self, value: str) -> str:
        """URL 안의 이메일과 인증정보를 마스킹한다."""
        # URL 자체를 정규화해도 쿼리에 토큰이 담겨 있으면 민감정보가 노출될 수 있음
        value = API_KEY_PATTERN.sub(lambda m: self.mask_api_key(m.group()), value)
        value = EMAIL_PATTERN.sub(lambda m: self.mask_email(m.group()), value)
        return re.sub(
            r"(?i)([?&](?:api_key|access_token|token|secret|key)=)[^&#]*",
            r"\1********",
            value,
        )


MONEY_INTEGER = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
# 소수부까지 한 후보로 읽어 원화 정수 단위 검사에서 전체를 제외한다.
WON_NUMBER = rf"{MONEY_INTEGER}(?:\.\d+)?"


class RegexInformationExtractor:
    """텍스트에서 정규표현식 후보를 추출 -> 정규화 -> 마스킹."""

    # 추출 가능한 형식만 선언한다. 모든 이메일/주소/API Key의 완전한 파서는 아님.
    PATTERNS = {
        "email": EMAIL_PATTERN,

        "phone": re.compile(r"""
            (?<!\d)
            (?:
                010[- ]?\d{4}[- ]?\d{4}  # 휴대전화 010-1234-5678
              | 02[- ]?\d{3,4}[- ]?\d{4} # 서울 02-123-4567
              | 0[3-6]\d[- ]?\d{3,4}[- ]?\d{4} # 기타 지역 031-1234-5678
            )
            (?!\d)
        """, re.VERBOSE),

        "date": re.compile(r"""
            (?<!\d)
            (?:
                \d{4}년\s*\d{1,2}월\s*\d{1,2}일 # 한국어 날짜
              | \d{4}(?P<sep>[-/.])\d{1,2}(?P=sep)\d{1,2} # 동일 구분자
            )
            (?!\d)
        """, re.VERBOSE),

        "money": re.compile(rf"""
            (?<![\d,.])
            (?:
                \$\s*{MONEY_INTEGER}(?:\.\d{{1,2}})?(?![.,]\d) # 달러
              |
                (?:
                    {WON_NUMBER}\s*
                    (?:경|천조|백조|십조|조|천억|백억|십억|억|
                       천만|백만|십만|만|천|백|십)\s*
                )+
                (?:{WON_NUMBER}\s*)?
                원                         # 한국어 복합 단위 + 원
              |
                {WON_NUMBER}\s*원 # 숫자 + 원
            )
            (?!\w)
        """, re.VERBOSE),

        "url": re.compile(r"""
            (?<![\w@])
            https?://                             # HTTP(S)
            (?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}  # 도메인
            (?::\d{1,5})?                        # 포트
            (?:[/?\#][^\s<>"'()[\]]*)?         # 경로, 쿼리, 프래그먼트
        """, re.VERBOSE | re.IGNORECASE),

        # 주민등록번호: YYMMDD-XXXXXXX / YYMMDD XXXXXXX / YYMMDDXXXXXXX
        # 두 번째 블록 첫 자리 1~8, 7자리. 출생일과 외형 검증은 normalizer 수행.
        "rrn": re.compile(r"""
            (?<!\d)\d{6}[- ]?[1-8]\d{6}(?!\d)
        """, re.VERBOSE),

        # 카드번호: 16자리 연속 / 4-4-4-4 하이픈 / 4 4 4 4 공백.
        # 단순한 숫자 나열과 구별하기 위해 정규화에서 Luhn 검사합 확인.
        "card": re.compile(r"""
            (?<!\d)
            (?:\d{16}|\d{4}(?P<card_sep>[ -])\d{4}(?P=card_sep)\d{4}(?P=card_sep)\d{4})
            (?!\d)
        """, re.VERBOSE),

        # 한국 주소의 제한된 범위: 시/도 + 1~3개 행정구역 + 도로명/동·리 + 건물번호
        # 예: 서울특별시 강남구 테헤란로 123
        # 예: 경기도 성남시 분당구 판교역로 235
        # 예: 부산광역시 해운대구 우동 123-45
        "address": re.compile(r"""
            (?<!\w)
            (?:서울특별시|서울시|부산광역시|부산시|대구광역시|인천광역시|
               광주광역시|대전광역시|울산광역시|세종특별자치시|경기도|
               강원특별자치도|충청북도|충청남도|전북특별자치도|
               전라남도|경상북도|경상남도|제주특별자치도)
            \s+
            (?:[가-힣0-9]+(?:시|군|구)\s+){1,3}
            (?:[가-힣0-9]+(?:로|길)|[가-힣0-9]+(?:동|읍|면|리))
            \s+\d+(?:-\d+)?
            (?!\d)
        """, re.VERBOSE),

        # 공급자 접두사에 근거한 일부 API Key 후보만 추출.
        # 임의의 난수/비밀번호를 API Key로 오인하지 않기 위한 범위 제한.
        # 예: sk-..., sk-proj-..., ghp_..., github_pat_..., AKIA...
        "api_key": API_KEY_PATTERN,
    }

    # 마스킹 정책: 기존 전화번호 예시 외 이메일도 식별정보일 수 있어 함께 처리.
    SENSITIVE_TYPES = {"email", "phone", "rrn", "card", "address", "api_key", "url"}

    def __init__(
        self,
        normalizer: InformationNormalizer | None = None,
        masker: PrivacyMasker | None = None,
    ) -> None:
        self.normalizer = normalizer if normalizer is not None else InformationNormalizer()
        self.masker = masker if masker is not None else PrivacyMasker()

        self._normalizers = {
            kind: getattr(self.normalizer, f"normalize_{kind}")
            for kind in self.PATTERNS
        }
        self._maskers = {
            kind: getattr(self.masker, f"mask_{kind}")
            for kind in self.SENSITIVE_TYPES
        }

    def extract(self, text: str) -> dict[str, list[str]]:
        """정규화 및 마스킹된 결과만 반환하며 원본 민감정보는 반환하지 않음."""
        result = {kind: [] for kind in self.PATTERNS}
        for kind, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                try:
                    normalized = self._normalizers[kind](match.group())
                    if kind in self._maskers:
                        normalized = self._maskers[kind](normalized)
                except (ValueError, OverflowError):
                    # 형식은 닮았지만 해석할 수 없는 후보는 제외
                    continue
                result[kind].append(normalized)
        return result
