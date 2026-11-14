"""미션 4-2 | Okt + KNU 사전 기반 한국어 감성 점수 계산.

설계 범위
- 원본 KNU와 확장 KNU를 모두 사용 가능하게 한다.
- Okt(norm=True, stem=True) 기준으로 사전 표현과 리뷰를 동일하게 토큰화한다.
- 형태소 + 품사 시퀀스를 키로 사용한다. 겹칠 때는 최장 표현을 우선한다.
- 짧은 부정(안/못), 긴 부정(지 않다/지 못하다), 강조어를 제한된 범위에서 처리한다.
- 점수 0은 '중립이라고 확정'이 아니라 '증거 부족 또는 점수 상쇄'일 수 있다.

주의: 이 모듈은 교육용 기본선이다. 구문·의존 관계 분석이나 반어법을 다루지 않는다.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


# 직접 조정할 정책 값. 부정어·강조어 자체는 감성어로 점수화하지 않는다.
PREFIX_NEGATORS = {"안", "못"}
SUFFIX_NEGATORS = {"않다", "못하다"}
INTENSIFIERS = {"매우": 1.5, "정말": 1.5, "아주": 1.3,
                "너무": 1.5, "굉장히": 1.5, "엄청": 1.5,
                "진짜": 1.3, "조금": 0.5, "약간": 0.7}
RULE_ONLY_FORMS = set(PREFIX_NEGATORS) | set(SUFFIX_NEGATORS) | set(INTENSIFIERS)
BARRIER_TAGS = {"Punctuation"}
# '하지만', '그러나' 등은 간단한 절 경계로 취급한다. 만능 구문 분석이 아님.
BARRIER_FORMS = {"하지만", "그러나", "그런데", "그렇지만", "반면"}
LEXICON_INDEX_SCHEMA = 2
MAX_LEXICON_TOKENS = 8


@dataclass(frozen=True)
class Token:
    form: str
    pos: str

    @property
    def key(self) -> tuple[str, str]:
        return self.form, self.pos


@dataclass(frozen=True)
class Entry:
    keys: tuple[tuple[str, str], ...]
    word: str
    score: int
    source: str  # knu / custom

    def to_json(self) -> dict[str, Any]:
        return {"keys": [list(key) for key in self.keys],
                "word": self.word, "score": self.score, "source": self.source}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Entry":
        return cls(tuple(tuple(k) for k in data["keys"]),
                   str(data["word"]), int(data["score"]), str(data["source"]))


class OktTokenizer:
    """이미 설치된 KoNLPy의 Okt를 사용한다. 별도 다운로드는 하지 않는다."""

    def __init__(self) -> None:
        from konlpy.tag import Okt
        self.okt = Okt()

    def tokenize(self, text: str) -> list[Token]:
        return [Token(str(form), str(pos))
                for form, pos in self.okt.pos(str(text), norm=True, stem=True)]


def load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError(f"사전 최상위 자료형은 list여야 합니다: {path}")
    for idx, record in enumerate(records):
        if not isinstance(record, dict) or not {"word", "polarity"}.issubset(record):
            raise ValueError(f"사전 구조 오류: {path}, index={idx}")
        try:
            score = int(record["polarity"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"polarity 정수 변환 실패: index={idx}") from exc
        if score not in (-2, -1, 0, 1, 2):
            raise ValueError(f"KNU 점수 범위(-2~2) 위반: index={idx}, score={score}")
    return records


def check_extension(original: list[dict], extended: list[dict]) -> list[dict]:
    """원본이 앞부분에 원형 그대로 존재하는지 검사하고, 추가분만 반환."""
    if len(extended) < len(original) or extended[:len(original)] != original:
        raise ValueError("확장 사전의 앞부분이 KNU 원본과 일치하지 않습니다. 병합 구조를 확인하세요.")
    additions = extended[len(original):]
    if len(additions) < 30:
        raise ValueError(f"도메인 추가 항목이 {len(additions)}개입니다. 최소 30개 필요")
    if len({str(r['word']) for r in additions}) != len(additions):
        raise ValueError("확장 사전 신규 항목의 문자열이 중복됩니다.")
    return additions


def read_custom_pos(path: Path, additions: list[dict]) -> dict[str, str]:
    """4번째 노트북에서 생성한 custom_entries CSV의 품사를 이용한다.

    없어도 분석은 가능하지만, 같은 문자열이 다른 품사로 등장하는 문제를 줄이려면 제공해야 한다.
    """
    if not path.is_file():
        return {}
    import csv
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            word = (row.get("token") or "").strip()
            pos = (row.get("pos") or "").strip()
            if word and pos:
                if word in result and result[word] != pos:
                    raise ValueError(f"동일 감성어에 품사 충돌: {word}")
                result[word] = pos
    additional_words = {str(r["word"]) for r in additions}
    unexpected = set(result) - additional_words
    if unexpected:
        raise ValueError(f"custom_entries CSV가 확장 사전과 맞지 않음: {sorted(unexpected)[:5]}")
    return result


def _make_entry(record: dict, tokenizer: Any, source: str,
                pos_hint: str | None = None) -> Entry | None:
    score = int(record["polarity"])
    if score == 0:
        return None
    word = str(record["word"]).strip()
    if not word:
        return None
    if pos_hint and not any(char.isspace() for char in word):
        # 선정 CSV의 단어는 이미 Okt stem=True로 나온 토큰이다.
        keys = ((word, pos_hint),)
    else:
        keys = tuple(t.key for t in tokenizer.tokenize(word))
    if not keys or len(keys) > MAX_LEXICON_TOKENS:
        return None
    if all(pos == "Punctuation" for _, pos in keys):
        return None
    # 부정어/강조어만으로 이루어진 한 토큰은 독립 감성 점수가 아니다.
    if len(keys) == 1 and keys[0][0] in RULE_ONLY_FORMS:
        return None
    return Entry(keys=keys, word=word, score=score, source=source)


class LexiconIndex:
    def __init__(self, entries: Sequence[Entry], collisions: int = 0):
        self.entries = list(entries)
        self.collisions = collisions
        self.by_first: dict[tuple[str, str], list[Entry]] = defaultdict(list)
        for entry in self.entries:
            self.by_first[entry.keys[0]].append(entry)
        # 겹치는 표현 중 가장 긴 것부터 탐색.
        for group in self.by_first.values():
            group.sort(key=lambda e: -len(e.keys))

    @classmethod
    def build(cls, original: list[dict], additions: list[dict],
              tokenizer: Any, custom_pos: dict[str, str] | None = None) -> "LexiconIndex":
        chosen: dict[tuple[tuple[str, str], ...], Entry] = {}
        collisions = 0
        for record in original:
            entry = _make_entry(record, tokenizer, "knu")
            if entry is None:
                continue
            if entry.keys in chosen:
                collisions += 1
                continue  # 원본 내 중복 정규화 키는 먼저 나온 것을 유지한다.
            chosen[entry.keys] = entry
        for record in additions:
            word = str(record["word"])
            entry = _make_entry(record, tokenizer, "custom", (custom_pos or {}).get(word))
            if entry is None:
                continue
            if entry.keys in chosen:
                collisions += 1
            chosen[entry.keys] = entry  # 확장판에서는 승인된 신규 항목 우선.
        return cls(list(chosen.values()), collisions)

    def with_additions(self, additions: list[dict], tokenizer: Any,
                       custom_pos: dict[str, str] | None = None) -> "LexiconIndex":
        """이미 생성한 원본 색인을 재사용해 신규 항목만 토큰화한다."""
        chosen = {entry.keys: entry for entry in self.entries}
        collisions = self.collisions
        for record in additions:
            word = str(record["word"])
            entry = _make_entry(record, tokenizer, "custom", (custom_pos or {}).get(word))
            if entry is None:
                continue
            if entry.keys in chosen:
                collisions += 1
            chosen[entry.keys] = entry  # 승인한 신규 항목 우선
        return LexiconIndex(list(chosen.values()), collisions)

    def match(self, tokens: Sequence[Token]) -> list[dict[str, Any]]:
        """토큰을 왼쪽부터 탐색해 최장 패턴 우선 매칭 (중복 점수화 방지)."""
        result = []
        token_keys = [t.key for t in tokens]
        i = 0
        while i < len(tokens):
            found = None
            for candidate in self.by_first.get(token_keys[i], []):
                k = len(candidate.keys)
                if tuple(token_keys[i:i+k]) == candidate.keys:
                    found = candidate
                    break
            if found is None:
                i += 1
                continue
            end = i + len(found.keys)
            result.append({"start": i, "end": end, "entry": found})
            i = end
        return result

    def to_json(self) -> dict[str, Any]:
        return {"schema": LEXICON_INDEX_SCHEMA, "collisions": self.collisions,
                "entries": [entry.to_json() for entry in self.entries]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LexiconIndex":
        if data.get("schema") != LEXICON_INDEX_SCHEMA:
            raise ValueError("색인 캐시 버전이 다릅니다. 재생성하세요.")
        return cls([Entry.from_json(e) for e in data["entries"]],
                   int(data.get("collisions", 0)))


def _is_barrier(token: Token) -> bool:
    return token.pos in BARRIER_TAGS or token.form in BARRIER_FORMS


def _around(tokens: Sequence[Token], start: int, end: int, window: int,
            reverse: bool) -> list[Token]:
    """절 경계를 넘지 않는 인접 형태소 목록. 시작/끝은 [start,end) 범위."""
    indexes = (range(start-1, max(-1, start-window-1), -1) if reverse
               else range(end, min(len(tokens), end+window)))
    found = []
    for i in indexes:
        token = tokens[i]
        if _is_barrier(token):
            break
        found.append(token)
    return found


class SentimentAnalyzer:
    def __init__(self, lexicon: LexiconIndex, *, use_rules: bool = True):
        self.lexicon = lexicon
        self.use_rules = use_rules

    def score_tokens(self, tokens: Sequence[Token]) -> dict[str, Any]:
        matches = self.lexicon.match(tokens)
        spans = [(m["start"], m["end"]) for m in matches]
        hits: list[dict[str, Any]] = []
        for idx, match in enumerate(matches):
            start, end, entry = match["start"], match["end"], match["entry"]
            # 다른 감성 표현을 관통하지 않도록 탐색 범위를 제한한다.
            left_limit = spans[idx-1][1] if idx > 0 else 0
            right_limit = spans[idx+1][0] if idx+1 < len(spans) else len(tokens)
            before = _around(tokens, start, end, min(3, start-left_limit), reverse=True)
            after = _around(tokens, start, end, min(3, right_limit-end), reverse=False)
            before_forms = [t.form for t in before]
            after_forms = [t.form for t in after]
            within = [t.form for t in tokens[start:end]]
            embedded_negation = any(x in PREFIX_NEGATORS or x in SUFFIX_NEGATORS for x in within)
            prefix_count = sum(x in PREFIX_NEGATORS for x in before_forms[:2])
            # '좋지 않다' 패턴의 '않다' 탐지. (Okt의 분해에 따라 '지'가 독립 토큰일 수도 있음)
            suffix_count = int(any(x in SUFFIX_NEGATORS for x in after_forms[:3]))
            negations = 0 if embedded_negation else prefix_count + suffix_count
            multiplier = 1.0
            modifier = None
            if self.use_rules:
                for form in before_forms:
                    if form in INTENSIFIERS:
                        modifier = form
                        multiplier = INTENSIFIERS[form]
                        break
                polarity = (-1 if negations % 2 else 1)
            else:
                negations, multiplier, modifier, polarity = 0, 1.0, None, 1
            adjusted = entry.score * polarity * multiplier
            hits.append({
                "term": entry.word, "source": entry.source,
                "matched_tokens": " / ".join(t.form for t in tokens[start:end]),
                "start": start, "end": end,
                "base_score": entry.score,
                "negation_count": negations,
                "modifier": modifier,
                "multiplier": multiplier,
                "adjusted_score": round(float(adjusted), 4),
            })
        return {
            "score": round(sum(h["adjusted_score"] for h in hits), 4),
            "hit_count": len(hits),
            "custom_hit_count": sum(h["source"] == "custom" for h in hits),
            "hits": hits,
        }

    def analyze(self, text: str, tokenizer: Any) -> dict[str, Any]:
        return self.score_tokens(tokenizer.tokenize(text))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_or_load_indexes(
    original_path: Path,
    extended_path: Path,
    custom_csv_path: Path,
    tokenizer: Any,
    cache_dir: Path,
) -> tuple[LexiconIndex, LexiconIndex, dict[str, Any]]:
    """원본/확장판을 검증하고, Okt 처리 비용이 큰 색인을 JSON으로 캐싱한다."""
    orig = load_records(original_path)
    extended = load_records(extended_path)
    additions = check_extension(orig, extended)
    pos_hints = read_custom_pos(custom_csv_path, additions)
    if len(pos_hints) != len(additions):
        # 오래된/별도 생성 사전일 수 있으므로 실패하지 않고 통계로 알린다.
        pass
    import konlpy
    sig = {
        "schema": LEXICON_INDEX_SCHEMA,
        "original_sha256": _sha256(original_path),
        "extended_sha256": _sha256(extended_path),
        "custom_csv_sha256": _sha256(custom_csv_path) if custom_csv_path.exists() else "missing",
        "konlpy_version": konlpy.__version__,
        "normalization": "Okt norm=True stem=True POS-aware; longest-first; custom override",
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"okt_lexicon_index_{digest}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("signature") != sig:
            raise ValueError("색인 캐시의 서명이 일치하지 않습니다.")
        base_index = LexiconIndex.from_json(payload["base"])
        extended_index = LexiconIndex.from_json(payload["extended"])
        reused = True
    else:
        # 1회 구축 시 KNU 원본 14,843개를 Okt로 분석하므로 시간이 걸릴 수 있음.
        base_index = LexiconIndex.build(orig, [], tokenizer)
        extended_index = base_index.with_additions(additions, tokenizer, pos_hints)
        payload = {"signature": sig, "base": base_index.to_json(),
                   "extended": extended_index.to_json()}
        # 캐시 파일은 새 digest 이름으로 작성. 기존 사전/캐시는 덮어쓰지 않는다.
        with cache_path.open("x", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        reused = False
    stats = {"original_records": len(orig), "custom_records": len(additions),
             "custom_pos_hints": len(pos_hints),
             "base_patterns": len(base_index.entries),
             "extended_patterns": len(extended_index.entries),
             "base_normalization_collisions": base_index.collisions,
             "extended_normalization_collisions": extended_index.collisions,
             "cache_path": str(cache_path), "cache_reused": reused,
             "signature": sig}
    return base_index, extended_index, stats
