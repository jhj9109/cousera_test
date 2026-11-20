"""작은 고정 사전으로 최장 일치, 부정 반전, 강도 및 절 경계를 검증한다."""
import unittest

from src.korean_sentiment import Entry, LexiconIndex, SentimentAnalyzer, Token


class TestSentimentRules(unittest.TestCase):
    def setUp(self):
        self.good = Token("좋다", "Adjective")
        self.negation = Token("않다", "Verb")
        self.masterpiece = Token("명작", "Noun")
        self.analyzer = SentimentAnalyzer(LexiconIndex([
            Entry((self.good.key,), "좋다", 2, "knu"),
            Entry((self.good.key, self.negation.key), "좋지 않다", -2, "knu"),
            Entry((self.masterpiece.key,), "명작", 2, "custom"),
        ]))

    def test_longest_match_keeps_embedded_negation(self):
        result = self.analyzer.score_tokens([self.good, self.negation])
        self.assertEqual(result["hit_count"], 1)
        self.assertEqual(result["hits"][0]["negation_count"], 0)
        self.assertEqual(result["score"], -2)

    def test_external_negation_reverses_score(self):
        self.assertEqual(self.analyzer.score_tokens([Token("안", "Adverb"), self.good])["score"], -2)

    def test_modifier_stops_at_punctuation(self):
        modifier = Token("매우", "Adverb")
        self.assertEqual(self.analyzer.score_tokens([modifier, self.masterpiece])["score"], 3)
        tokens = [modifier, Token(",", "Punctuation"), self.masterpiece]
        self.assertEqual(self.analyzer.score_tokens(tokens)["score"], 2)
