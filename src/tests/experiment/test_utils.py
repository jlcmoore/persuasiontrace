"""
src/tests/experiment/test_utils.py

Author: Jared Moore
Date: July, 2025

Tests for utils.
"""

import unittest

from experiment.utils import (
    comma_with_and,
    dict_to_string,
    escape_string,
    int_to_words,
    limit_to_n_characters,
    make_text_transcript,
    model_name_short,
    number_to_words_ordinal,
    prefix_with_conjunction,
    replace_json_chars,
    string_to_dict,
    token_time_totals_verbose,
    unescape_string,
)


class TestUtils(unittest.TestCase):
    def test_limit_to_n_characters(self):
        s = "hello world"
        self.assertEqual(limit_to_n_characters(s, 5), "hello")
        self.assertEqual(limit_to_n_characters(s, len(s)), s)
        self.assertEqual(limit_to_n_characters(s, len(s) + 10), s)
        self.assertEqual(limit_to_n_characters("", 3), "")

    def test_model_name_short(self):
        # Known mapping
        self.assertEqual(
            model_name_short("meta-llama/Llama-2-70b-chat-hf"), "llama2-70b"
        )
        # Unknown model should return unchanged
        self.assertEqual(model_name_short("some-unknown-model"), "some-unknown-model")

    def test_prefix_with_conjunction(self):
        self.assertEqual(prefix_with_conjunction("and", "or", []), "")
        self.assertEqual(prefix_with_conjunction("and", "or", ["one"]), "one")
        # prefix only acts as the separator between all but the last+conjunction
        self.assertEqual(
            prefix_with_conjunction(">", "and", ["a", "b", "c"]), "a> b and c"
        )
        self.assertEqual(
            prefix_with_conjunction(">>", "nor", ["x y", "z"]), "x y nor z"
        )

    def test_comma_with_and(self):
        self.assertEqual(comma_with_and([]), "")
        self.assertEqual(comma_with_and(["foo"]), "foo")
        self.assertEqual(comma_with_and(["a", "b", "c"]), "a, b and c")

    def test_int_to_words_and_ordinal(self):
        self.assertEqual(int_to_words(1), "one")
        self.assertIn("twenty", int_to_words(21))
        # test ordinal
        self.assertEqual(number_to_words_ordinal(1), "first")
        self.assertIn("second", number_to_words_ordinal(2))
        self.assertIn("twenty", number_to_words_ordinal(21).lower())

    def test_replace_json_chars(self):
        s = "“foo”： ‘bar’，```jsonbaz```"
        replaced = replace_json_chars(s)
        self.assertNotIn("“", replaced)
        self.assertNotIn("”", replaced)
        self.assertNotIn("：", replaced)
        self.assertNotIn("，", replaced)
        self.assertNotIn("```json", replaced)
        self.assertNotIn("```", replaced)
        self.assertIn('"foo"', replaced)
        self.assertIn('"bar"', replaced)
        self.assertIn("baz", replaced)

    def test_escape_and_unescape_string(self):
        # normal case
        self.assertEqual(escape_string("a/b"), "a__b")
        self.assertEqual(unescape_string("a__b"), "a/b")
        # multiple slashes
        self.assertEqual(escape_string("x/y/z"), "x__y__z")
        self.assertEqual(unescape_string("x__y__z"), "x/y/z")
        # disallowed characters
        with self.assertRaises(TypeError):
            escape_string("bad&value")
        with self.assertRaises(TypeError):
            escape_string("bad\\value")

    def test_dict_to_string_and_string_to_dict_roundtrip(self):
        d = {
            "num": 42,
            "flag": True,
            "name": "hello/world",
            "nothing": None,
            "text": "plain",
        }
        s = dict_to_string(d)
        # Should sort keys alphabetically and join with &
        # Keys sorted: flag, name, neg... but numeric check:
        pieces = s.split("&")
        keys_in_order = [p.split("=")[0] for p in pieces]
        self.assertEqual(keys_in_order, sorted(d.keys()))
        # Roundtrip
        d2 = string_to_dict(s)
        self.assertEqual(d2["num"], 42)
        self.assertIs(d2["flag"], True)
        self.assertEqual(d2["name"], "hello/world")
        self.assertIsNone(d2["nothing"])
        self.assertEqual(d2["text"], "plain")

    def test_dict_to_string_unsupported_type(self):
        with self.assertRaises(TypeError):
            dict_to_string({"x": 3.14})  # float is unsupported
        with self.assertRaises(TypeError):
            dict_to_string({"y": object()})

    def test_string_to_dict_edge_cases(self):
        # Empty string should give { "": ??? } but our implementation will
        # try to split and likely error; we test a minimal valid example
        s = "a=True&b=False&c=None&d=123&e=some_str"
        d = string_to_dict(s)
        self.assertIs(d["a"], True)
        self.assertIs(d["b"], False)
        self.assertIsNone(d["c"])
        self.assertEqual(d["d"], 123)
        self.assertEqual(d["e"], "some_str")


EX_TRANSCRIPT = {
    "duration": 13.199999809265137,
    "language": "english",
    "task": "transcribe",
    "text": "I totally believe Pet Sounds is the best album of all time. The "
    "innovation in production and harmonies is unmatched. It really "
    "redefined what popular music could be. What do you think?",
    "usage": {"seconds": 14.0, "type": "duration"},
    "words": [
        {"end": 0.5, "start": 0.0, "word": "I"},
        {"end": 0.9800000190734863, "start": 0.5, "word": "totally"},
        {"end": 1.399999976158142, "start": 0.9800000190734863, "word": "believe"},
        {"end": 1.840000033378601, "start": 1.399999976158142, "word": "Pet"},
        {"end": 2.119999885559082, "start": 1.840000033378601, "word": "Sounds"},
        {"end": 2.559999942779541, "start": 2.119999885559082, "word": "is"},
        {"end": 2.8399999141693115, "start": 2.559999942779541, "word": "the"},
        {"end": 3.140000104904175, "start": 2.8399999141693115, "word": "best"},
        {"end": 3.4600000381469727, "start": 3.140000104904175, "word": "album"},
        {"end": 3.859999895095825, "start": 3.4600000381469727, "word": "of"},
        {"end": 4.199999809265137, "start": 3.859999895095825, "word": "all"},
        {"end": 4.559999942779541, "start": 4.199999809265137, "word": "time"},
        {"end": 5.260000228881836, "start": 5.199999809265137, "word": "The"},
        {
            "end": 5.739999771118164,
            "start": 5.260000228881836,
            "word": "innovation",
        },
        {"end": 6.139999866485596, "start": 5.739999771118164, "word": "in"},
        {
            "end": 6.619999885559082,
            "start": 6.139999866485596,
            "word": "production",
        },
        {"end": 7.099999904632568, "start": 6.619999885559082, "word": "and"},
        {"end": 7.480000019073486, "start": 7.099999904632568, "word": "harmonies"},
        {"end": 8.0, "start": 7.480000019073486, "word": "is"},
        {"end": 8.460000038146973, "start": 8.0, "word": "unmatched"},
        {"end": 9.199999809265137, "start": 9.100000381469727, "word": "It"},
        {"end": 9.65999984741211, "start": 9.199999809265137, "word": "really"},
        {"end": 10.239999771118164, "start": 9.65999984741211, "word": "redefined"},
        {"end": 10.5600004196167, "start": 10.239999771118164, "word": "what"},
        {"end": 10.960000038146973, "start": 10.5600004196167, "word": "popular"},
        {"end": 11.300000190734863, "start": 10.960000038146973, "word": "music"},
        {"end": 11.520000457763672, "start": 11.300000190734863, "word": "could"},
        {"end": 11.779999732971191, "start": 11.520000457763672, "word": "be"},
        {"end": 12.5, "start": 12.319999694824219, "word": "What"},
        {"end": 12.619999885559082, "start": 12.5, "word": "do"},
        {"end": 12.819999694824219, "start": 12.619999885559082, "word": "you"},
        {"end": 13.0, "start": 12.819999694824219, "word": "think"},
    ],
}


class TestTranscript(unittest.TestCase):

    def test_make_text_transcript_empty(self):
        tr = make_text_transcript("")
        self.assertIsInstance(tr, dict)
        self.assertIn("text", tr)
        self.assertIn("duration", tr)
        self.assertIn("words", tr)
        self.assertEqual(tr["text"], "")
        self.assertEqual(tr["duration"], 0.0)
        self.assertEqual(tr["words"], [])

    def test_make_text_transcript_simple_structure(self):
        tr = make_text_transcript("Hello world.")
        self.assertGreater(tr["duration"], 0.0)
        self.assertGreaterEqual(len(tr["words"]), 1)
        # At least one non-punctuation "word"
        words = tr["words"]
        self.assertTrue(any((not w.get("punct", False)) for w in words))
        # Duration should match last segment end (within rounding)
        self.assertAlmostEqual(tr["duration"], tr["words"][-1]["end"], places=3)

    def test_make_text_transcript_handles_newlines_and_ellipses(self):
        text = "One, two, three...\nNew line!"
        tr = make_text_transcript(text)
        self.assertEqual(tr["text"], text)
        # Likely split into multiple segments due to "..." and newline and "!"
        self.assertGreaterEqual(len(tr["words"]), 2)
        self.assertGreater(tr["duration"], 0.0)

    def test_make_text_transcript_monotonic_word_timings(self):
        text = "Timing check: this should not go backwards."
        tr = make_text_transcript(text)
        # Flatten all word entries across segments
        all_words = []
        for seg in tr["words"]:
            all_words.extend(seg.get("words", []))
        # Each word should have end >= start and non-negative
        self.assertTrue(all(w["end"] >= w["start"] >= 0.0 for w in all_words))
        # Segment times should be non-decreasing and within total duration
        prev_end = 0.0
        for seg in tr["words"]:
            self.assertGreaterEqual(seg["end"], seg["start"])
            self.assertGreaterEqual(seg["start"], prev_end)
            prev_end = seg["end"]
        # Duration should be at least the last segment end
        self.assertAlmostEqual(tr["duration"], tr["words"][-1]["end"], places=3)

    def test_make_text_transcript_long_word(self):
        # Ensure long/complex tokens don't cause errors
        text = "Antidisestablishmentarianism."
        tr = make_text_transcript(text)
        self.assertGreater(tr["duration"], 0.0)
        self.assertGreaterEqual(len(tr["words"]), 1)
        words = tr["words"]
        self.assertTrue(
            len(words) >= 1
        )  # should at least include the word and maybe punctuation

    def test_totals(self):
        totals = token_time_totals_verbose(EX_TRANSCRIPT)
        self.assertEqual(len(totals), 32)

        self.assertAlmostEqual(
            EX_TRANSCRIPT["duration"], sum(msg["duration"] for msg in totals), places=1
        )


if __name__ == "__main__":
    unittest.main()
