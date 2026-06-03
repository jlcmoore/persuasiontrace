"""
src/tests/api/test_read_database.py

Tests for database-export model alias normalization.
"""

import unittest

from api.read_database import (
    CANONICAL_GPT5_MODEL,
    _canonicalize_condition_for_export,
    _canonicalize_round_for_export,
)
from experiment.condition import Condition, Roles
from experiment.round import Round


class TestReadDatabaseCanonicalization(unittest.TestCase):
    """Tests for GPT-5 alias canonicalization during export."""

    def test_canonicalize_exact_gpt5_alias_in_condition(self):
        """Map bare gpt-5 llm_persuader to the canonical pinned identifier."""
        condition = Condition(
            roles=Roles(llm_persuader="gpt-5", human_target=True),
            factual_domain=False,
        )
        canonical = _canonicalize_condition_for_export(condition)
        self.assertEqual(canonical.roles.llm_persuader, CANONICAL_GPT5_MODEL)

    def test_canonicalize_condition_llm_persuader_only(self):
        """Canonicalize only the llm_persuader field in condition roles."""
        condition = Condition(
            roles=Roles(
                llm_persuader="gpt-5",
                llm_target="openai/gpt-5",
            ),
            factual_domain=False,
        )
        canonical = _canonicalize_condition_for_export(condition)
        self.assertEqual(canonical.roles.llm_persuader, CANONICAL_GPT5_MODEL)
        self.assertEqual(canonical.roles.llm_target, "openai/gpt-5")

    def test_keep_non_exact_persuader_aliases_unchanged(self):
        """Do not rewrite non-exact gpt-5 llm_persuader values."""
        condition = Condition(
            roles=Roles(llm_persuader="gpt-5-2026-01-01", human_target=True),
            factual_domain=False,
        )
        canonical = _canonicalize_condition_for_export(condition)
        self.assertEqual(canonical.roles.llm_persuader, "gpt-5-2026-01-01")

    def test_canonicalize_round_condition(self):
        """Canonicalize embedded round condition for export payloads."""
        round_obj = Round(
            proposition="Sample proposition",
            condition=Condition(
                roles=Roles(llm_persuader="gpt-5", human_target=True),
                factual_domain=False,
            ),
            target_initial_belief=0.2,
            target_final_belief=0.8,
            persuader_supports_proposition=True,
            messages=[],
        )
        canonical_round = _canonicalize_round_for_export(round_obj)
        self.assertEqual(
            canonical_round.condition.roles.llm_persuader,
            CANONICAL_GPT5_MODEL,
        )


if __name__ == "__main__":
    unittest.main()
