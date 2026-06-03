"""
src/tests/experiment/test_conditions.py

Author: Jared Moore
Date: July, 2025

Tests for condition objects.
"""

import unittest

from experiment.condition import (
    MAX_DIR_COMPONENT_CHARS,
    PAIRED_HUMAN_ROLE,
    Condition,
    LLMPersuasionStyle,
    PropositionSource,
    Roles,
)


class TestRoles(unittest.TestCase):

    def test_invalid_combinations(self):
        # No persuader at all
        with self.assertRaises(ValueError):
            Roles(human_target=False, llm_target="A")

        # No target at all
        with self.assertRaises(ValueError):
            Roles(llm_persuader="A", human_target=False)

        # Both human and llm persuaders
        with self.assertRaises(ValueError):
            Roles(human_persuader=True, llm_persuader="A", human_target=False)

        # Both human and llm targets
        with self.assertRaises(ValueError):
            Roles(human_persuader=True, human_target=True, llm_target="A")

    def test_is_paired_human(self):
        self.assertTrue(PAIRED_HUMAN_ROLE.is_paired_human())
        # paired-human role has two humans, so it's not "rational"
        self.assertEqual(PAIRED_HUMAN_ROLE.target_type(), "Human")
        self.assertEqual(PAIRED_HUMAN_ROLE.persuader_type(), "Human")

    def test_as_non_id_role(self):
        # Start with numeric ids
        orig = Roles(
            human_persuader=7, human_target=11, llm_persuader=None, llm_target=None
        )

        # default (no_target_id=None) strips both ids to booleans
        no_ids = orig.as_non_id_role()
        self.assertIs(no_ids.human_persuader, True)
        self.assertIs(no_ids.human_target, True)

        # no_target_id=True: keep target id, drop persuader id
        keep_target = orig.as_non_id_role(no_target_id=True)
        self.assertIs(keep_target.human_persuader, True)
        self.assertEqual(keep_target.human_target, 11)

        # no_target_id=False: keep persuader id, drop target id
        keep_persuader = orig.as_non_id_role(no_target_id=False)
        self.assertEqual(keep_persuader.human_persuader, 7)
        self.assertIs(keep_persuader.human_target, True)


class TestCondition(unittest.TestCase):
    def setUp(self):
        self.roles = Roles(human_persuader=True, human_target=True)

    def test_str_matches_roles(self):
        cond = Condition(roles=PAIRED_HUMAN_ROLE, factual_domain=False)
        # __str__ should just delegate to Roles.__str__
        self.assertEqual(str(cond), f"{str(PAIRED_HUMAN_ROLE)} [text]")

    def test_to_dir_from_dir_roundtrip(self):
        # Build a condition with non-default factual_domain
        roles = Roles(
            human_persuader=3, human_target=False, llm_target="other-model/v3"
        )
        cond = Condition(
            roles=roles,
            factual_domain=False,
            llm_target_effect_scale=1.5,
        )

        dir_name = cond.to_dir()
        # round-trip decode/encode should be identical
        cond2 = Condition.from_dir(dir_name)
        self.assertEqual(cond2.roles, cond.roles)
        self.assertEqual(cond2.factual_domain, cond.factual_domain)
        self.assertEqual(cond2.to_dir(), dir_name)
        # Float scale should round-trip via string encoding.
        self.assertAlmostEqual(cond2.llm_target_effect_scale, 1.5)

    def test_to_dir_raises_when_encoding_too_long(self):
        """Very long condition encodings fail fast with a clear error."""
        cond = Condition(
            roles=Roles(
                llm_persuader="openai/" + ("gpt-5.4-nano" * 10),
                simulated_target="openai/" + ("gpt-4.1-nano" * 10),
                simulated_target_persona="logical",
            ),
            factual_domain=False,
            continuous_measure="serial-questions",
            proposition_source=PropositionSource.DEBATEGPT,
            turn_limit=6,
            no_early_end=True,
            simulated_target_no_rhetoric=True,
        )

        with self.assertRaises(ValueError):
            cond.to_dir()

    def test_to_dir_uses_semantic_aliases_for_sim_target_fields(self):
        """Sim-target long keys are shortened to readable aliases."""
        cond = Condition(
            roles=Roles(
                llm_persuader="p",
                simulated_target="s",
                simulated_target_persona="logical",
            ),
            factual_domain=False,
            continuous_measure="serial-questions",
            proposition_source=PropositionSource.DEBATEGPT,
            turn_limit=6,
            no_early_end=True,
            simulated_target_no_rhetoric=True,
            simulated_target_effect_scale=1.5,
            simulated_target_verbalize_beliefs=True,
        )

        dir_name = cond.to_dir()
        self.assertLessEqual(len(dir_name), MAX_DIR_COMPONENT_CHARS)
        self.assertIn("st_persona=logical", dir_name)
        self.assertIn("st_no_rhet=True", dir_name)
        self.assertIn("st_scale=1.5", dir_name)
        self.assertIn("st_v=True", dir_name)
        self.assertNotIn("simulated_target_persona", dir_name)
        self.assertNotIn("simulated_target_no_rhetoric", dir_name)
        self.assertNotIn("simulated_target_effect_scale", dir_name)
        self.assertNotIn("simulated_target_verbalize_beliefs", dir_name)

        restored = Condition.from_dir(dir_name)
        self.assertEqual(restored.roles.simulated_target_persona, "logical")
        self.assertTrue(restored.simulated_target_no_rhetoric)
        self.assertAlmostEqual(restored.simulated_target_effect_scale, 1.5)
        self.assertTrue(restored.simulated_target_verbalize_beliefs)

        old_alias_dir_name = dir_name.replace("st_v=True", "st_verbal=True")
        restored_old_alias = Condition.from_dir(old_alias_dir_name)
        self.assertTrue(restored_old_alias.simulated_target_verbalize_beliefs)

    def test_sim_target_effect_scale_requires_simulated_target(self):
        """Sim-target effect scale only applies to simulated-target roles."""
        with self.assertRaises(ValueError):
            Condition(
                roles=Roles(llm_persuader="gpt-4o", llm_target="gpt-4o"),
                factual_domain=False,
                simulated_target_effect_scale=1.2,
            )

    def test_sim_target_effect_scale_must_be_positive(self):
        """Sim-target effect scale must be positive when set."""
        with self.assertRaises(ValueError):
            Condition(
                roles=Roles(llm_persuader="gpt-4o", simulated_target="sim-target"),
                factual_domain=False,
                simulated_target_effect_scale=0.0,
            )

    def test_sim_target_verbal_beliefs_requires_simulated_target(self):
        """Sim-target verbal-belief toggle only applies to simulated targets."""
        with self.assertRaises(ValueError):
            Condition(
                roles=Roles(llm_persuader="gpt-4o", llm_target="gpt-4o"),
                factual_domain=False,
                simulated_target_verbalize_beliefs=True,
            )

    def test_condition_as_non_id_role(self):
        roles = Roles(human_persuader=42, human_target=99)
        cond = Condition(roles=roles, factual_domain=False)
        # strip persuader id only
        cond_stripped = cond.as_non_id_role(no_target_id=False)
        self.assertIs(cond_stripped.roles.human_target, True)
        self.assertEqual(cond_stripped.roles.human_persuader, 42)

        # strip target id only
        cond_stripped2 = cond.as_non_id_role(no_target_id=True)
        self.assertIs(cond_stripped2.roles.human_persuader, True)
        self.assertEqual(cond_stripped2.roles.human_target, 99)

    def test_factual_requires_proposition_is_correct(self):
        # Missing proposition_is_correct
        with self.assertRaises(ValueError):
            Condition(
                roles=self.roles,
                factual_domain=True,
                proposition_is_correct=None,
                persuader_supports_proposition=None,
            )
        # Providing it works
        c = Condition(
            roles=self.roles,
            factual_domain=True,
            proposition_is_correct=True,
            persuader_supports_proposition=False,
        )
        self.assertEqual(c.proposition_is_correct, True)

    def test_llm_persuasion_style_does_not_change_scale_unless_explicit(self):
        """Styles do not change effect scale unless explicitly set."""
        roles = Roles(llm_persuader="gpt-4o", llm_target="gpt-4o")

        cond_emotion = Condition(
            roles=roles,
            factual_domain=False,
            llm_persuasion_style=LLMPersuasionStyle.EMOTION,
        )
        self.assertIsNone(cond_emotion.llm_target_effect_scale)

        cond_facts = Condition(
            roles=roles,
            factual_domain=False,
            llm_persuasion_style=LLMPersuasionStyle.FACTS,
        )
        self.assertIsNone(cond_facts.llm_target_effect_scale)

        # Explicit scale should not be overridden by style.
        cond_override = Condition(
            roles=roles,
            factual_domain=False,
            llm_persuasion_style=LLMPersuasionStyle.EMOTION,
            llm_target_effect_scale=2.0,
        )
        self.assertAlmostEqual(cond_override.llm_target_effect_scale, 2.0)

    def test_proposition_source_allowed_for_human_target(self):
        cond_human = Condition(
            roles=self.roles,
            factual_domain=False,
            proposition_source=PropositionSource.PPT,
        )
        self.assertEqual(cond_human.proposition_source, PropositionSource.PPT)

        cond = Condition(
            roles=Roles(human_persuader=True, simulated_target="sim-target-v1"),
            factual_domain=False,
            proposition_source="debategpt",
        )
        self.assertEqual(cond.proposition_source, PropositionSource.DEBATEGPT)

    def test_proposition_source_allowed_with_node_belief_survey(self):
        """Node-belief survey conditions may pin proposition source."""
        cond = Condition(
            roles=Roles(llm_persuader="naive", human_target=True),
            factual_domain=False,
            proposition_source=PropositionSource.DEBATEGPT,
            enable_node_belief_survey=True,
        )
        self.assertEqual(cond.proposition_source, PropositionSource.DEBATEGPT)

    def test_llm_structure_allows_participant_proposition(self):
        """Structure-conditioned LLM targets may carry participant metadata."""
        cond = Condition(
            roles=Roles(human_persuader=True, llm_target="openai/gpt-4.1-nano"),
            factual_domain=False,
            participant_proposition=True,
            proposition_source=PropositionSource.PPT,
            llm_target_use_bayes_structure=True,
        )
        self.assertTrue(cond.participant_proposition)
        self.assertEqual(cond.proposition_source, PropositionSource.PPT)

    def test_llm_target_allows_participant_proposition_without_structure(self):
        """Vanilla LLM targets may use participant propositions."""
        cond = Condition(
            roles=Roles(human_persuader=True, llm_target="openai/gpt-4.1-nano"),
            factual_domain=False,
            participant_proposition=True,
            llm_target_use_bayes_structure=False,
        )
        self.assertTrue(cond.participant_proposition)
        self.assertIsNone(cond.proposition_source)

    def test_node_belief_survey_alias_roundtrip(self):
        """Node-belief survey flag round-trips through directory encoding."""
        cond = Condition(
            roles=Roles(human_persuader=True, human_target=True),
            factual_domain=False,
            enable_node_belief_survey=True,
        )
        dir_name = cond.to_dir()
        self.assertIn("bn_survey=True", dir_name)

        restored = Condition.from_dir(dir_name)
        self.assertTrue(restored.enable_node_belief_survey)


if __name__ == "__main__":
    unittest.main()
