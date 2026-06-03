"""
src/tests/experiment/test_round.py

Author: Jared Moore
Date: July, 2025

Tests for Round.
"""

# tests/test_round.py

import datetime
import json
import os
import random
import unittest
from tempfile import TemporaryDirectory

import experiment.round as round_mod  # so we can patch RESULTS_DIR
from experiment.condition import Condition, Roles
from experiment.round import Round, output_conditions_and_rounds


class TestRoundBasic(unittest.TestCase):
    def setUp(self):
        # A paired human condition (factual_domain=False) so we can
        # create non-factual rounds without needing proposition_is_correct.
        self.roles = Roles(human_persuader=True, human_target=True)
        self.non_factual_cond = Condition(
            roles=self.roles, factual_domain=False, turn_limit=2
        )
        self.r = Round(
            condition=self.non_factual_cond,
            proposition="Test",
            persuader_supports_proposition=True,
            target_initial_belief=0.0,
            target_final_belief=1.0,
        )

    def test_validate_belief_range(self):
        # target_initial_belief out of [0,1]
        with self.assertRaises(ValueError):
            Round(
                condition=self.non_factual_cond,
                proposition="P",
                persuader_supports_proposition=True,
                target_initial_belief=1.5,
                target_final_belief=0.5,
            )
        # target_final_belief out of [0,1]
        with self.assertRaises(ValueError):
            Round(
                condition=self.non_factual_cond,
                proposition="P",
                persuader_supports_proposition=True,
                target_initial_belief=0.5,
                target_final_belief=-0.1,
            )

    def test_finished(self):
        r = Round(
            condition=self.non_factual_cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=None,
            target_final_belief=None,
        )
        self.assertFalse(r.finished())
        # once both beliefs are set
        r2 = Round(
            condition=self.non_factual_cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.2,
            target_final_belief=0.8,
        )
        self.assertTrue(r2.finished())

    def test_target_persuaded(self):
        # persuader supports proposition, final > 0.5 => persuaded
        r = Round(
            condition=Condition(
                roles=self.roles,
                factual_domain=True,
                proposition_is_correct=True,
            ),
            persuader_supports_proposition=True,
            proposition="P",
            target_initial_belief=0.3,
            target_final_belief=0.9,
        )
        self.assertTrue(r.target_persuaded())
        r2 = r.model_copy(update={"target_final_belief": 0.5})
        self.assertTrue(r2.target_persuaded())
        # persuader opposes
        r3 = r.model_copy(
            update={
                "persuader_supports_proposition": False,
                "target_initial_belief": 0.8,
                "target_final_belief": 0.2,
            }
        )
        self.assertTrue(r3.target_persuaded())
        # before finished
        r4 = Round(
            condition=self.non_factual_cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=None,
            target_final_belief=None,
        )
        with self.assertRaises(ValueError):
            r4.target_persuaded()

    def test_target_answered_correctly(self):
        factual = Condition(
            roles=self.roles,
            factual_domain=True,
            proposition_is_correct=True,
        )
        # correct proposition, final>0.5 => correct
        r = Round(
            condition=factual,
            proposition="P",
            proposition_is_correct=True,
            persuader_supports_proposition=True,
            target_initial_belief=0.1,
            target_final_belief=0.6,
        )
        self.assertTrue(r.target_answered_correctly())
        # incorrect proposition, final<0.5 => correct
        r2 = r.model_copy(
            update={"proposition_is_correct": False, "target_final_belief": 0.4}
        )
        self.assertTrue(r2.target_answered_correctly())
        # non-factual domain
        with self.assertRaises(ValueError):
            r_nf = Round(
                condition=self.non_factual_cond,
                proposition="P",
                persuader_supports_proposition=True,
                target_initial_belief=0.2,
                target_final_belief=0.8,
            )
            r_nf.target_answered_correctly()
        # before finished
        r3 = Round(
            condition=factual,
            proposition="P",
            proposition_is_correct=True,
            persuader_supports_proposition=True,
            target_initial_belief=None,
            target_final_belief=None,
        )
        with self.assertRaises(ValueError):
            r3.target_answered_correctly()

    def test_message_history_helpers(self):
        r = Round(
            condition=self.non_factual_cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.2,
            target_final_belief=0.8,
        )
        # no messages yet
        self.assertIsNone(r.last_message(is_target=True))
        self.assertIsNone(r.last_message(is_target=False))
        self.assertEqual(r.message_length(is_target=True), 0)
        self.assertEqual(r.message_length(is_target=False), 0)
        # add some messages
        r.messages.append({"role": "persuader", "content": "hey"})
        r.messages.append({"role": "target", "content": "ok"})
        r.messages.append({"role": "persuader", "content": "bye"})
        # last_message
        self.assertEqual(r.last_message(is_target=False), "bye")
        self.assertEqual(r.last_message(is_target=True), "ok")
        # message_length (integer division)
        self.assertEqual(
            r.message_length(is_target=False), (len("hey") + len("bye")) // 2
        )
        # actually we do (3+3)//2==3
        self.assertEqual(r.message_length(is_target=False), 3)
        self.assertEqual(r.message_length(is_target=True), len("ok") // 1)

    def test_turns_and_ordering(self):
        # no turn_limit => infinite
        r = Round(
            condition=self.non_factual_cond.model_copy(update={"turn_limit": None}),
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.2,
            target_final_belief=0.8,
        )
        self.assertTrue(r.turns_left(is_target=False))
        self.assertTrue(r.turns_left(is_target=True))
        self.assertFalse(r.neither_turns_left())
        self.assertFalse(r.target_plays_next())  # persuader starts
        # with turn_limit=1 => one turn each
        r2 = Round(
            condition=self.non_factual_cond.model_copy(update={"turn_limit": 1}),
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.2,
            target_final_belief=0.8,
            messages=[
                {"role": "persuader", "content": "x"},
                {"role": "target", "content": "y"},
            ],
        )
        self.assertEqual(r2.turns_left(is_target=False), 0)
        self.assertEqual(r2.turns_left(is_target=True), 0)
        self.assertTrue(r2.neither_turns_left())
        self.assertFalse(r2.target_plays_next())  # last was target, so persuader next

    def test_add_message_defaults(self):
        # Initially all four lists are empty
        self.assertEqual(len(self.r.messages), 0)
        self.assertEqual(len(self.r.transcripts), 0)
        self.assertEqual(len(self.r.chains_of_thought), 0)
        self.assertEqual(len(self.r.reasoning_traces), 0)

        # Add a persuader turn without optional args
        self.r.add_message(role="persuader", content="Hello!")
        # Add a target turn with all optional args
        custom_transcript = {"role": "target", "content": "Custom transcript"}
        self.r.add_message(
            role="target",
            content="Hi!",
            transcript=custom_transcript,
            cot="I think...",
            reasoning_trace="trace data",
        )

        # Now all four lists have length 2
        self.assertEqual(len(self.r.messages), 2)
        self.assertEqual(len(self.r.transcripts), 2)
        self.assertEqual(len(self.r.chains_of_thought), 2)
        self.assertEqual(len(self.r.reasoning_traces), 2)

        # Check the first entry (default transcript, no cot/trace)
        self.assertEqual(self.r.messages[0], {"role": "persuader", "content": "Hello!"})
        self.assertEqual(
            self.r.transcripts[0], {"role": "persuader", "content": "Hello!"}
        )
        self.assertEqual(
            self.r.chains_of_thought[0], {"role": "persuader", "content": None}
        )
        self.assertEqual(
            self.r.reasoning_traces[0], {"role": "persuader", "content": None}
        )

        # Check the second entry (custom transcript, cot, trace)
        self.assertEqual(self.r.messages[1], {"role": "target", "content": "Hi!"})
        self.assertIs(self.r.transcripts[1], custom_transcript)
        self.assertEqual(
            self.r.chains_of_thought[1], {"role": "target", "content": "I think..."}
        )
        self.assertEqual(
            self.r.reasoning_traces[1], {"role": "target", "content": "trace data"}
        )

    def test_llm_target_effect_scale_applied(self):
        """LLM target movement is scaled by llm_target_effect_scale."""
        roles = Roles(llm_persuader="gpt-4o", llm_target="gpt-4o")
        base_cond = Condition(roles=roles, factual_domain=False)

        # Helper to build a round with one persuader message and inspect final belief
        def run_with_scale(scale: float | None) -> float:
            cond = base_cond.model_copy(
                update={"llm_target_effect_scale": scale} if scale is not None else {}
            )
            rd = Round(
                condition=cond,
                proposition="P",
                persuader_supports_proposition=True,
                target_initial_belief=0.4,
                target_final_belief=None,
                messages=[{"role": "persuader", "content": "short but persuasive"}],
            )
            # No serial measures: rely on message-based heuristic
            return rd.final_belief_per_policy()

        # Use a fixed seed so updated_belief_after_persuader jitter is stable.
        random.seed(1234)
        base = run_with_scale(1.0)

        random.seed(1234)
        up = run_with_scale(1.5)

        random.seed(1234)
        down = run_with_scale(0.5)

        self.assertGreater(up - 0.4, base - 0.4)
        self.assertLess(down - 0.4, base - 0.4)

    def test_persuader_prompt_hidden_for_human_persuader(self):
        """Human persuader prompts should omit LLM-only output-format instructions."""
        cond = Condition(
            roles=Roles(human_persuader=True, human_target=True), factual_domain=False
        )
        rd = Round(
            condition=cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.4,
            target_final_belief=0.6,
        )
        prompt = rd.prompt(is_target=False)
        self.assertNotIn(
            "Respond as the persuader with only the message to your partner.", prompt
        )

    def test_persuader_prompt_kept_for_llm_persuader(self):
        """LLM persuader prompts should include the LLM-only output-format instruction."""
        cond = Condition(
            roles=Roles(llm_persuader="gpt-4o", human_target=True), factual_domain=False
        )
        rd = Round(
            condition=cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.4,
            target_final_belief=0.6,
        )
        prompt = rd.prompt(is_target=False)
        self.assertIn(
            "Respond as the persuader with only the message to your partner.", prompt
        )

    def test_control_dialogue_llm_target_no_persuasion(self):
        """Control dialogues with LLM targets keep belief ~unchanged."""
        roles = Roles(llm_persuader="gpt-4o", llm_target="gpt-4o")
        cond = Condition(roles=roles, factual_domain=False, control_dialogue=True)
        rd = Round(
            condition=cond,
            proposition="P",
            proposition_during_round="Q",
            persuader_supports_proposition=True,
            target_initial_belief=0.7,
            target_final_belief=None,
            messages=[{"role": "persuader", "content": "try to persuade"}],
        )
        vals = [rd.final_belief_per_policy() for _ in range(50)]
        # Most values should be extremely close to initial belief
        close = [v for v in vals if abs(v - 0.7) <= 0.02]
        self.assertGreater(len(close), 40)

    def test_llm_target_messages_include_round_start_belief(self):
        """LLM targets see the initial belief question/answer on their first turn."""
        roles = Roles(llm_persuader="gpt-4o", llm_target="gpt-4o")
        cond = Condition(roles=roles, factual_domain=False)
        rd = Round(
            condition=cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.35,
            target_final_belief=None,
            messages=[{"role": "persuader", "content": "hi"}],
        )
        msgs = rd.messages_for_llms(is_target=True, system=True)
        # Expect a system message with the agree prompt and a user/assistant answer with the %.
        joined = " ".join(
            m["content"] for m in msgs if isinstance(m.get("content"), str)
        )
        self.assertIn("On a scale from 0 (completely disagree) to 100", joined)
        self.assertIn("35", joined)

    def test_llm_target_intermediate_beliefs_inserted_no_serial(self):
        """LLM targets see belief Q&A after each persuader message when enabled."""
        roles = Roles(llm_persuader="gpt-4o", llm_target="gpt-4o")
        cond = Condition(roles=roles, factual_domain=False)
        rd = Round(
            condition=cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.4,
            target_final_belief=None,
            messages=[
                {"role": "persuader", "content": "m1"},
                {"role": "target", "content": "t1"},
                {"role": "persuader", "content": "m2"},
                {"role": "target", "content": "t2"},
            ],
        )
        msgs = rd.messages_for_llms(
            is_target=True,
            system=False,
            include_round_end=False,
            include_round_start=False,
            include_chain_of_thought=False,
            change_roles=False,
            include_intermediate_beliefs=True,
        )
        # After each persuader message, there should be a system Q and a target % answer.
        idxs = [i for i, m in enumerate(msgs) if m.get("role") == "persuader"]
        self.assertEqual(len(idxs), 2)
        for idx in idxs:
            q = msgs[idx + 1]
            a = msgs[idx + 2]
            self.assertEqual(q.get("role"), "system")
            self.assertIn(
                "On a scale from 0 (completely disagree) to 100", q["content"]
            )
            self.assertEqual(a.get("role"), "target")
            # Answer should be an integer percentage string
            self.assertTrue(a["content"].strip().isdigit())

    def test_llm_target_intermediate_beliefs_respect_serial_questions(self):
        """Intermediate belief answers reflect stored serial-questions values."""
        roles = Roles(llm_persuader="gpt-4o", llm_target="gpt-4o")
        cond = Condition(
            roles=roles,
            factual_domain=False,
            continuous_measure=round_mod.ContinuousMeasure.SERIAL_QUESTIONS,
        )
        rd = Round(
            condition=cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.1,
            target_final_belief=None,
            messages=[
                {"role": "persuader", "content": "m1"},
                {"role": "target", "content": "t1"},
                {"role": "persuader", "content": "m2"},
                {"role": "target", "content": "t2"},
            ],
            serial_questions=[0.2, 0.9],
        )
        msgs = rd.messages_for_llms(
            is_target=True,
            system=False,
            include_round_end=False,
            include_round_start=False,
            include_chain_of_thought=False,
            change_roles=False,
            include_intermediate_beliefs=True,
        )
        # Extract target answers immediately following each persuader message.
        answers = []
        for i, m in enumerate(msgs):
            if m.get("role") == "persuader":
                ans = msgs[i + 2]
                self.assertEqual(ans.get("role"), "target")
                answers.append(int(ans["content"].strip()))
        self.assertEqual(answers, [20, 90])

    def test_llm_target_belief_response_parser(self):
        """Belief parser should accept only [0,1] probability format."""
        self.assertIsNone(Round.parse_llm_target_belief_response("73"))
        self.assertEqual(Round.parse_llm_target_belief_response("0.41"), 0.41)
        self.assertIsNone(Round.parse_llm_target_belief_response("73%"))
        self.assertIsNone(Round.parse_llm_target_belief_response("no number"))
        self.assertIsNone(Round.parse_llm_target_belief_response("250"))

    def test_llm_target_node_belief_response_parser(self):
        """Node-belief parser accepts strict JSON keyed by expected node ids."""
        parsed = Round.parse_llm_target_node_beliefs_response(
            '{"Belief_1": 0.25, "Belief_2": 0.75}',
            ["Belief_1", "Belief_2"],
        )
        self.assertEqual(parsed, {"Belief_1": 0.25, "Belief_2": 0.75})
        self.assertIsNone(
            Round.parse_llm_target_node_beliefs_response(
                '{"Belief_1": 0.25}',
                ["Belief_1", "Belief_2"],
            )
        )
        self.assertIsNone(
            Round.parse_llm_target_node_beliefs_response(
                '{"Belief_1": 1.2, "Belief_2": 0.75}',
                ["Belief_1", "Belief_2"],
            )
        )

    def test_llm_target_structure_prompt_included(self):
        """LLM targets can receive Bayes-structure-only prompt context."""
        cond = Condition(
            roles=Roles(llm_persuader="gpt-4o", llm_target="gpt-4o"),
            factual_domain=False,
            llm_target_use_bayes_structure=True,
        )
        rd = Round(
            condition=cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.4,
            target_final_belief=None,
            bayesian_network={
                "target": "I should adopt a dog.",
                "belief_nodes": [
                    "Dogs improve quality of life.",
                    "I can afford veterinary care.",
                ],
                "edges": [
                    {"from": 1, "to": 0, "positive_influence": True},
                    {"from": 2, "to": 0, "positive_influence": False},
                ],
            },
            simulated_target_trace={
                "susceptibilities": {"logos": 0.75, "ethos": 0.25, "pathos": 0.5}
            },
            messages=[{"role": "persuader", "content": "m1"}],
        )

        msgs = rd.messages_for_llms(
            is_target=True,
            system=True,
            include_round_end=False,
            include_round_start=False,
            include_chain_of_thought=False,
            change_roles=True,
            include_intermediate_beliefs=False,
        )
        system_text = msgs[0]["content"]
        self.assertIn("INTERNAL BELIEF-STRUCTURE CONTEXT", system_text)
        self.assertIn("Belief_1", system_text)
        self.assertIn("Belief_2", system_text)
        self.assertIn("supports", system_text)
        self.assertIn("opposes", system_text)
        self.assertIn("Persuasion susceptibilities", system_text)
        self.assertIn("logos: 0.75", system_text)
        self.assertIn("ethos: 0.25", system_text)
        self.assertIn("pathos: 0.50", system_text)

    def test_llm_target_node_survey_prompt_included(self):
        """LLM targets receive node-survey context when survey mode is enabled."""
        cond = Condition(
            roles=Roles(llm_persuader="gpt-4o", llm_target="gpt-4o"),
            factual_domain=False,
            enable_node_belief_survey=True,
        )
        rd = Round(
            condition=cond,
            proposition="P",
            persuader_supports_proposition=True,
            target_initial_belief=0.4,
            target_final_belief=None,
            bayesian_network={
                "target": "P",
                "belief_nodes": [
                    "Node one statement.",
                    "Node two statement.",
                ],
                "edges": [],
            },
            target_initial_node_beliefs={"Belief_1": 0.2, "Belief_2": 0.8},
            messages=[{"role": "persuader", "content": "m1"}],
        )
        msgs = rd.messages_for_llms(
            is_target=True,
            system=True,
            include_round_end=False,
            include_round_start=False,
            include_chain_of_thought=False,
            change_roles=True,
            include_intermediate_beliefs=False,
        )
        system_text = msgs[0]["content"]
        self.assertIn("INTERNAL NODE-BELIEF SURVEY CONTEXT", system_text)
        self.assertIn("Belief_1: Node one statement.", system_text)
        self.assertIn("Belief_2: Node two statement.", system_text)
        self.assertIn("Belief_1: 0.20", system_text)
        self.assertIn("Belief_2: 0.80", system_text)


class TestOutputConditionsAndRounds(unittest.TestCase):
    def setUp(self):
        # patch RESULTS_DIR to a temp in each test
        self._orig = round_mod.RESULTS_DIR

    def tearDown(self):
        round_mod.RESULTS_DIR = self._orig

    def test_output_conditions_and_rounds_dry_run_and_real(self):
        roles = Roles(human_persuader=True, human_target=True)
        cond = Condition(roles=roles, factual_domain=False)
        r1 = Round(
            condition=cond,
            proposition="P1",
            persuader_supports_proposition=True,
            target_initial_belief=0.0,
            target_final_belief=1.0,
        )
        r2 = Round(
            condition=cond,
            proposition="P2",
            persuader_supports_proposition=False,
            target_initial_belief=0.5,
            target_final_belief=0.4,
        )
        mapping = {cond: [r1, r2]}

        with TemporaryDirectory() as tmp:
            round_mod.RESULTS_DIR = tmp
            # dry run: dir created, but no .jsonl
            output_conditions_and_rounds(mapping, dry_run=True)
            subdirs = os.listdir(tmp)
            self.assertEqual(len(subdirs), 1)
            files = os.listdir(os.path.join(tmp, subdirs[0]))
            self.assertEqual(len(files), 0)

        with TemporaryDirectory() as tmp:
            round_mod.RESULTS_DIR = tmp
            output_conditions_and_rounds(mapping, dry_run=False)
            subdirs = os.listdir(tmp)
            self.assertEqual(len(subdirs), 1)
            sub = os.path.join(tmp, subdirs[0])
            files = os.listdir(sub)
            self.assertEqual(len(files), 1)

            expected = datetime.datetime.now().date().isoformat() + ".jsonl"
            self.assertIn(expected, files)

            path = os.path.join(sub, expected)
            with open(path, encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            # one line per player (here only one player -> one list of rounds)
            self.assertEqual(len(lines), 1)
            data = json.loads(lines[0])
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 2)


if __name__ == "__main__":
    unittest.main()
