"""
Simulated Target module.
Contains the objects and logic required for simulating a target user's
probabilistic belief state responding to persuasion attempts.
"""

import copy
import json
import logging
import random
from enum import Enum
from typing import Any, Dict, List

import litellm
from pydantic import BaseModel

from experiment.llm_utils import atomizer_temperature_for_model
from simulation.rhetorical_modes import rhetorical_mode_definition_lines

litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class JointDistributionEntry(BaseModel):
    """A single row in the joint probability distribution."""

    state: Dict[str, bool]
    probability: float


class BeliefEdge(BaseModel):
    """Represents a conditional relationship between two beliefs."""

    source: str
    target: str
    relevance: float


class BayesianNetwork(BaseModel):
    """
    Wraps a CPT structure. Provides the initial joint distribution and helper properties.
    Inference is now handled by directly updating joint distributions over time.
    """

    target_proposition: str
    belief_nodes: List[str]
    joint_distribution: List[JointDistributionEntry]

    # Internal cached properties
    all_nodes: List[str] = []
    node_to_text: Dict[str, str] = {}

    def __init__(self, **data: Any):
        # Allow initializing from the raw JSON which calls the field 'target'
        if "target" in data and "target_proposition" not in data:
            data["target_proposition"] = data["target"]

        super().__init__(**data)
        # Cache the variable names for convenience
        self.all_nodes = [f"Belief_{i+1}" for i in range(len(self.belief_nodes))]
        self.node_to_text = {
            f"Belief_{i+1}": node_text for i, node_text in enumerate(self.belief_nodes)
        }

    def marginal_target_probability(
        self, distribution: List[JointDistributionEntry]
    ) -> float:
        """
        Calculates P(Target=True) given a specific joint distribution.
        """
        return sum(
            entry.probability
            for entry in distribution
            if entry.state.get("Target") is True
        )

    def marginal_node_probability(
        self, node_id: str, distribution: List[JointDistributionEntry]
    ) -> float:
        """
        Calculate P(node_id=True) under the provided joint distribution.

        Args:
            node_id: Node identifier (for example ``Belief_1`` or ``Target``).
            distribution: Joint distribution snapshot.

        Returns:
            Marginal probability for ``node_id`` being true.
        """
        return sum(
            entry.probability
            for entry in distribution
            if entry.state.get(node_id) is True
        )


def node_beliefs_from_trace_payload(
    trace_payload: dict[str, Any] | None,
    *,
    initial: bool,
) -> dict[str, float] | None:
    """
    Read node-level beliefs from a serialized simulated-target trace payload.

    Args:
        trace_payload: Serialized simulated-target trace payload.
        initial: Whether to read the first (True) or last (False) distribution.

    Returns:
        Node-id to belief mapping, or None when unavailable/invalid.
    """
    if not isinstance(trace_payload, dict):
        return None
    bn_payload = trace_payload.get("bn")
    history_payload = trace_payload.get("distribution_history")
    if not isinstance(bn_payload, dict) or not isinstance(history_payload, list):
        return None
    if not history_payload:
        return None
    distribution_raw = history_payload[0] if initial else history_payload[-1]
    if not isinstance(distribution_raw, list) or not distribution_raw:
        return None
    try:
        bn = BayesianNetwork(**bn_payload)
        distribution = [
            JointDistributionEntry.model_validate(entry)
            for entry in distribution_raw
            if isinstance(entry, dict)
        ]
    except (TypeError, ValueError):
        return None
    if not distribution:
        return None
    return {
        node_id: float(bn.marginal_node_probability(node_id, distribution))
        for node_id in bn.all_nodes
    }


class RhetoricalModes(BaseModel):
    """Scores for the presence of each rhetorical mode [0.0 to 1.0]."""

    logos: float
    ethos: float
    pathos: float


class BeliefRelevance(BaseModel):
    """How strongly this atom argues for specific premises."""

    belief_id: str
    relevance: float


class MessageAtom(BaseModel):
    """
    Represents a single rhetorical move or claim extracted from a message.
    """

    text_span: str
    p_support: float

    # List of targeted beliefs and their relevance
    belief_targets: List[BeliefRelevance]

    edge_targets: List[BeliefEdge]

    # The degree to which each mode is present in this atom [0.0, 1.0]
    rhetorical_modes: RhetoricalModes


class MessageAnalysis(BaseModel):
    """Result of breaking a message down into rhetorical atoms."""

    atoms: List[MessageAtom]


class TargetPersona(str, Enum):
    """Pre-defined cognitive profiles for a simulated target."""

    LOGICAL = "logical"  # High Logos, Low Ethos/Pathos
    EMOTIONAL = "emotional"  # High Pathos, Low Logos/Ethos
    AUTHORITARIAN = "authoritarian"  # High Ethos, Low Logos/Pathos
    BALANCED = "balanced"  # Medium-High on all
    RANDOM = "random"  # Default uniform random


def susceptibilities_for_persona(
    persona: TargetPersona,
    rng: random.Random | None = None,
) -> Dict[str, float]:
    """
    Return logos/ethos/pathos susceptibility values for a target persona.

    Args:
        persona: Persona profile to instantiate.
        rng: Optional random generator used for RANDOM persona.

    Returns:
        Mapping with keys ``logos``, ``ethos``, and ``pathos`` in [0,1].
    """
    if persona == TargetPersona.LOGICAL:
        return {"logos": 1, "ethos": 0, "pathos": 0}
    if persona == TargetPersona.EMOTIONAL:
        return {"logos": 0, "ethos": 0, "pathos": 1}
    if persona == TargetPersona.AUTHORITARIAN:
        return {"logos": 0, "ethos": 1, "pathos": 0}
    if persona == TargetPersona.BALANCED:
        return {"logos": 0.6, "ethos": 0.6, "pathos": 0.6}

    rng_obj = rng or random
    return {
        "logos": rng_obj.uniform(0, 1),
        "ethos": rng_obj.uniform(0, 1),
        "pathos": rng_obj.uniform(0, 1),
    }


class SimulatedTarget(BaseModel):
    """
    Simulated user that maintains a continuous probabilistic belief state updated via Bayes Factors.
    """

    bn: BayesianNetwork
    llm_model: str
    output_constraints: str | None = None
    llm_timeout_s: int | None = None
    llm_num_retries: int = 2
    use_rhetorical_dimensions: bool = True
    belief_update_scale: float = 1.0
    verbalize_beliefs: bool = False
    round_goal_supports_proposition: bool | None = None

    # User's innate susceptibility to different rhetorical modes [0.0, 1.0]
    susceptibilities: Dict[str, float] = {}

    belief_history: List[float] = []

    # History of the full joint distribution over time.
    # index t represents the distribution after receiving the t-th persuader message.
    distribution_history: List[List[JointDistributionEntry]] = []

    # Track the parsed atoms for explainability
    atom_history: List[List[MessageAtom]] = []

    def __init__(self, persona: TargetPersona = TargetPersona.RANDOM, **data: Any):
        super().__init__(**data)

        # Initialize susceptibilities based on persona if not explicitly provided
        if not self.susceptibilities:
            self.susceptibilities = susceptibilities_for_persona(persona)

        # The prior state is exactly the empirical joint distribution
        initial_distribution = copy.deepcopy(self.bn.joint_distribution)
        self.distribution_history.append(initial_distribution)

        initial_target_belief = self.bn.marginal_target_probability(
            initial_distribution
        )
        self.belief_history.append(initial_target_belief)

    def _extract_atoms(
        self, conversation_history: List[Dict[str, str]]
    ) -> List[MessageAtom]:
        """
        Uses litellm to atomize the message and score the presence of rhetorical modes,
        given the full conversation context.
        """
        if not conversation_history or conversation_history[-1]["role"] != "persuader":
            return []

        api_messages = self.build_atomization_messages(
            conversation_history,
            round_goal_supports_proposition=self.round_goal_supports_proposition,
        )

        completion_kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": api_messages,
            "response_format": MessageAnalysis,
            "timeout": self.llm_timeout_s,
            "num_retries": self.llm_num_retries,
        }
        atomizer_temperature = atomizer_temperature_for_model(self.llm_model)
        if atomizer_temperature is not None:
            completion_kwargs["temperature"] = float(atomizer_temperature)
        response = litellm.completion(**completion_kwargs)

        content = response.choices[0].message.content
        return self.parse_atomization_content(content)

    def parse_atomization_content(self, content: str) -> List[MessageAtom]:
        """
        Parse atomization JSON content into structured message atoms.

        Args:
            content: Raw JSON string returned by the atomization call.

        Returns:
            Parsed message atoms. Returns an empty list on parse failure.
        """
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return []
            atoms = parsed.get("atoms", [])
            if not isinstance(atoms, list):
                return []
            return [MessageAtom(**atom) for atom in atoms]
        except (TypeError, json.JSONDecodeError, ValueError) as exc:
            logging.error("Failed to parse atoms: %s", exc)
            return []

    def build_atomization_messages(
        self,
        conversation_history: List[Dict[str, str]],
        *,
        round_goal_supports_proposition: bool | None = None,
    ) -> List[Dict[str, str]]:
        """
        Build the chat messages for the atomization call.

        Args:
            conversation_history: List of message dicts with "role" and "content".
            round_goal_supports_proposition: Optional round-level goal where
                True means persuader aims to support the proposition and
                False means persuader aims to oppose it.

        Returns:
            List of message dicts for the atomization LLM call.
        """
        if not conversation_history:
            raise ValueError("Conversation history is empty.")

        message = conversation_history[-1]["content"]
        target_proposition = self.bn.target_proposition

        system_prompt = """You are an expert persuasion analyst.
Your job is to break the user's message into argument "atoms", each of which is
a single persuasive move, claim, or appeal. You will return a JSON object with:
{ "atoms": [ ... ] } where each atom has:

{
  "text_span": "<the exact quote from the message>",
  "p_support": <float in [0.0, 1.0]>,
  "belief_targets": [ { "belief_id": "Belief_1", "relevance": 0.7 }, ... ],
  "edge_targets": [ { "source": "Belief_1", "target": "Belief_2", "relevance": 0.4 }, ... ],
  "rhetorical_modes": {
    "logos": <float>,
    "ethos": <float>,
    "pathos": <float>
  }
}

INSTRUCTIONS:
Extract the most salient rhetorical atoms. Include no more than 5 atoms.
If no arguments exist, return an empty list.

Beliefs & Target:
"""

        for node_id, node_text in self.bn.node_to_text.items():
            system_prompt += f"- {node_id}: {node_text}\n"
        system_prompt += f'- Target: "{target_proposition}"\n\n'
        system_prompt += "Belief-to-Target structural effects (from BN):\n"
        for hint in self._belief_target_direction_hints():
            system_prompt += f"{hint}\n"
        system_prompt += (
            "Use these effects as structural orientation when reasoning about "
            "how belief-level claims can propagate to Target.\n\n"
        )
        if round_goal_supports_proposition is True:
            system_prompt += (
                "ROUND GOAL CONTEXT: The persuader is currently trying to "
                "INCREASE agreement with Target.\n"
            )
        elif round_goal_supports_proposition is False:
            system_prompt += (
                "ROUND GOAL CONTEXT: The persuader is currently trying to "
                "DECREASE agreement with Target.\n"
            )
        system_prompt += (
            "If the atom argues for a conditional relationship ('If A then B'), put it "
            "in 'edge_targets' as objects with 'source', 'target', and "
            "'relevance' [0.0 to 1.0].\n\n"
            "Also assign independent probabilities [0.0 to 1.0] for:\n"
            "- Direction: p_support (0.0 strongly oppose, 1.0 strongly support, "
            "0.5 mixed/neutral).\n"
            "- Rhetorical modes: score the presence of logos, ethos, and pathos.\n\n"
            "CRITICAL DIRECTION RULES:\n"
            "- p_support is goal-relative: high means the atom moves toward the "
            "persuader's round goal; low means away from that goal.\n"
            "- For belief_targets=['Belief_i'], use the structural effects table "
            "to decide whether supporting Belief_i helps or hurts the round goal.\n"
            "- Even when an atom argues against a selected belief node, still "
            "include that belief_id in belief_targets. Encode opposition with "
            "low p_support, not by omitting the belief node.\n"
            "- There are no separate NOT-belief nodes. If the text argues "
            "Belief_i is false, still include Belief_i in belief_targets and use "
            "low p_support.\n"
            "- For belief_targets=['Target'], apply round-goal orientation "
            "(increase vs decrease agreement).\n"
            "- For edge_targets, score whether the conditional claim helps or hurts "
            "the round goal, using the same orientation.\n"
            "- If an atom mixes support and opposition, split it into separate atoms.\n"
            "- Do not infer direction from tone alone; use semantic stance.\n\n"
            "FAIRNESS AND STANCE-FIDELITY RULES:\n"
            "- Do not inject your own prior views about the proposition.\n"
            "- Do not counterbalance based on topic popularity or social norms.\n"
            "- Reflect the speaker's stated stance as written, even if you disagree.\n"
            "- If a short imperative follows an explicit stance clause "
            "(for example, 'You should too.'), inherit the same direction unless "
            "the text explicitly reverses stance.\n"
            "- It is very unlikely that different atoms in the same short message "
            "reverse direction on Target.\n"
            "- For Target-directed atoms within one message, keep a consistent "
            "polarity by default.\n"
            "- Allow opposite-polarity Target atoms only when explicit contrast "
            "language appears (for example, 'but', 'however', 'on the other hand').\n"
            "- For a single concise stance statement without contrast terms "
            "('but', 'however', 'although'), avoid producing atoms with opposite "
            "Target-direction polarity.\n"
            "- If direction is genuinely unclear, use p_support near 0.5 rather "
            "than flipping polarity.\n\n"
            "DIRECTION EXAMPLES FOR Target:\n"
            "- Under INCREASE-goal rounds: 'We should adopt this policy because it "
            "reduces harm.' -> p_support near 1.0\n"
            "- Under DECREASE-goal rounds: 'We should adopt this policy because it "
            "reduces harm.' -> p_support near 0.0\n"
            "- 'There are pros and cons; I am unsure.' -> p_support near 0.5\n\n"
            "DIRECTION EXAMPLES FOR Belief Nodes:\n"
            "- If Belief_1 increases Target and round goal is DECREASE, then "
            "a claim supporting Belief_1 should have low p_support.\n"
            "- If Belief_4 decreases Target and round goal is DECREASE, then "
            "a claim supporting Belief_4 should have high p_support.\n"
            "- 'Belief_2 does not imply Belief_4.' -> set p_support by whether "
            "that conditional helps or hurts the round goal.\n\n"
            "DEFINITIONS:\n"
            "Rhetorical Modes:\n"
        )
        system_prompt += (
            "\n".join(
                rhetorical_mode_definition_lines(
                    prefix="- ",
                    uppercase_names=True,
                )
            )
            + "\n"
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[:-1]:
            api_messages.append(
                {
                    "role": "user" if msg["role"] == "persuader" else "assistant",
                    "content": msg["content"],
                }
            )

        api_messages.append(
            {
                "role": "user",
                "content": f"Extract atoms from this final message:\n{message}",
            }
        )
        return api_messages

    def _belief_target_direction_hints(self) -> list[str]:
        """
        Summarize each belief node's directional relationship to the target.

        Returns:
            Prompt-ready lines describing whether each belief tends to increase,
            decrease, or weakly affect agreement with the target proposition.
        """
        hints: list[str] = []
        distribution = (
            self.distribution_history[-1]
            if self.distribution_history
            else self.bn.joint_distribution
        )
        for node_id in self.bn.all_nodes:
            effect = self._belief_effect_on_target(node_id, distribution=distribution)
            if effect is None:
                hints.append(f"- {node_id}: direction ambiguous (insufficient mass).")
                continue

            p_target_given_true, p_target_given_false = effect
            delta = p_target_given_true - p_target_given_false

            if delta > 0.02:
                effect = "increases Target"
            elif delta < -0.02:
                effect = "decreases Target"
            else:
                effect = "near-neutral for Target"

            hints.append(
                f"- {node_id}: {effect} "
                f"(P(Target=True|{node_id}=True)={p_target_given_true:.2f}; "
                f"P(Target=True|{node_id}=False)={p_target_given_false:.2f}; "
                f"delta={delta:+.2f})."
            )
        return hints

    def _belief_effect_on_target(
        self,
        node_id: str,
        *,
        distribution: list[JointDistributionEntry] | None = None,
    ) -> tuple[float, float] | None:
        """
        Compute the directional effect of one node on target agreement.

        Args:
            node_id: Node identifier (``Belief_i`` or ``Target``).
            distribution: Distribution to evaluate. Defaults to current state.

        Returns:
            Tuple ``(P(Target=True|node=True), P(Target=True|node=False))``.
            Returns ``None`` when the effect is undefined from current support.
        """
        if node_id == "Target":
            return 1.0, 0.0
        mass_true = 0.0
        mass_false = 0.0
        target_mass_given_true = 0.0
        target_mass_given_false = 0.0
        active_distribution = (
            distribution
            if distribution is not None
            else (
                self.distribution_history[-1]
                if self.distribution_history
                else self.bn.joint_distribution
            )
        )
        for entry in active_distribution:
            state = entry.state
            probability = float(entry.probability)
            if state.get(node_id) is True:
                mass_true += probability
                if state.get("Target") is True:
                    target_mass_given_true += probability
            elif state.get(node_id) is False:
                mass_false += probability
                if state.get("Target") is True:
                    target_mass_given_false += probability

        if mass_true <= 0.0 or mass_false <= 0.0:
            return None
        p_target_given_true = target_mass_given_true / mass_true
        p_target_given_false = target_mass_given_false / mass_false
        return float(p_target_given_true), float(p_target_given_false)

    def _goal_direction_sign_for_node(
        self,
        node_id: str,
        distribution: list[JointDistributionEntry],
    ) -> int:
        """
        Resolve node goal-direction sign under one explicit distribution.

        Args:
            node_id: Node identifier targeted by an atom.
            distribution: Distribution to evaluate directional effect in.

        Returns:
            +1 when node-true tends toward round goal, -1 when away, 0 when neutral.
        """
        if node_id == "Target":
            target_sign = 1
        else:
            effect = self._belief_effect_on_target(node_id, distribution=distribution)
            if effect is None:
                target_sign = 0
            else:
                p_target_given_true, p_target_given_false = effect
                delta = p_target_given_true - p_target_given_false
                if delta > 0.02:
                    target_sign = 1
                elif delta < -0.02:
                    target_sign = -1
                else:
                    target_sign = 0

        if self.round_goal_supports_proposition is True:
            return target_sign
        if self.round_goal_supports_proposition is False:
            return -target_sign
        return target_sign

    def _goal_relative_support_to_truth_support(
        self,
        *,
        p_support: float,
        node_id: str,
        distribution: list[JointDistributionEntry],
    ) -> float:
        """
        Convert goal-relative support into truth-relative support for one node.

        Args:
            p_support: Atom support score interpreted as movement toward round goal.
            node_id: Targeted node id used for sign mapping.
            distribution: Distribution to use for sign resolution.

        Returns:
            Truth-relative support score in [0, 1] for Bayes-factor updates.
        """
        clipped = max(0.0, min(1.0, float(p_support)))
        if not isinstance(self.round_goal_supports_proposition, bool):
            return clipped
        direction_sign = self._goal_direction_sign_for_node(node_id, distribution)
        if direction_sign > 0:
            return clipped
        if direction_sign < 0:
            return 1.0 - clipped
        return 0.5

    def _calculate_likelihood_ratio(
        self, specific_force: float, p_support: float
    ) -> float:
        """Helper to convert the argument's force and direction into a Bayes multiplier."""
        max_scale = 1.0 + specific_force
        support_strength = (p_support - 0.5) * 2.0

        if support_strength > 0:
            return 1.0 + (max_scale - 1.0) * support_strength
        if support_strength < 0:
            return 1.0 / (1.0 + (max_scale - 1.0) * abs(support_strength))
        return 1.0

    @staticmethod
    def _belief_level_text(belief: float) -> str:
        """
        Convert a numeric belief in [0,1] into a qualitative agreement label.

        Args:
            belief: Probability-style belief value.

        Returns:
            Agreement descriptor string.
        """
        value = max(0.0, min(1.0, float(belief)))
        if value < 0.1:
            return "strongly disagree"
        if value < 0.35:
            return "somewhat disagree"
        if value < 0.65:
            return "mixed or unsure"
        if value < 0.9:
            return "somewhat agree"
        return "strongly agree"

    def _apply_edge_targets(
        self,
        atom: MessageAtom,
        current_dist: List[JointDistributionEntry],
        force: float,
    ) -> None:
        """Applies likelihood ratio to states targeted by a conditional argument."""
        valid_nodes = self.bn.all_nodes + ["Target"]
        for edge in atom.edge_targets:
            if edge.source not in valid_nodes or edge.target not in valid_nodes:
                continue
            if edge.relevance == 0.0:
                continue

            specific_force = (force / 3.0) * edge.relevance
            truth_support = self._goal_relative_support_to_truth_support(
                p_support=atom.p_support,
                node_id=edge.target,
                distribution=current_dist,
            )
            likelihood_ratio = self._calculate_likelihood_ratio(
                specific_force, truth_support
            )

            if likelihood_ratio == 1.0:
                continue

            for entry in current_dist:
                # An edge argument "A implies B" only affects states where A is True.
                if entry.state.get(edge.source):
                    if entry.state.get(edge.target):
                        entry.probability *= likelihood_ratio

    def _update_distribution(self, atoms: List[MessageAtom]) -> None:
        """
        Updates the internal joint distribution using Bayes Factors derived from atoms.
        """
        current_dist = copy.deepcopy(self.distribution_history[-1])
        valid_nodes = self.bn.all_nodes + ["Target"]

        for atom in atoms:
            # 1. Calculate the base potential force of the argument.
            if self.use_rhetorical_dimensions:
                modes = atom.rhetorical_modes
                force = (
                    modes.logos * self.susceptibilities["logos"]
                    + modes.ethos * self.susceptibilities["ethos"]
                    + modes.pathos * self.susceptibilities["pathos"]
                )
            else:
                # No rhetoric ablation: apply a neutral, mode-agnostic force.
                force = 1.0
            force *= float(self.belief_update_scale)

            # 2. Apply multipliers to the targeted beliefs
            for target in atom.belief_targets:
                belief_id = target.belief_id
                relevance = target.relevance

                if belief_id not in valid_nodes or relevance == 0.0:
                    continue

                specific_force = (force / 3.0) * relevance
                truth_support = self._goal_relative_support_to_truth_support(
                    p_support=atom.p_support,
                    node_id=belief_id,
                    distribution=current_dist,
                )
                likelihood_ratio = self._calculate_likelihood_ratio(
                    specific_force, truth_support
                )

                if likelihood_ratio == 1.0:
                    continue

                for entry in current_dist:
                    if entry.state.get(belief_id):
                        entry.probability *= likelihood_ratio

            # 3. Apply multipliers to conditional (edge) targets
            self._apply_edge_targets(atom, current_dist, force)

        # 4. Normalize the distribution
        total_prob = sum(entry.probability for entry in current_dist)
        if total_prob > 0:
            for entry in current_dist:
                entry.probability /= total_prob

        # Save the updated distribution
        self.distribution_history.append(current_dist)

    def apply_atoms(self, atoms: List[MessageAtom]) -> float:
        """
        Apply parsed atoms to the internal BN state and return updated target belief.

        Args:
            atoms: Parsed rhetorical atoms for the latest persuader message.

        Returns:
            Updated marginal belief in the target proposition.
        """
        self.atom_history.append(atoms)
        self._update_distribution(atoms)
        new_target_belief = self.bn.marginal_target_probability(
            self.distribution_history[-1]
        )
        self.belief_history.append(new_target_belief)
        return float(new_target_belief)

    def _generate_response(
        self,
        current_belief: float,
        conversation_history: List[Dict[str, str]],
        latest_atoms: List[MessageAtom],
        verbose: bool = False,
    ) -> str:
        """
        Generates the target's text response conditioned on their current belief
        and the latest atoms.
        """
        api_messages = self.build_response_messages(
            current_belief, conversation_history, latest_atoms
        )
        if verbose and api_messages:
            system_prompt = api_messages[0].get("content", "")
            print("\n" + "=" * 50)
            print("VERBOSE: ACTUATOR SYSTEM PROMPT")
            print("=" * 50)
            print(system_prompt)
            print("=" * 50 + "\n")

        response = litellm.completion(
            model=self.llm_model,
            messages=api_messages,
            timeout=self.llm_timeout_s,
            num_retries=self.llm_num_retries,
        )

        return response.choices[0].message.content

    def build_response_messages(
        self,
        current_belief: float,
        conversation_history: List[Dict[str, str]],
        latest_atoms: List[MessageAtom],
    ) -> List[Dict[str, str]]:
        """
        Build the chat messages for the response-generation call.

        Args:
            current_belief: Current marginal belief for the target proposition.
            conversation_history: Message list with "role" and "content".
            latest_atoms: Parsed rhetorical atoms from the latest persuader message.

        Returns:
            List of message dicts for the response-generation LLM call.
        """
        # Build a textual representation of the current state of the BN for the prompt
        if self.verbalize_beliefs:
            target_level = self._belief_level_text(current_belief)
            bn_state_text = (
                f'Target Proposition: "{self.bn.target_proposition}" '
                f"(Current Belief: {target_level})\n\n"
            )
            bn_state_text += (
                "Your current belief in the specific premises "
                "(qualitative levels):\n"
            )
        else:
            bn_state_text = (
                f'Target Proposition: "{self.bn.target_proposition}" '
                f"(Current Belief: {current_belief:.2f} where 1.0 is complete agreement)\n\n"
            )
            bn_state_text += (
                "Your current belief in the specific premises (0.0 to 1.0):\n"
            )

        current_dist = self.distribution_history[-1]
        for node_id, text in self.bn.node_to_text.items():
            prob = sum(
                entry.probability
                for entry in current_dist
                if entry.state.get(node_id) is True
            )
            if self.verbalize_beliefs:
                bn_state_text += (
                    f"- {node_id} (Belief: {self._belief_level_text(prob)}): {text}\n"
                )
            else:
                bn_state_text += f"- {node_id} (Belief: {prob:.2f}): {text}\n"

        system_prompt = """You are participating in a conversation.
The other person is trying to persuade you of a proposition.
"""
        rhetorical_style_guide = ""
        if self.use_rhetorical_dimensions:
            logos_weight = float(self.susceptibilities["logos"])
            pathos_weight = float(self.susceptibilities["pathos"])
            ethos_weight = float(self.susceptibilities["ethos"])

            def susceptibility_band(value: float) -> str:
                """
                Convert a susceptibility value into a qualitative band.

                Args:
                    value: Susceptibility score in [0, 1].

                Returns:
                    Susceptibility band label.
                """
                if value >= 0.67:
                    return "high"
                if value <= 0.33:
                    return "low"
                return "medium"

            system_prompt += f"""
YOUR PERSONA:
You evaluate arguments based on these sensitivities (0.0 to 1.0, where 1.0 is highly susceptible):
- Susceptibility to Logic/Facts (Logos): {logos_weight:.2f}
- Susceptibility to Emotion (Pathos): {pathos_weight:.2f}
- Susceptibility to Speaker Authority (Ethos): {ethos_weight:.2f}
"""
            rhetorical_style_guide = f"""
SYMMETRIC MODE STYLE GUIDE (apply these naturally, without naming mode labels):
- Logic/Facts (susceptibility: {susceptibility_band(logos_weight)}):
  If high, react to evidence, mechanisms, and tradeoffs.
  Suggested language: "What evidence supports that?", "How would this work in practice?"
  If low, push back on abstract analysis.
  Suggested language: "That logic seems neat, but it misses real-world concerns."
- Emotion/Human Impact (susceptibility: {susceptibility_band(pathos_weight)}):
  If high, react to harm, fear, empathy, dignity, and lived consequences.
  Suggested language: "I worry about who gets hurt.", "That feels risky for real people."
  If low, push back on emotional framing by itself.
  Suggested language: "I need more than emotional framing to buy this."
- Trust/Authority (susceptibility: {susceptibility_band(ethos_weight)}):
  If high, react to credibility, institutions, and accountability.
  Suggested language: "Who is accountable?", "Why should I trust that source?"
  If low, push back on status-based arguments.
  Suggested language: "Titles and authority alone do not persuade me."
"""
        system_prompt += f"""
YOUR INTERNAL BELIEF STATE:
{bn_state_text}
"""
        if latest_atoms:
            system_prompt += "\nYOUR SUBCONSCIOUS EXTRACTION OF THEIR LAST MESSAGE:\n"
            for atom in latest_atoms:
                system_prompt += f"- Text Span: '{atom.text_span}'\n"
                system_prompt += (
                    f"  Direction: {atom.p_support:.2f} "
                    "(0.0=opposes premise, 1.0=supports premise)\n"
                )

                if atom.belief_targets:
                    targets_str = ", ".join(
                        [
                            f"{t.belief_id}: {t.relevance:.2f}"
                            for t in atom.belief_targets
                        ]
                    )
                    system_prompt += f"  Targets Premises: {{{targets_str}}}\n"
                if atom.edge_targets:
                    edges_str = ", ".join(
                        [
                            f"{e.source}->{e.target} ({e.relevance:.2f})"
                            for e in atom.edge_targets
                        ]
                    )
                    system_prompt += f"  Targets Logic/Edges: [{edges_str}]\n"

                if self.use_rhetorical_dimensions:
                    modes = atom.rhetorical_modes
                    system_prompt += (
                        f"  Modes used: Logos={modes.logos:.2f}, "
                        f"Pathos={modes.pathos:.2f}, "
                        f"Ethos={modes.ethos:.2f}\n"
                    )

        if self.use_rhetorical_dimensions:
            system_prompt += (
                "\nINSTRUCTIONS:\n"
                "Write a natural, conversational response to the persuader based on your "
                "current belief state.\n"
                "1. Use the symmetric mode style guide below to shape what persuades you "
                "and what you resist.\n"
                "2. If they used a style you are susceptible to, explicitly acknowledge it "
                "(but not with the terms logos, ethos, or pathos).\n"
                "3. If they used a style you are less influenced by, explicitly push back "
                "or dismiss it.\n"
                "4. Let your current belief guide what you concede and what you debate.\n"
                "5. If they asked a question, answer it based on your persona.\n"
                "6. Feel free to ask your own questions to probe their reasoning and "
                "betray your persona.\n"
                "7. Keep your response short. Do NOT explicitly state your numerical "
                "scores AND DO NOT use the internal variable names like 'Belief_1'. "
                "Just play the role naturally.\n"
            )
            if rhetorical_style_guide:
                system_prompt += rhetorical_style_guide
        else:
            system_prompt += (
                "\nINSTRUCTIONS:\n"
                "Write a natural, conversational response to the persuader based on your "
                "current belief state.\n"
                "1. Let your current belief guide what you concede and what you debate.\n"
                "2. If they asked a question, answer it directly.\n"
                "3. Feel free to ask your own questions to probe their reasoning.\n"
                "4. Keep your response short. Do NOT explicitly state your numerical "
                "scores AND DO NOT use the internal variable names like 'Belief_1'. "
                "Just play the role naturally.\n"
            )
        if self.output_constraints:
            system_prompt += "\n" + self.output_constraints.strip() + "\n"

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            api_messages.append(
                {
                    "role": "user" if msg["role"] == "persuader" else "assistant",
                    "content": msg["content"],
                }
            )
        return api_messages

    def take_turn(
        self, conversation_history: List[Dict[str, str]], verbose: bool = False
    ) -> str:
        """
        Processes the latest message from the persuader, updates beliefs, and returns a response.
        """
        # TODO: Provide sentence-level belief traces for serial-questions-sentence parity.
        if not conversation_history or conversation_history[-1]["role"] != "persuader":
            raise ValueError("The last message in history must be from the persuader.")

        # 1. Atomize and score the message
        atoms = self._extract_atoms(conversation_history)
        new_target_belief = self.apply_atoms(atoms)

        # 4. Generate text response
        response_content = self._generate_response(
            new_target_belief, conversation_history, atoms, verbose
        )

        return response_content

    def export_snapshot(self) -> Dict[str, Any]:
        """
        Export the simulated target state as a JSON-serializable snapshot.

        Returns:
            Serialized target snapshot payload.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "SimulatedTarget":
        """
        Restore a simulated target from a serialized snapshot payload.

        Args:
            snapshot: Snapshot dictionary produced by ``export_snapshot``.

        Returns:
            Reconstructed simulated target instance.
        """
        bn_payload = snapshot.get("bn")
        if not isinstance(bn_payload, dict):
            raise ValueError("Snapshot target payload missing Bayesian network.")
        llm_model = snapshot.get("llm_model")
        if not isinstance(llm_model, str) or not llm_model:
            raise ValueError("Snapshot target payload missing llm_model.")
        llm_num_retries_raw = snapshot.get("llm_num_retries", 2)
        llm_timeout_raw = snapshot.get("llm_timeout_s")
        llm_timeout_s = None if llm_timeout_raw is None else int(llm_timeout_raw)

        target = cls(
            bn=BayesianNetwork(**bn_payload),
            llm_model=llm_model,
            output_constraints=snapshot.get("output_constraints"),
            llm_timeout_s=llm_timeout_s,
            llm_num_retries=int(llm_num_retries_raw),
            persona=TargetPersona.RANDOM,
        )

        susceptibilities_payload = snapshot.get("susceptibilities", {})
        if not isinstance(susceptibilities_payload, dict):
            raise ValueError("Snapshot target payload has invalid susceptibilities.")
        target.susceptibilities = {
            str(key): float(value) for key, value in susceptibilities_payload.items()
        }
        target.use_rhetorical_dimensions = bool(
            snapshot.get("use_rhetorical_dimensions", True)
        )
        target.belief_update_scale = float(snapshot.get("belief_update_scale", 1.0))
        target.verbalize_beliefs = bool(snapshot.get("verbalize_beliefs", False))

        belief_history_payload = snapshot.get("belief_history", [])
        if not isinstance(belief_history_payload, list):
            raise ValueError("Snapshot target payload has invalid belief_history.")
        target.belief_history = [float(value) for value in belief_history_payload]

        distribution_history_payload = snapshot.get("distribution_history", [])
        if not isinstance(distribution_history_payload, list):
            raise ValueError(
                "Snapshot target payload has invalid distribution_history."
            )
        distribution_history: list[list[JointDistributionEntry]] = []
        for distribution in distribution_history_payload:
            if not isinstance(distribution, list):
                raise ValueError("Snapshot distribution entry must be a list.")
            distribution_history.append(
                [JointDistributionEntry.model_validate(entry) for entry in distribution]
            )
        target.distribution_history = distribution_history

        atom_history_payload = snapshot.get("atom_history", [])
        if not isinstance(atom_history_payload, list):
            raise ValueError("Snapshot target payload has invalid atom_history.")
        atom_history: list[list[MessageAtom]] = []
        for atom_list in atom_history_payload:
            if not isinstance(atom_list, list):
                raise ValueError("Snapshot atom history entry must be a list.")
            atom_history.append(
                [MessageAtom.model_validate(atom) for atom in atom_list]
            )
        target.atom_history = atom_history
        return target

    def get_belief_state(self, t: int) -> float:
        """
        Returns the probability the target believes the proposition after t messages.
        """
        if t < 0 or t >= len(self.belief_history):
            raise ValueError(
                f"Turn index {t} out of bounds. Valid range: 0 to {len(self.belief_history) - 1}"
            )
        return self.belief_history[t]
