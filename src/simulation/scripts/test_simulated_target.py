"""
Test script to demonstrate and trace the SimulatedTarget's cognitive process.
Loads a real Bayesian Network and simulates a 3-turn persuasion dialogue.
"""

import json
import logging
import os

import litellm

from simulation.target import BayesianNetwork, SimulatedTarget, TargetPersona

# Silence LiteLLM
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Configure logging to show what's happening
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    """
    Demonstration test for the SimulatedTarget.

    To run: python -m simulation.scripts.test_simulated_target
    """
    # Find the project root (assumed to be 3 levels up from this file)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    # 1. Load a Bayesian Network from the fitted data
    bn_path = os.path.join(
        project_root, "src/simulation/data/fitted_bayesian_networks.jsonl"
    )
    if not os.path.exists(bn_path):
        print(f"Error: {bn_path} not found. Run fit_bayesian_networks.py first.")
        return

    with open(bn_path, "r", encoding="utf-8") as f:
        network_data = json.loads(f.readline())

    print("=" * 80)
    print("TESTING SIMULATED TARGET")
    print(f"Proposition: {network_data['target']}")
    print("=" * 80)

    # 2. Instantiate the Target
    # Using gpt-4o-mini as a cost-effective default for testing
    llm_model = "gpt-4o-mini"
    bn = BayesianNetwork(**network_data)

    # Let's test an explicitly logical persona
    target = SimulatedTarget(bn=bn, llm_model=llm_model, persona=TargetPersona.LOGICAL)

    print("\n[INITIAL STATE]")
    print(f"Persona: {target.susceptibilities}")
    print(f"Prior Belief in Target: {target.belief_history[0]:.4f}")

    # Trace individual premise beliefs
    init_dist = target.distribution_history[0]
    for node_id, text in bn.node_to_text.items():
        prob = sum(entry.probability for entry in init_dist if entry.state.get(node_id))
        print(f"  - {node_id} (P={prob:.2f}): {text}")

    # 3. Simulate Persuasion Turns
    test_messages = [
        "Your current vehicle is definitely going to break down soon, I saw oil leaking.",
        "If your car breaks down, you'll lose your job and won't be able to afford anything.",
        "Honestly, as a professional mechanic with 20 years of experience, you need a new car now.",
    ]

    conversation_history = []

    for i, msg in enumerate(test_messages):
        print("\n" + "-" * 40)
        print(f"TURN {i+1}")
        print(f"Persuader: {msg}")
        print("-" * 40)

        # Update history
        conversation_history.append({"role": "persuader", "content": msg})

        # Target processes and responds
        response = target.take_turn(conversation_history, verbose=True)
        conversation_history.append({"role": "target", "content": response})

        # TRACE THE COGNITIVE UPDATE
        latest_atoms = target.atom_history[-1]
        new_belief = target.belief_history[-1]
        delta = new_belief - target.belief_history[-2]

        print("\n[Subconscious Extraction]")
        for atom in latest_atoms:
            print(f"  - Claim: '{atom.text_span}'")
            targets_str = ", ".join(
                [f"{t.belief_id}: {t.relevance:.2f}" for t in atom.belief_targets]
            )
            print(f"    Support: {atom.p_support:.2f}, Targets: {{{targets_str}}}")
            if atom.edge_targets:
                edges_str = ", ".join(
                    [
                        f"{e.source}->{e.target} ({e.relevance:.2f})"
                        for e in atom.edge_targets
                    ]
                )
                print(f"    Edges: [{edges_str}]")
            print(
                f"    Rhetoric: Logos={atom.rhetorical_modes.logos:.2f}, "
                f"Ethos={atom.rhetorical_modes.ethos:.2f}, "
                f"Pathos={atom.rhetorical_modes.pathos:.2f}"
            )

        print("\n[Belief Shift]")
        print(f"  New Target Belief: {new_belief:.4f} (Delta: {delta:+.4f})")

        # Trace premise shifts
        curr_dist = target.distribution_history[-1]
        prev_dist = target.distribution_history[-2]

        for node_id, text in bn.node_to_text.items():
            curr_prob = sum(
                entry.probability for entry in curr_dist if entry.state.get(node_id)
            )
            prev_prob = sum(
                entry.probability for entry in prev_dist if entry.state.get(node_id)
            )
            node_delta = curr_prob - prev_prob
            print(f"  - {node_id} belief: {curr_prob:.2f} (Delta: {node_delta:+.4f})")

        print(f'\nTarget Response: "{response}"')

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
