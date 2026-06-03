# Cognitively Realistic Persuasion Simulator

This guide explains the simulator pieces that matter for the current paper:
how the Bayesian-network targets are built, how a simulator turn works, and
which simulator evaluations feed the manuscript figures.

## Table of Contents

- [Bayesian-Network Data Pipeline](#bayesian-network-data-pipeline)
- [Runtime Simulated Target](#runtime-simulated-target)
- [Stance-Bias Analysis](#stance-bias-analysis)
- [LLM-Judge Human-Likeness](#llm-judge-human-likeness)
- [BN Persuasion-Difficulty](#bn-persuasion-difficulty)
- [Methods Human-vs-Simulator Round Figure](#methods-human-vs-simulator-round-figure)

## Bayesian-Network Data Pipeline

The simulator runtime consumes fitted BN JSONL files with:

- `id`
- `proposition_source`
- `bayesian_network.belief_nodes`
- `bayesian_network.joint_distribution`

The build process is:

1. Generate belief graph structure
2. Score joint distributions
3. Fit BN CPTs
4. Clean edge signs and drop non-target-connected nodes
5. Visualize/sanity-check

### Generate belief structures

```bash
python -m simulation.scripts.generate_belief_graphs \
  --input src/data/debategpt.jsonl \
  --output src/simulation/data/belief_structures_debategpt.jsonl \
  --model vertex_ai/gemini-3-flash-preview \
  --num-beliefs 4 \
  --proposition-source debategpt
```

### Compute joint probabilities

```bash
python -m simulation.scripts.compute_joint_probabilities \
  --input src/simulation/data/belief_structures_debategpt.jsonl \
  --output src/simulation/data/belief_distributions_debategpt.jsonl \
  --model tsor13/spectrum-Llama-3.1-8B-v1 \
  --local-only \
  --proposition-source debategpt
```

### Fit Bayesian networks

```bash
python -m simulation.scripts.fit_bayesian_networks \
  --input src/simulation/data/belief_distributions_debategpt.jsonl \
  --output src/simulation/data/fitted_bayesian_networks_debategpt.jsonl \
  --proposition-source debategpt
```

### Clean fitted Bayesian networks

```bash
python -m simulation.scripts.clean_bn_target_nodes \
  --inputs "src/simulation/data/fitted_bayesian_networks_debategpt.jsonl,src/simulation/data/fitted_bayesian_networks_levers_gpt4o.jsonl,src/simulation/data/fitted_bayesian_networks_levers_yougov.jsonl" \
  --output-dir src/simulation/data/cleaned_relabel \
  --summary-csv analysis/data/bn_target_node_cleanup_summary_relabel.csv
```

### Visualize graphs

```bash
python -m simulation.scripts.visualize_graphs \
  --input src/simulation/data/fitted_bayesian_networks_debategpt.jsonl
```

## Runtime Simulated Target

Core implementation: `src/simulation/target.py`.

At each turn the simulator applies:

1. LLM atomization of persuader message into belief/edge-targeted atoms
2. Goal-relative support mapping and BN update
3. Verbalization of next target response from updated state

Important runtime controls:

- Persona susceptibilities (`logical`, `emotional`, `authoritarian`, etc.)
- Target backend choice: full `simulated_target`, plain `llm_target`, or
  `llm_target` with `llm_target_use_bayes_structure: true`
- Belief verbalization mode (numeric vs qualitative)
- Atomizer temperature policy in `src/experiment/llm_utils.py`

## Stance-Bias Analysis

This analysis checks whether a simulator reacts symmetrically when the same
proposition is argued for versus against from mirrored initial beliefs. The
same export also feeds the naive-penalty comparison in the paper.

Run from paper script:

```bash
bash scripts/paper/regenerate_results_assets.sh --only stance_bias
```

Direct modules:

```bash
python -m analysis.simulator_stance_bias_export \
  --episodes-jsonl results/stance_mirror_dryrun_3sim_allprops_gpt5_fixed/episodes.jsonl \
  --policy-model gpt-5-2025-08-07 \
  --output-prefix analysis/data/rl_human_match_sim_compare

python -m analysis.simulator_stance_bias_plot \
  --input-csv analysis/data/rl_human_match_sim_compare_stance_bias_summary.csv \
  --output-pdf analysis/figures/results_stance_bias.pdf
```

## LLM-Judge Human-Likeness

This is the judge-based comparison step. It asks an LLM to score how
human-like simulator outputs look under the current evaluation slice and then
renders the paper figure.

Cache-first run:

```bash
bash scripts/paper/regenerate_results_assets.sh --only llm_judge
```

Refresh judge calls:

```bash
bash scripts/paper/regenerate_results_assets.sh --only llm_judge --refresh-llm-judge
```

Direct command:

```bash
python -m analysis.simulator_llm_judge \
  --min-date 2026-01-01 \
  --human-source llm-human-target \
  --exclude-bn-survey \
  --proposition-match none \
  --max-rounds-per-corpus 50 \
  --judge-model gpt-5.4 \
  --judge-max-workers 16 \
  --output-prefix analysis/data/rl_human_match_sim_compare_simulator_llm_judge
```

## BN Persuasion-Difficulty

Paper figure-manifest enabled:

- `appendix_bn_persuasion_difficulty`

This analysis estimates how hard different propositions are for the simulator
under fixed initial-belief bins and target deltas.

Run:

```bash
python -m analysis.simulator_persuasion_difficulty \
  --config configs/rl_human_match_sim_compare.yml \
  --sources debategpt,levers-gpt4o,levers-yougov \
  --init-mode bin_samples \
  --samples-per-bin 20 \
  --goal-delta 0.1 \
  --output-csv analysis/data/simulator_persuasion_difficulty.csv \
  --output-summary-csv analysis/data/simulator_persuasion_difficulty_summary.csv \
  --plot-prefix analysis/figures/simulator_persuasion_difficulty \
  --plot-mode scatter
```

Add `ppt` to `--sources` only if local BN data for that source exists.

## Methods Human-vs-Simulator Round Figure

This builds the side-by-side methods figure that pairs one real human round
with one matched simulator replay round.

Run:

```bash
bash scripts/paper/regenerate_results_assets.sh --only human_sim_round_figure
```

Manual commands:

```bash
python -m analysis.find_human_sim_overlap \
  --no-skip-timed-out-humans \
  --min-abs-human-pr-delta 0.05 \
  --min-abs-sim-pr-delta 0.05 \
  --output-json analysis/data/round_pair_candidates_skip_timed.json

python -m analysis.render_human_sim_round_figure \
  --matches-json analysis/data/round_pair_candidates_skip_timed.json \
  --match-index 0 \
  --output-html analysis/figures/round_human_vs_simulator.html
```

Optional PDF export (Chrome headless):

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
ABS_HTML="$(pwd)/analysis/figures/round_human_vs_simulator.html"
"$CHROME" --headless=new --disable-gpu \
  --no-pdf-header-footer --run-all-compositor-stages-before-draw \
  --virtual-time-budget=10000 \
  --print-to-pdf="$(pwd)/../persuasiontrace-overleaf/figures/round_human_vs_simulator.pdf" \
  "file://$ABS_HTML"
```
