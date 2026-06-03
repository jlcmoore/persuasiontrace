#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN_DEFAULT="${REPO_ROOT}/env-continuouspersuasion/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"
MPLCONFIGDIR_DEFAULT="${REPO_ROOT}/.matplotlib"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${MPLCONFIGDIR_DEFAULT}}"
mkdir -p "${MPLCONFIGDIR}"
STANCE_BIAS_EPISODES_JSONL_DEFAULT="${REPO_ROOT}/results/stance_mirror_dryrun_3sim_allprops_gpt5_fixed/episodes.jsonl"
STANCE_BIAS_EPISODES_JSONL="${STANCE_BIAS_EPISODES_JSONL:-${STANCE_BIAS_EPISODES_JSONL_DEFAULT}}"

ALL_STEPS=(
  persuasiveness
  human_clusters
  rhetoric_regression
  counterfactual_replay
  llm_judge
  stance_bias
  model_sweep_policy_rank
  human_sim_round_figure
  export_assets
)

ONLY_STEPS_CSV=""
SKIP_EXPORT=0
REFRESH_LLM_JUDGE=0

usage() {
  cat <<'EOF'
Regenerate paper Results/Appendix assets for the manuscript.

Usage:
  bash scripts/paper/regenerate_results_assets.sh [--only step1,step2] [--skip-export] [--refresh-llm-judge]

Steps:
  persuasiveness
  human_clusters
  rhetoric_regression
  counterfactual_replay
  llm_judge
  stance_bias
  model_sweep_policy_rank
  human_sim_round_figure
  export_assets

Examples:
  bash scripts/paper/regenerate_results_assets.sh
  bash scripts/paper/regenerate_results_assets.sh --only human_clusters,counterfactual_replay
  bash scripts/paper/regenerate_results_assets.sh --only stance_bias --skip-export
  bash scripts/paper/regenerate_results_assets.sh --only model_sweep_policy_rank
  bash scripts/paper/regenerate_results_assets.sh --only human_sim_round_figure
  bash scripts/paper/regenerate_results_assets.sh --only llm_judge --refresh-llm-judge

Notes:
  - llm_judge is cache-first: by default it consumes existing
    analysis/data/rl_human_match_sim_compare_simulator_llm_judge_summary.csv
    and renders the bar plot only. Pass --refresh-llm-judge to recompute
    summary data with live LLM judge calls.
  - stance_bias reads mirrored for-vs-against rollout episodes from
    STANCE_BIAS_EPISODES_JSONL (default:
    results/stance_mirror_dryrun_3sim_allprops_gpt5_fixed/episodes.jsonl),
    exports paired stance-bias summaries, and renders the stance-bias figure.
  - counterfactual_replay reads cached replay round-errors and regenerates only
    paper-used figures (target/node errors, human LOO, and core average-error
    panels); it does not run replay model calls.
  - paper-facing counterfactual figure mapping uses the strict conditional
    core average-error ranking (no CI bars) as the main results figure.
  - model_sweep_policy_rank is analysis-only (reads existing episodes JSONL and
    writes plot/table assets); expensive rollout commands are included as
    comments in that step and are not executed.
  - human_sim_round_figure regenerates overlap matches and the
    round_human_vs_simulator HTML, then renders PDF via Chrome/Chromium when
    available.
  - export_assets also regenerates the methods round-pair figure in
    ../continuouspersuasion-overleaf/figures when that repo exists.
    Override paths with OVERLEAF_REPO and CHROME_BIN if needed.
  - human_sim_round_figure auto-prefers replay artifacts at
    analysis/data/simulator_counterfactual_replay_social_bn_* and filters
    candidates to full BN simulator corpora by default
    (ROUND_PAIR_SIM_CORPUS_PREFIX=full_simulated_target__).
    It also requires human-first replay mode and exact initialization by default
    (ROUND_PAIR_REQUIRED_TURN_MODE=human_first_then_policy and
    ROUND_PAIR_REQUIRED_INITIALIZATION_MODE=exact). Override with
    ROUND_PAIR_REPLAY_PREFIX, ROUND_PAIR_SIM_CORPUS_PREFIX,
    ROUND_PAIR_REQUIRED_TURN_MODE, ROUND_PAIR_REQUIRED_INITIALIZATION_MODE, or
    force explicit files via ROUND_PAIR_SIM_EPISODES and ROUND_PAIR_SIM_STEPS.
  - replay generation now defaults to --persuader-turn-mode policy
    (matched initial beliefs only; no forced human persuader transcript).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY_STEPS_CSV="${2:-}"
      if [[ -z "${ONLY_STEPS_CSV}" ]]; then
        echo "error: --only requires a comma-separated step list" >&2
        exit 2
      fi
      shift 2
      ;;
    --skip-export)
      SKIP_EXPORT=1
      shift
      ;;
    --refresh-llm-judge)
      REFRESH_LLM_JUDGE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

ONLY_STEPS=()
if [[ -n "${ONLY_STEPS_CSV}" ]]; then
  IFS=',' read -r -a raw_steps <<< "${ONLY_STEPS_CSV}"
  for raw_step in "${raw_steps[@]}"; do
    step_name="$(echo "${raw_step}" | xargs)"
    if [[ -z "${step_name}" ]]; then
      continue
    fi
    ONLY_STEPS+=("${step_name}")
  done
  for candidate in "${ONLY_STEPS[@]}"; do
    is_known=0
    for known in "${ALL_STEPS[@]}"; do
      if [[ "${candidate}" == "${known}" ]]; then
        is_known=1
        break
      fi
    done
    if [[ "${is_known}" -eq 0 ]]; then
      echo "error: unknown step in --only: ${candidate}" >&2
      echo "known steps: ${ALL_STEPS[*]}" >&2
      exit 2
    fi
  done
fi

wants_step() {
  local step_name="$1"
  if [[ -z "${ONLY_STEPS_CSV}" ]]; then
    return 0
  fi
  for selected in "${ONLY_STEPS[@]}"; do
    if [[ "${selected}" == "${step_name}" ]]; then
      return 0
    fi
  done
  return 1
}

run_module() {
  MPLBACKEND=Agg "${PYTHON_BIN}" -m "$@"
}

resolve_chrome_bin() {
  if [[ -n "${CHROME_BIN:-}" && -x "${CHROME_BIN}" ]]; then
    echo "${CHROME_BIN}"
    return 0
  fi

  local candidate=""
  for candidate in \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "$(command -v chromium 2>/dev/null || true)" \
    "$(command -v chromium-browser 2>/dev/null || true)" \
    "$(command -v google-chrome 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

run_step_banner() {
  local step_name="$1"
  echo
  echo "== ${step_name} =="
}

run_persuasiveness() {
  run_step_banner "persuasiveness"
  run_module analysis.persuasiveness_bucket_summary \
    --min 2026-02-01 \
    --drop-empty \
    --bucket "control:continuous_measure=serial-questions,control_dialogue=true,human_persuader=false,human_target=true,llm_persuader=gpt-5-2025-08-07,use_audio=false,participant_proposition=false,enable_node_belief_survey=false,proposition_source=debategpt,minimum_turns=4,turn_limit=4,factual_domain=false" \
    --bucket "standard_text:continuous_measure=serial-questions,control_dialogue=false,human_target=true,use_audio=false,participant_proposition=false,enable_node_belief_survey=false,llm_persuader=gpt-5-2025-08-07,proposition_source=debategpt,minimum_turns=4,turn_limit=4,factual_domain=false" \
    --bucket "personal_text:continuous_measure=serial-questions,control_dialogue=false,human_target=true,use_audio=false,participant_proposition=true" \
    --bucket "audio:continuous_measure=serial-questions,control_dialogue=false,human_persuader=false,human_target=true,llm_persuader=gpt-5-2025-08-07,use_audio=true,show_transcript=true,participant_proposition=false,enable_node_belief_survey=false,proposition_source=debategpt,minimum_turns=4,turn_limit=4,factual_domain=false" \
    --output-csv analysis/data/persuasiveness_bucket_summary.csv \
    --output-pdf analysis/figures/persuasiveness_bucket_summary.pdf

  run_module analysis.persuasiveness_welch_vs_control \
    --min 2026-02-01 \
    --output-csv analysis/data/persuasiveness_welch_vs_control.csv

  local overleaf_repo="${OVERLEAF_REPO:-${REPO_ROOT}/../continuouspersuasion-overleaf}"
  local overleaf_data_dir="${overleaf_repo}/data"
  if [[ -d "${overleaf_repo}" ]]; then
    mkdir -p "${overleaf_data_dir}"
    cp -f analysis/data/persuasiveness_welch_vs_control.csv \
      "${overleaf_data_dir}/persuasiveness_welch_vs_control.csv"
  fi
}

run_human_clusters() {
  run_step_banner "human_clusters"
  run_module simulation.scripts.build_human_trajectory_cluster_model \
    --results-dir results \
    --human-source llm-human-target \
    --persuader-model gpt-5-2025-08-07 \
    --turn-limit 4 \
    --participant-proposition false \
    --require-bn-survey \
    --condition-substring "cm=serial-questions" \
    --feature-set trajectory \
    --k 2 \
    --normalize-trajectories \
    --normalize-init-bin-distribution \
    --output-model analysis/data/human_trajectory_cluster_model_k2_v1.json \
    --output-summary-csv analysis/data/human_clusters_k2_summary.csv \
    --output-by-init-bin-csv analysis/data/human_clusters_k2_by_init_bin.csv \
    --output-by-init-bin-heatmap analysis/data/human_clusters_k2_by_init_bin_heatmap.pdf \
    --output-cluster-shapes analysis/data/human_clusters_k2_shapes.pdf \
    --output-pca-scatter analysis/data/human_clusters_k2_pca_scatter.pdf \
    --pca-fig-width 3.0 \
    --pca-fig-height 2.0 \
    --pca-hide-title
}

run_rhetoric_regression() {
  run_step_banner "rhetoric_regression"
  local input_csv="analysis/data/annotation_regression_llm_human_nocontrol_noaudio_ppt_false.csv"
  local salvi_summary_csv="analysis/data/salvi_rhetoric_ordinal_summary.csv"
  local rhetoric_annotation_csv="annotations/rhetoric_standard_current.jsonl"

  if [[ -f "${rhetoric_annotation_csv}" ]]; then
    run_module analysis.annotation_regression \
      --annotations "${rhetoric_annotation_csv}" \
      --continuous-measure serial-questions \
      --control-dialogue false \
      --human-persuader false \
      --human-target true \
      --llm-persuader gpt-5-2025-08-07 \
      --use-audio false \
      --participant-proposition false \
      --enable-node-belief-survey false \
      --proposition-source debategpt \
      --minimum-turns 4 \
      --turn-limit 4 \
      --factual-domain false \
      --output-csv "${input_csv}"
  elif [[ ! -f "${input_csv}" ]]; then
    echo "error: missing rhetoric annotations and missing ${input_csv}." >&2
    echo "error: expected ${rhetoric_annotation_csv} to rebuild input." >&2
    exit 1
  fi

  local overlay_args=()
  if [[ -f "${salvi_summary_csv}" ]]; then
    overlay_args=(
      --overlay-summary-csv "${salvi_summary_csv}"
      --overlay-label "Salvi"
    )
  else
    echo "warning: missing ${salvi_summary_csv}; plotting Ours-only coefficients." >&2
  fi

  run_module analysis.rhetoric_llm_nonppt_test_plot \
    --input-csv "${input_csv}" \
    --output-csv analysis/data/results_rhetoric_llm_nonppt_coefficients.csv \
    --output-pdf analysis/figures/results_rhetoric_llm_nonppt_coefficients.pdf \
    --fig-width 3.0 \
    --fig-height 2.0 \
    --cov-type none \
    --ci-level 0.95 \
    "${overlay_args[@]}"
}

run_counterfactual_replay() {
  run_step_banner "counterfactual_replay"
  local prefix="analysis/data/simulator_counterfactual_replay_social_bn"
  local round_errors_csv="${prefix}_round_errors.jsonl"

  if [[ ! -f "${round_errors_csv}" ]]; then
    echo "warning: missing ${round_errors_csv}; skipped counterfactual replay plots." >&2
    echo "warning: run analysis.simulator_counterfactual_replay first to generate round errors." >&2
    return 0
  fi

  MPLBACKEND=Agg "${PYTHON_BIN}" - <<PY
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.simulator_counterfactual_replay_plot import (
    NODE_DELTA_ERROR_COLUMN,
    NODE_ERROR_COLUMN,
    ROUND_KEY_COLUMNS,
    TARGET_ERROR_COLUMN,
    _bn_bin_lookup_from_round_errors,
    _exclude_no_rhetoric_corpora,
    _human_loo_rows,
    _human_loo_summary_row,
    _human_round_features_from_lookup,
    _load_round_errors_dataframe,
    append_human_reference_rows,
    append_persona_average_rows,
    build_conditional_core_avg_error_bin_sensitivity_tables,
    build_conditional_core_avg_error_tables,
    build_corpus_summary_with_ci,
    save_conditional_core_avg_error_plot,
    save_human_loo_plot,
    save_metric_plot,
)

prefix = Path("${prefix}")
round_errors_path = Path("${round_errors_csv}")
n_bootstrap = 4000
seed = 17

round_errors = _load_round_errors_dataframe(round_errors_path)
round_errors = _exclude_no_rhetoric_corpora(round_errors)
required_columns = {
    "corpus",
    TARGET_ERROR_COLUMN,
    "serial_trajectory_mae",
    NODE_ERROR_COLUMN,
    NODE_DELTA_ERROR_COLUMN,
    *ROUND_KEY_COLUMNS,
}
missing_columns = sorted(required_columns.difference(round_errors.columns))
if missing_columns:
    raise ValueError(f"Round-error CSV missing columns: {missing_columns}")

round_errors = append_persona_average_rows(round_errors)
round_errors = append_human_reference_rows(round_errors)
lookup = _bn_bin_lookup_from_round_errors(
    round_errors,
    include_proposition=True,
    scheme="fixed3",
    quantile=(1.0 / 3.0),
)
if lookup.empty:
    raise ValueError("BN bin lookup is empty; cannot regenerate paper counterfactual plots.")

round_errors = round_errors.merge(lookup, on=ROUND_KEY_COLUMNS, how="inner")
if round_errors.empty:
    raise ValueError("Round-error join with BN lookup produced zero rows.")

summary_with_ci = build_corpus_summary_with_ci(
    round_errors,
    n_bootstrap=n_bootstrap,
    rng=np.random.default_rng(seed),
)
if summary_with_ci.empty:
    raise ValueError("No corpus rows found while rebuilding counterfactual summary.")

save_metric_plot(
    summary_with_ci,
    metric_column="mean_final_target_abs_error",
    ci_low_column="target_ci_low",
    ci_high_column="target_ci_high",
    title="Final target error (lower is better)",
    xlabel="mean final_target_abs_error",
    color="#1D4ED8",
    ecolor="#93C5FD",
    output_path=Path(f"{prefix}_target_error_plot.png"),
)
save_metric_plot(
    summary_with_ci,
    metric_column="mean_final_node_mae",
    ci_low_column="node_ci_low",
    ci_high_column="node_ci_high",
    title="Final node error (lower is better)",
    xlabel="mean final_node_mae",
    color="#0F766E",
    ecolor="#99F6E4",
    output_path=Path(f"{prefix}_node_error_plot.png"),
)

features = _human_round_features_from_lookup(lookup)
if features:
    unconditional_rows = _human_loo_rows(features, conditional_on_bin=False)
    conditional_rows = _human_loo_rows(features, conditional_on_bin=True)
    loo_summary = pd.DataFrame(
        [
            _human_loo_summary_row("unconditional", unconditional_rows),
            _human_loo_summary_row("conditional", conditional_rows),
        ]
    )
    if int(loo_summary["n_evals"].fillna(0).sum()) > 0:
        save_human_loo_plot(loo_summary, Path(f"{prefix}_human_loo_plot.png"))

core_summary, core_pairwise = build_conditional_core_avg_error_tables(
    round_errors,
    lookup,
    n_bootstrap=n_bootstrap,
    rng=np.random.default_rng(seed),
)
if not core_summary.empty:
    core_summary.to_csv(
        Path(f"{prefix}_conditional_core_avg_error_all_bins_summary.csv"),
        index=False,
    )
    if not core_pairwise.empty:
        core_pairwise.to_csv(
            Path(f"{prefix}_conditional_core_avg_error_all_bins_pairwise.csv"),
            index=False,
        )

core_loo_summary, core_loo_pairwise = build_conditional_core_avg_error_tables(
    round_errors,
    lookup,
    n_bootstrap=n_bootstrap,
    rng=np.random.default_rng(seed),
    restrict_to_human_loo_bins=True,
)
if not core_loo_summary.empty:
    core_loo_summary.to_csv(
        Path(f"{prefix}_conditional_core_avg_error_summary.csv"),
        index=False,
    )
    core_loo_summary.to_csv(
        Path(f"{prefix}_conditional_core_avg_error_loo_bins_summary.csv"),
        index=False,
    )
    if not core_loo_pairwise.empty:
        core_loo_pairwise.to_csv(
            Path(f"{prefix}_conditional_core_avg_error_pairwise.csv"),
            index=False,
        )
        core_loo_pairwise.to_csv(
            Path(f"{prefix}_conditional_core_avg_error_loo_bins_pairwise.csv"),
            index=False,
        )
    save_conditional_core_avg_error_plot(
        core_loo_summary,
        Path(f"{prefix}_conditional_core_avg_error_plot.png"),
    )
    save_conditional_core_avg_error_plot(
        core_loo_summary,
        Path(f"{prefix}_conditional_core_avg_error_loo_bins_plot.png"),
    )
    save_conditional_core_avg_error_plot(
        core_loo_summary,
        Path(f"{prefix}_conditional_score_divergence_plot.png"),
    )

core_sensitivity_summary, core_sensitivity_pairwise = (
    build_conditional_core_avg_error_bin_sensitivity_tables(
        round_errors,
        lookup,
        n_bootstrap=n_bootstrap,
        rng=np.random.default_rng(seed),
    )
)
if not core_sensitivity_summary.empty:
    core_sensitivity_summary.to_csv(
        Path(f"{prefix}_conditional_core_avg_error_bin_sensitivity_summary.csv"),
        index=False,
    )
if not core_sensitivity_pairwise.empty:
    core_sensitivity_pairwise.to_csv(
        Path(f"{prefix}_conditional_core_avg_error_bin_sensitivity_pairwise.csv"),
        index=False,
    )
PY
}

run_llm_judge() {
  run_step_banner "llm_judge"
  local prefix="analysis/data/rl_human_match_sim_compare_simulator_llm_judge"
  local summary_csv="${prefix}_summary.csv"
  local output_plot="${prefix}_bar.pdf"
  local judge_model="${LLM_JUDGE_MODEL:-openai/gpt-5.4-2026-03-17}"

  if [[ "${REFRESH_LLM_JUDGE}" -eq 1 ]]; then
    if ! run_module analysis.simulator_llm_judge \
      --min-date 2026-01-01 \
      --human-source llm-human-target \
      --persuader-model gpt-5-2025-08-07 \
      --turn-limit 4 \
      --participant-proposition false \
      --exclude-bn-survey \
      --proposition-match none \
      --max-rounds-per-corpus 50 \
      --judge-model "${judge_model}" \
      --plot-format pdf \
      --output-prefix "${prefix}"; then
      echo "warning: simulator_llm_judge command failed; continuing with existing summary CSV if present." >&2
    fi
  else
    echo "llm_judge: skipping live judge recomputation (cache-first mode)." >&2
  fi

  if [[ -f "${summary_csv}" ]]; then
    run_module analysis.simulator_llm_judge_plot \
      --summary-csv "${summary_csv}" \
      --output-plot "${output_plot}"
  else
    echo "warning: missing ${summary_csv}; skipped simulator_llm_judge_plot." >&2
    if [[ "${REFRESH_LLM_JUDGE}" -eq 0 ]]; then
      echo "warning: rerun with --refresh-llm-judge to generate it." >&2
    fi
  fi
}

run_stance_bias() {
  run_step_banner "stance_bias"
  local prefix="analysis/data/rl_human_match_sim_compare"
  local proposition_by_policy_csv="${prefix}_proposition_stance_deltas_by_policy.csv"

  if [[ ! -f "${STANCE_BIAS_EPISODES_JSONL}" ]]; then
    echo "error: STANCE_BIAS_EPISODES_JSONL does not exist: ${STANCE_BIAS_EPISODES_JSONL}" >&2
    exit 1
  fi

  run_module analysis.simulator_stance_bias_export \
    --episodes-jsonl "${STANCE_BIAS_EPISODES_JSONL}" \
    --policy-model gpt-5-2025-08-07 \
    --output-prefix "${prefix}"

  run_module analysis.simulator_stance_bias_plot \
    --input-csv "${prefix}_stance_bias_summary.csv" \
    --output-pdf analysis/figures/results_stance_bias.pdf

  if [[ -f "${proposition_by_policy_csv}" ]]; then
    run_module analysis.simulator_naive_penalty_plot \
      --input-csv "${proposition_by_policy_csv}" \
      --non-naive-policy-model gpt-5-2025-08-07 \
      --naive-policy-model naive \
      --min-rounds-per-cell 1 \
      --output-csv analysis/data/rl_human_match_sim_compare_naive_penalty.csv \
      --output-pdf analysis/figures/results_naive_penalty.pdf
  else
    echo "warning: missing ${proposition_by_policy_csv}; skipped naive-penalty refresh." >&2
  fi
}

run_model_sweep_policy_rank() {
  run_step_banner "model_sweep_policy_rank"

  # Expensive rollout commands to regenerate input episodes (intentionally
  # disabled in this refresh script):
  #
  # "${PYTHON_BIN}" -m rl.baseline_runner \
  #   --config configs/rl_model_sweep_debategpt_bn.yml \
  #   --split test \
  #   --output-dir results/rl_model_sweep_debategpt_bn
  # Model sweep now includes naive directly in the main sweep rollout, so no
  # supplemental episodes input is needed.

  run_module analysis.model_sweep_policy_rank_plot \
    --episodes-jsonl results/rl_model_sweep_debategpt_bn/episodes.jsonl \
    --output-csv analysis/data/model_sweep_policy_rank_by_simulator.csv \
    --output-pdf analysis/figures/model_sweep_policy_rank_by_simulator.pdf
}

run_human_sim_round_figure_impl() {
  local overleaf_repo="${OVERLEAF_REPO:-${REPO_ROOT}/../continuouspersuasion-overleaf}"
  local overleaf_figures="${overleaf_repo}/figures"
  local matches_json="analysis/data/round_pair_candidates_skip_timed.json"
  local figure_html="analysis/figures/round_human_vs_simulator.html"
  local figure_pdf="${overleaf_figures}/round_human_vs_simulator.pdf"
  local sim_episodes="${ROUND_PAIR_SIM_EPISODES:-}"
  local sim_steps="${ROUND_PAIR_SIM_STEPS:-}"
  local replay_prefix_default="${REPO_ROOT}/analysis/data/simulator_counterfactual_replay_social_bn"
  local replay_prefix="${ROUND_PAIR_REPLAY_PREFIX:-${replay_prefix_default}}"
  local replay_episodes="${replay_prefix}_episodes.jsonl"
  local replay_steps="${replay_prefix}_steps.jsonl"
  local sim_corpus_prefix="${ROUND_PAIR_SIM_CORPUS_PREFIX:-full_simulated_target__}"
  local required_turn_mode="${ROUND_PAIR_REQUIRED_TURN_MODE:-human_first_then_policy}"
  local required_initialization_mode="${ROUND_PAIR_REQUIRED_INITIALIZATION_MODE:-exact}"
  local require_node_bin_match="${ROUND_PAIR_REQUIRE_NODE_BIN_MATCH:-0}"
  local match_index_override="${ROUND_PAIR_MATCH_INDEX:-}"
  local match_index=""
  local node_match_arg="--no-require-node-bin-match"
  local chrome_bin=""

  if [[ -z "${sim_episodes}" || -z "${sim_steps}" ]]; then
    local exactprop_root="${REPO_ROOT}/results/sim_compare_exactprop_smoke_2026-05-03"
    local exactprop_episodes="${exactprop_root}/episodes.jsonl"
    local exactprop_steps="${exactprop_root}/steps.jsonl"
    local baseline_root="${REPO_ROOT}/results/rl_baseline/sim_compare"
    local baseline_episodes="${baseline_root}/episodes.jsonl"
    local baseline_steps="${baseline_root}/steps.jsonl"

    if [[ -f "${replay_episodes}" && -f "${replay_steps}" ]]; then
      sim_episodes="${replay_episodes}"
      sim_steps="${replay_steps}"
    elif [[ -f "${exactprop_episodes}" && -f "${exactprop_steps}" ]]; then
      sim_episodes="${exactprop_episodes}"
      sim_steps="${exactprop_steps}"
    else
      sim_episodes="${baseline_episodes}"
      sim_steps="${baseline_steps}"
    fi
  fi

  if [[ "${require_node_bin_match}" == "1" ]]; then
    node_match_arg="--require-node-bin-match"
  fi

  if [[ ! -d "${overleaf_figures}" ]]; then
    echo "warning: missing overleaf figures dir: ${overleaf_figures}" >&2
    echo "warning: skipped round_human_vs_simulator export." >&2
    return 0
  fi

  run_module analysis.find_human_sim_overlap \
    --sim-episodes "${sim_episodes}" \
    --sim-steps "${sim_steps}" \
    --sim-corpus-prefix "${sim_corpus_prefix}" \
    --required-persuader-turn-mode "${required_turn_mode}" \
    --required-initialization-mode "${required_initialization_mode}" \
    ${node_match_arg} \
    --no-skip-timed-out-humans \
    --min-abs-human-pr-delta 0.05 \
    --min-abs-sim-pr-delta 0.05 \
    --output-json "${matches_json}"

  if [[ -n "${match_index_override}" ]]; then
    match_index="${match_index_override}"
  else
    match_index="$("${PYTHON_BIN}" - <<PY
import json
import re
from pathlib import Path

matches_path = Path("${matches_json}")
rows = json.loads(matches_path.read_text(encoding="utf-8"))
if not isinstance(rows, list) or not rows:
    raise ValueError("round-pair matches JSON is empty; cannot select match-index.")

def _source_date(source: str) -> str:
    if not isinstance(source, str):
        return ""
    match = re.search(r"(\\d{4}-\\d{2}-\\d{2})\\.jsonl", source)
    if match is None:
        return ""
    return match.group(1)

latest_date = max(_source_date(str(row.get("human_source", ""))) for row in rows)
candidate_indices = [
    index
    for index, row in enumerate(rows)
    if _source_date(str(row.get("human_source", ""))) == latest_date
]
if not candidate_indices:
    candidate_indices = list(range(len(rows)))

def _score(index: int) -> tuple[float, float]:
    row = rows[index]
    return (
        float(row.get("initial_abs_diff", 1e9)),
        -float(row.get("movement_score", 0.0)),
    )

print(min(candidate_indices, key=_score))
PY
)"
  fi

  echo "round_human_vs_simulator: sim_episodes=${sim_episodes}" >&2
  echo "round_human_vs_simulator: sim_steps=${sim_steps}" >&2
  echo "round_human_vs_simulator: sim_corpus_prefix=${sim_corpus_prefix}" >&2
  echo "round_human_vs_simulator: required_turn_mode=${required_turn_mode}" >&2
  echo "round_human_vs_simulator: required_initialization_mode=${required_initialization_mode}" >&2
  echo "round_human_vs_simulator: node_bin_match=${require_node_bin_match}" >&2
  echo "round_human_vs_simulator: match_index=${match_index}" >&2

  run_module analysis.render_human_sim_round_figure \
    --matches-json "${matches_json}" \
    --match-index "${match_index}" \
    --sim-episodes "${sim_episodes}" \
    --sim-steps "${sim_steps}" \
    --output-html "${figure_html}"

  if ! chrome_bin="$(resolve_chrome_bin)"; then
    echo "warning: could not find a Chromium/Chrome binary." >&2
    echo "warning: generated HTML only: ${figure_html}" >&2
    return 0
  fi

  "${chrome_bin}" --headless=new --disable-gpu \
    --no-pdf-header-footer --run-all-compositor-stages-before-draw \
    --virtual-time-budget=10000 \
    --print-to-pdf="${figure_pdf}" \
    "file://$(pwd)/${figure_html}"
}

run_human_sim_round_figure() {
  run_step_banner "human_sim_round_figure"
  run_human_sim_round_figure_impl
}

run_export_assets() {
  run_step_banner "export_assets"
  run_module analysis.latex.export_paper_assets
  run_human_sim_round_figure_impl
}

cd "${REPO_ROOT}"

echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
if [[ -n "${ONLY_STEPS_CSV}" ]]; then
  echo "Selected steps: ${ONLY_STEPS_CSV}"
else
  echo "Selected steps: all"
fi
if [[ "${SKIP_EXPORT}" -eq 1 ]]; then
  echo "Export step: skipped"
fi
if [[ "${REFRESH_LLM_JUDGE}" -eq 1 ]]; then
  echo "LLM judge refresh: enabled"
else
  echo "LLM judge refresh: disabled (cache-first)"
fi

for step in "${ALL_STEPS[@]}"; do
  if [[ "${step}" == "export_assets" && "${SKIP_EXPORT}" -eq 1 ]]; then
    continue
  fi
  if wants_step "${step}"; then
    "run_${step}"
  fi
done

echo
echo "Done."
