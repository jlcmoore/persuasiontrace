"""Generate a cleaned DebateGPT BN sample table with tiny TikZ graphs."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from analysis.latex.latex_helpers import escape_latex_inline, load_jsonl_records

CLEANED_DEBATEGPT_BN_RELATIVE_PATH = Path(
    "src/simulation/data/fitted_bayesian_networks_debategpt.jsonl"
)
DEBATEGPT_BN_SAMPLES_TABLE_TEX = "debategpt_bn_samples_table.tex"
SAMPLE_ROW_COUNT = 6
PREFERRED_PROPOSITIONS = [
    "Social media are making people stupid.",
    "Artificial intelligence is good for society.",
    "Every citizen should receive a basic income from the government.",
    "Students should have to wear school uniforms.",
    "Space exploration is a worthwhile investment for humanity.",
    "Elected or appointed government officials should be paid the minimum wage.",
]


def _extract_cleaned_bn_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return validated BN payload from one cleaned fitted-network record.

    Args:
        record: One parsed JSONL record.

    Returns:
        Normalized Bayesian-network payload.
    """
    bn = record.get("bayesian_network")
    if not isinstance(bn, dict):
        return {"belief_nodes": [], "edges": []}
    belief_nodes = bn.get("belief_nodes")
    edges = bn.get("edges")
    if not isinstance(belief_nodes, list):
        belief_nodes = []
    if not isinstance(edges, list):
        edges = []
    return {"belief_nodes": belief_nodes, "edges": edges}


def _select_sample_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose a deterministic set of cleaned DebateGPT records for display.

    Args:
        records: Full cleaned DebateGPT fitted-network records.

    Returns:
        Ordered subset of records used in the appendix table.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        proposition_id = record.get("id")
        if isinstance(proposition_id, str):
            by_id[proposition_id] = record

    selected: list[dict[str, Any]] = []
    for proposition_id in PREFERRED_PROPOSITIONS:
        chosen = by_id.get(proposition_id)
        if chosen is not None:
            selected.append(chosen)

    if len(selected) >= SAMPLE_ROW_COUNT:
        return selected[:SAMPLE_ROW_COUNT]

    already_selected = {
        str(record.get("id"))
        for record in selected
        if isinstance(record.get("id"), str)
    }
    fallback_candidates = sorted(
        records,
        key=lambda record: str(record.get("id") or ""),
    )
    for record in fallback_candidates:
        proposition_id = record.get("id")
        if not isinstance(proposition_id, str):
            continue
        if proposition_id in already_selected:
            continue
        bn = _extract_cleaned_bn_payload(record)
        if len(bn["belief_nodes"]) < 3:
            continue
        selected.append(record)
        already_selected.add(proposition_id)
        if len(selected) >= SAMPLE_ROW_COUNT:
            break
    return selected[:SAMPLE_ROW_COUNT]


def _compute_node_positions(
    belief_node_count: int,
    edges: list[dict[str, Any]],
) -> dict[int, tuple[float, float]]:
    """Compute compact left-to-right positions for `T` and `B_i` nodes.

    Args:
        belief_node_count: Number of belief nodes.
        edges: Directed edge list using integer ``from``/``to`` IDs.

    Returns:
        Mapping from node index to ``(x, y)`` coordinates where ``0`` is target.
    """
    children = _build_children_map(belief_node_count, edges)
    layers = _build_depth_layers(belief_node_count, children)

    positions: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
    layer_spacing = 1.55
    vertical_spacing = 1.2
    for layer_depth, layer_nodes in sorted(layers.items()):
        ordered_nodes = sorted(layer_nodes)
        layer_size = len(ordered_nodes)
        for offset, node_index in enumerate(ordered_nodes):
            y_position = ((layer_size - 1) / 2.0 - offset) * vertical_spacing
            x_position = -layer_depth * layer_spacing
            positions[node_index] = (x_position, y_position)
    return positions


def _build_children_map(
    belief_node_count: int, edges: list[dict[str, Any]]
) -> dict[int, list[int]]:
    """Build adjacency mapping from source node index to target node indices."""
    children: dict[int, list[int]] = {node: [] for node in range(belief_node_count + 1)}
    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        if not isinstance(source, int) or not isinstance(target, int):
            continue
        if source < 0 or source > belief_node_count:
            continue
        if target < 0 or target > belief_node_count:
            continue
        children[source].append(target)
    return children


def _build_depth_layers(
    belief_node_count: int, children: dict[int, list[int]]
) -> dict[int, list[int]]:
    """Group belief nodes by shortest directed distance to the target node."""

    @lru_cache(maxsize=None)
    def distance_to_target(node: int) -> int:
        if node == 0:
            return 0
        valid_child_distances = [
            distance_to_target(child)
            for child in children.get(node, [])
            if child != node and distance_to_target(child) >= 0
        ]
        if not valid_child_distances:
            return 1
        return 1 + min(valid_child_distances)

    layers: dict[int, list[int]] = defaultdict(list)
    for node_index in range(1, belief_node_count + 1):
        layers[distance_to_target(node_index)].append(node_index)
    return layers


def _render_tikz_graph(bn_payload: dict[str, Any]) -> str:
    """Render one qualitative BN graph as a tiny TikZ picture.

    Args:
        bn_payload: Cleaned BN payload with ``belief_nodes`` and ``edges``.

    Returns:
        TikZ source string.
    """
    belief_nodes = bn_payload["belief_nodes"]
    edge_rows = bn_payload["edges"]
    belief_node_count = len(belief_nodes)
    positions = _compute_node_positions(belief_node_count, edge_rows)

    lines = [
        r"\begin{tikzpicture}[",
        r"x=0.58cm, y=0.58cm,",
        r">=stealth,",
        (
            r"bnnode/.style={draw=gray!70, rounded corners=1.2pt, fill=white, "
            r"inner sep=1.1pt, font=\scriptsize},"
        ),
        r"bntarget/.style={bnnode, fill=blue!10, draw=blue!60!black},",
        r"bnpos/.style={->, draw=green!60!black, line width=0.55pt},",
        r"bnneg/.style={->, draw=red!70!black, dashed, line width=0.55pt}",
        r"]",
        r"\node[bntarget] (T) at (0,0) {T};",
    ]
    for node_index in range(1, belief_node_count + 1):
        x_position, y_position = positions.get(node_index, (-1.0, 0.0))
        lines.append(
            r"\node[bnnode] (B"
            + str(node_index)
            + ") at ("
            + f"{x_position:.2f},{y_position:.2f}"
            + r") {B"
            + str(node_index)
            + r"};"
        )

    for edge in edge_rows:
        source = edge.get("from")
        target = edge.get("to")
        if not isinstance(source, int) or not isinstance(target, int):
            continue
        if source < 0 or source > belief_node_count:
            continue
        if target < 0 or target > belief_node_count:
            continue
        source_name = "T" if source == 0 else f"B{source}"
        target_name = "T" if target == 0 else f"B{target}"
        style = "bnpos" if bool(edge.get("positive_influence", True)) else "bnneg"
        lines.append(
            r"\draw[" + style + r"] (" + source_name + r") -- (" + target_name + r");"
        )
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _render_belief_legend(belief_nodes: list[Any]) -> str:
    """Render one row's belief-node statement legend.

    Args:
        belief_nodes: Ordered belief node text list.

    Returns:
        LaTeX legend content wrapped in a top-aligned parbox.
    """
    legend_lines: list[str] = []
    for index, belief_node in enumerate(belief_nodes, start=1):
        belief_text = str(belief_node) if isinstance(belief_node, str) else ""
        legend_lines.append(
            r"\textbf{B" + str(index) + r"}: " + escape_latex_inline(belief_text)
        )
    return (
        r"\parbox[t]{\linewidth}{\vspace{0pt}\raggedright "
        + r"\\".join(legend_lines)
        + r"}"
    )


def _render_debategpt_bn_samples_table_tex(repo_root: Path) -> str:
    """Render appendix table with cleaned DebateGPT BN graph samples.

    Args:
        repo_root: Repository root used to resolve source file paths.

    Returns:
        LaTeX table source for the cleaned DebateGPT BN sample appendix table.
    """
    records = load_jsonl_records(repo_root / CLEANED_DEBATEGPT_BN_RELATIVE_PATH)
    selected_records = _select_sample_records(records)

    lines = [
        "% Auto-generated by analysis/latex/export_paper_assets.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.24\textwidth}p{0.22\textwidth}p{0.48\textwidth}}",
        r"\toprule",
        r"DebateGPT proposition & Cleaned BN graph (qualitative) & Belief-node legend \\",
        r"\midrule",
    ]

    for record in selected_records:
        proposition_id = str(record.get("id") or "")
        bn_payload = _extract_cleaned_bn_payload(record)
        proposition_cell = (
            r"\parbox[t]{\linewidth}{\vspace{0pt}"
            + escape_latex_inline(proposition_id)
            + r"}"
        )
        graph_cell = (
            r"\parbox[t]{\linewidth}{\vspace{0pt}\centering"
            + "\n"
            + _render_tikz_graph(bn_payload)
            + "\n"
            + r"}"
        )
        legend_cell = _render_belief_legend(bn_payload["belief_nodes"])
        lines.append(proposition_cell + " &")
        lines.append(graph_cell + " &")
        lines.append(legend_cell + r" \\")
        lines.append(r"\midrule")

    if len(lines) >= 1 and lines[-1] == r"\midrule":
        lines.pop()

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Cleaned fitted Bayesian-network structure samples for "
                r"DebateGPT propositions (from "
                r"\texttt{fitted\_bayesian\_networks\_debategpt.jsonl}). "
                r"Arrows show qualitative influence direction only: solid green "
                r"indicates positive influence; dashed red indicates negative influence."
                r"}"
            ),
            r"\label{tab:appendix-debategpt-bn-samples}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_debategpt_bn_samples_table_tex(output_dir: Path, repo_root: Path) -> Path:
    """Write the cleaned DebateGPT BN sample table include.

    Args:
        output_dir: Destination include directory.
        repo_root: Repository root used to resolve source file paths.

    Returns:
        Absolute path to the generated table include.
    """
    include_path = output_dir / DEBATEGPT_BN_SAMPLES_TABLE_TEX
    include_path.write_text(_render_debategpt_bn_samples_table_tex(repo_root), "utf-8")
    return include_path
