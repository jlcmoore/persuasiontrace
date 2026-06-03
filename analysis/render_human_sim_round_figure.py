"""Render a compact side-by-side human/simulator round figure as one-page HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from string import Template
from typing import Any

from analysis.human_sim_round_common import (
    belief_bin,
    extract_belief_node_names_from_bn,
    iter_round_trace_snapshots,
    parse_jsonl_dict_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_HTML = (
    REPO_ROOT / "analysis" / "figures" / "round_human_vs_simulator.html"
)
FIGURE_TEMPLATE_PATH = (
    REPO_ROOT / "analysis" / "templates" / "human_sim_round_figure.html"
)
FIGURE_CSS_PATH = REPO_ROOT / "analysis" / "templates" / "human_sim_round_figure.css"


def parse_source_spec(source_spec: str) -> tuple[Path, int]:
    """Parse source spec like '/path/file.jsonl:123'."""
    path_str, line_str = source_spec.rsplit(":", 1)
    return Path(path_str), int(line_str)


def _float(value: Any) -> float | None:
    """Convert value to finite float if possible."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _clean_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _display_node_id(node_id: str) -> str:
    """Format node ids for compact figure display."""
    if node_id == "Target":
        return "prop."
    match = re.fullmatch(r"Belief_(\d+)", node_id)
    if match:
        return f"b_{match.group(1)}"
    return node_id


def _node_map(payload: Any) -> dict[str, float]:
    """Normalize node-belief map from payload."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.startswith("Belief_"):
            continue
        parsed = _float(value)
        if parsed is None:
            continue
        if 0.0 <= parsed <= 1.0:
            out[key] = parsed
    return dict(sorted(out.items()))


def _node_labels_from_trace(trace_payload: dict[str, Any]) -> dict[str, str]:
    """Build Belief_* to human-readable node text from BN trace metadata."""
    bn_payload = trace_payload.get("bn")
    if not isinstance(bn_payload, dict):
        return {}
    node_to_text = bn_payload.get("node_to_text")
    if not isinstance(node_to_text, dict):
        return {}

    out: dict[str, str] = {}
    for key, value in node_to_text.items():
        if not isinstance(key, str) or not key.startswith("Belief_"):
            continue
        if not isinstance(value, str):
            continue
        text = _clean_whitespace(value)
        if text:
            out[key] = text
    return out


def node_probs_from_trace(trace_payload: dict[str, Any]) -> dict[str, float]:
    """Compute initial node marginals from simulator distribution history."""
    history = trace_payload.get("distribution_history")
    bn_payload = trace_payload.get("bn")
    if not isinstance(history, list) or not history:
        return {}
    if not isinstance(history[0], list):
        return {}
    if not isinstance(bn_payload, dict):
        return {}

    initial_distribution = history[0]
    node_names = extract_belief_node_names_from_bn(bn_payload, initial_distribution)
    if not node_names:
        return {}

    probs = {node: 0.0 for node in node_names}
    for entry in initial_distribution:
        if not isinstance(entry, dict):
            continue
        state = entry.get("state")
        prob = _float(entry.get("probability"))
        if not isinstance(state, dict) or prob is None:
            continue
        for node in node_names:
            if state.get(node) is True:
                probs[node] += prob
    return dict(sorted(probs.items()))


def _count_persuader_turns(messages: list[dict[str, Any]]) -> int:
    """Count persuader turns in the message list."""
    return sum(1 for message in messages if message.get("role") == "persuader")


def load_human_round(match: dict[str, Any]) -> dict[str, Any]:
    """Load full human round record from match metadata."""
    source = match.get("human_source")
    if not isinstance(source, str):
        raise ValueError("Match is missing human_source.")

    source_path, source_line = parse_source_spec(source)
    proposition = match.get("proposition")
    init_target = _float(match.get("human_initial"))
    stance = match.get("stance_supports_proposition")

    records = parse_jsonl_dict_records(source_path)
    line_records = [record for line, record in records if line == source_line]
    candidates = line_records if line_records else [record for _, record in records]

    for record in candidates:
        if proposition and record.get("proposition") != proposition:
            continue
        init = _float(record.get("target_initial_belief"))
        if (
            init_target is not None
            and init is not None
            and abs(init - init_target) > 1e-6
        ):
            continue
        rec_stance = record.get("persuader_supports_proposition")
        if (
            isinstance(stance, bool)
            and isinstance(rec_stance, bool)
            and rec_stance != stance
        ):
            continue
        return record

    raise ValueError(f"Could not resolve human round from {source!r}.")


def _pick_best_snapshot(
    step_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick snapshot with richest trace payload."""
    best_score = -1
    best_round: dict[str, Any] | None = None
    best_trace: dict[str, Any] | None = None

    for round_payload, trace_payload in iter_round_trace_snapshots(step_rows):
        atom_hist = trace_payload.get("atom_history")
        belief_hist = trace_payload.get("belief_history")
        atom_len = len(atom_hist) if isinstance(atom_hist, list) else 0
        belief_len = len(belief_hist) if isinstance(belief_hist, list) else 0
        score = atom_len * 100 + belief_len
        if score <= best_score:
            continue
        best_score = score
        best_round = round_payload
        best_trace = trace_payload

    if best_round is None or best_trace is None:
        raise ValueError("No usable simulator snapshot payload found.")
    return best_round, best_trace


def load_sim_round(
    *, match: dict[str, Any], episodes_path: Path, steps_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load simulator episode row, aligned step rows, round snapshot, and trace."""
    episode_id = match.get("sim_episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("Match is missing sim_episode_id.")

    episode_records = [record for _, record in parse_jsonl_dict_records(episodes_path)]
    episode_row = None
    for rec in episode_records:
        if rec.get("episode_id") == episode_id:
            episode_row = rec
            break
    if episode_row is None:
        raise ValueError(f"Simulator episode_id {episode_id!r} not found.")

    step_rows = [
        rec
        for _, rec in parse_jsonl_dict_records(steps_path)
        if rec.get("episode_id") == episode_id
    ]
    if not step_rows:
        raise ValueError(f"No simulator steps found for episode_id {episode_id!r}.")

    round_payload, trace_payload = _pick_best_snapshot(step_rows)
    return episode_row, step_rows, round_payload, trace_payload


def _fmt(value: float | None) -> str:
    """Format optional numeric value."""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _fmt_delta(value: float | None) -> str:
    """Format optional delta value."""
    if value is None:
        return "N/A"
    return f"{value:+.3f}"


def _node_display_name(node_id: str, node_label_map: dict[str, str]) -> str:
    """Return display name for a node without exposing raw variable ids."""
    label = node_label_map.get(node_id)
    if label:
        return label
    suffix = node_id.split("_", maxsplit=1)[-1]
    return f"Belief node {suffix}"


def _wrap_text_lines(text: str, *, max_chars: int, max_lines: int) -> list[str]:
    """Wrap text into compact line chunks with a trailing ellipsis when clipped."""
    cleaned = _clean_whitespace(text)
    if not cleaned:
        return [""]

    words = cleaned.split(" ")
    lines: list[str] = []
    current = ""
    consumed_words = 0
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            consumed_words += 1
            continue
        if current:
            lines.append(current)
            current = word
            consumed_words += 1
        else:
            lines.append(word[:max_chars])
            current = word[max_chars:]
            consumed_words += 1
        if len(lines) >= max_lines:
            break

    if len(lines) < max_lines and current:
        lines.append(current[:max_chars])

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    clipped = consumed_words < len(words) or len(" ".join(lines)) < len(cleaned)
    if clipped and lines:
        last = lines[-1]
        if len(last) >= max_chars:
            lines[-1] = last[: max_chars - 3] + "..."
        else:
            lines[-1] = last + "..."
    return lines


def _wrap_text_lines_no_truncation(text: str, *, max_chars: int) -> list[str]:
    """Wrap text to a target width without dropping content."""
    cleaned = _clean_whitespace(text)
    if not cleaned:
        return [""]

    words = cleaned.split(" ")
    lines: list[str] = []
    current = ""
    for raw_word in words:
        word = raw_word
        if not current and len(word) > max_chars:
            while len(word) > max_chars:
                lines.append(word[:max_chars])
                word = word[max_chars:]
            current = word
            continue

        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        while len(word) > max_chars:
            lines.append(word[:max_chars])
            word = word[max_chars:]
        current = word

    if current:
        lines.append(current)
    return lines


def _edge_node_id(raw_value: Any, *, belief_count: int) -> str | None:
    """Map integer edge ids to canonical belief/target ids."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return "Target"
    if 1 <= value <= belief_count:
        return f"Belief_{value}"
    return None


def _graph_edges_from_bn(
    *, bn_payload: dict[str, Any], node_ids: list[str]
) -> list[tuple[str, str, int, bool]]:
    """Build directed signed edges from a BN payload, or fallback edges."""
    node_set = set(node_ids) | {"Target"}
    edges: list[tuple[str, str, int, bool]] = []
    seen: set[tuple[str, str]] = set()

    belief_nodes = bn_payload.get("belief_nodes")
    belief_count = (
        len(belief_nodes) if isinstance(belief_nodes, list) else len(node_ids)
    )
    raw_edges = bn_payload.get("edges")
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            source = _edge_node_id(edge.get("from"), belief_count=belief_count)
            target = _edge_node_id(edge.get("to"), belief_count=belief_count)
            if source is None or target is None or source == target:
                continue
            if source not in node_set or target not in node_set:
                continue
            key = (source, target)
            if key in seen:
                continue
            sign_raw = edge.get("positive_influence")
            sign = 1 if sign_raw is True else -1 if sign_raw is False else 0
            edges.append((source, target, sign, True))
            seen.add(key)

    if edges:
        return edges

    for node_id in node_ids:
        key = (node_id, "Target")
        if key in seen:
            continue
        edges.append((node_id, "Target", 0, False))
        seen.add(key)
    return edges


def _graph_levels(
    *, node_ids: list[str], edges: list[tuple[str, str, int, bool]]
) -> dict[str, int]:
    """Assign each node a level by shortest directed distance to target."""
    predecessors: dict[str, list[str]] = {}
    for source, target, _, _ in edges:
        predecessors.setdefault(target, []).append(source)

    levels: dict[str, int] = {"Target": 0}
    queue: list[str] = ["Target"]
    while queue:
        current = queue.pop(0)
        base = levels[current]
        for predecessor in predecessors.get(current, []):
            candidate = base + 1
            known = levels.get(predecessor)
            if known is not None and known <= candidate:
                continue
            levels[predecessor] = candidate
            queue.append(predecessor)

    max_level = max(levels.values())
    for node_id in node_ids:
        if node_id not in levels:
            levels[node_id] = max_level + 1
    return levels


def _render_related_beliefs(
    *,
    human_nodes: dict[str, float],
    sim_nodes: dict[str, float],
    node_label_map: dict[str, str],
    bn_payload: dict[str, Any],
    proposition: str,
    proposition_meta: str,
) -> str:
    """Render compact left-to-right BN graph using HTML cards and overlay edges."""
    node_ids = sorted(set(human_nodes.keys()) | set(sim_nodes.keys()))
    if not node_ids:
        return "<div class='nodes-empty'>No initial node beliefs available.</div>"

    edges = _graph_edges_from_bn(bn_payload=bn_payload, node_ids=node_ids)
    levels = _graph_levels(node_ids=node_ids, edges=edges)
    level_values = sorted({levels[node_id] for node_id in node_ids}, reverse=True)
    columns = [
        sorted(
            [node_id for node_id in node_ids if levels[node_id] == level],
            key=lambda value: int(value.split("_", maxsplit=1)[1]),
        )
        for level in level_values
    ]

    graph_width = 468.0
    graph_padding_x = 10.0
    graph_padding_y = 4.0
    column_gap = 14.0
    row_gap = 3.0
    cluster_label_band = 13.0
    target_gap = 18.0
    target_width_ratio_vs_left_cluster = 2.0 / 3.0
    min_belief_width = 86.0
    max_belief_width = 126.0
    title_row_height = 8.2
    node_text_line_height = 5.4
    bin_row_height = 5.4
    node_vertical_padding = 2.0
    belief_cluster_pad_x = 8.0

    belief_column_count = max(1, len(columns))
    available_width = graph_width - (2.0 * graph_padding_x) - target_gap
    left_cluster_extra_width = 2.0 * belief_cluster_pad_x
    belief_columns_total_target_width = (
        available_width - target_width_ratio_vs_left_cluster * left_cluster_extra_width
    ) / (1.0 + target_width_ratio_vs_left_cluster)
    belief_width = (
        belief_columns_total_target_width - max(0, belief_column_count - 1) * column_gap
    ) / belief_column_count
    belief_width = min(max_belief_width, max(min_belief_width, belief_width))
    belief_text_max_chars = max(24, int((belief_width - 8.0) / 3.05))

    node_texts: dict[str, str] = {}
    node_heights: dict[str, float] = {}
    for node_id in node_ids:
        full_text = _node_display_name(node_id, node_label_map)
        node_texts[node_id] = full_text
        estimated_lines = len(
            _wrap_text_lines_no_truncation(full_text, max_chars=belief_text_max_chars)
        )
        node_heights[node_id] = (
            node_vertical_padding
            + title_row_height
            + estimated_lines * node_text_line_height
            + bin_row_height
            + node_vertical_padding
        )

    max_column_height = 0.0
    for column in columns:
        if not column:
            continue
        col_height = (
            sum(node_heights[node_id] for node_id in column)
            + max(0, len(column) - 1) * row_gap
        )
        max_column_height = max(max_column_height, col_height)

    graph_height = 2.0 * graph_padding_y + cluster_label_band + max_column_height
    belief_columns_total_width = (
        belief_column_count * belief_width
        + max(0, belief_column_count - 1) * column_gap
    )
    target_x = graph_padding_x + belief_columns_total_width + target_gap
    target_width = max(90.0, graph_width - graph_padding_x - target_x)
    target_height = graph_height - 2.0 * graph_padding_y
    target_y = graph_padding_y

    node_boxes: dict[str, tuple[float, float, float, float]] = {
        "Target": (target_x, target_y, target_width, target_height)
    }
    for col_idx, column in enumerate(columns):
        if not column:
            continue
        column_height = (
            sum(node_heights[node_id] for node_id in column)
            + max(0, len(column) - 1) * row_gap
        )
        node_x = graph_padding_x + col_idx * (belief_width + column_gap)
        node_y = (
            graph_padding_y
            + cluster_label_band
            + (max_column_height - column_height) / 2.0
        )
        for node_id in column:
            height = node_heights[node_id]
            node_boxes[node_id] = (node_x, node_y, belief_width, height)
            node_y += height + row_gap

    min_belief_x = min(node_boxes[node_id][0] for node_id in node_ids)
    min_belief_y = min(node_boxes[node_id][1] for node_id in node_ids)
    max_belief_x = max(
        node_boxes[node_id][0] + node_boxes[node_id][2] for node_id in node_ids
    )
    max_belief_y = max(
        node_boxes[node_id][1] + node_boxes[node_id][3] for node_id in node_ids
    )
    belief_cluster_pad_y = 4.0
    belief_cluster_title_band = 8.0
    belief_cluster_x = min_belief_x - belief_cluster_pad_x
    belief_cluster_y = min_belief_y - belief_cluster_pad_y - belief_cluster_title_band
    belief_cluster_width = (max_belief_x - min_belief_x) + 2.0 * belief_cluster_pad_x
    belief_cluster_height = (
        (max_belief_y - min_belief_y)
        + 2.0 * belief_cluster_pad_y
        + belief_cluster_title_band
    )

    incoming_by_target: dict[str, list[str]] = {}
    for source_id, target_id, _, _ in edges:
        incoming_by_target.setdefault(target_id, []).append(source_id)
    edge_rank: dict[tuple[str, str], tuple[int, int]] = {}
    for target_id, source_ids in incoming_by_target.items():
        ordered_sources = sorted(source_ids)
        total = len(ordered_sources)
        for index, source_id in enumerate(ordered_sources):
            edge_rank[(source_id, target_id)] = (index, total)

    edge_parts: list[str] = []
    edge_label_parts: list[str] = []
    for source_id, target_id, sign, _ in edges:
        source_x, source_y, source_w, source_h = node_boxes[source_id]
        target_x_box, target_y_box, _, target_h = node_boxes[target_id]
        rank, total = edge_rank.get((source_id, target_id), (0, 1))
        fan_offset = (rank - (total - 1) / 2.0) * 5.0
        start_x = source_x + source_w
        start_y = source_y + source_h / 2.0 + fan_offset
        end_x = target_x_box
        end_y = target_y_box + target_h / 2.0 + fan_offset
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        length = max(4.0, math.hypot(delta_x, delta_y) - 6.0)
        angle = math.degrees(math.atan2(delta_y, delta_x))
        edge_class = (
            "graph-edge-positive"
            if sign > 0
            else "graph-edge-negative" if sign < 0 else "graph-edge-neutral"
        )
        edge_parts.append(
            "<div "
            f"class='graph-edge-line {edge_class}' "
            f"style='left:{start_x:.1f}px; top:{start_y:.1f}px; width:{length:.1f}px; "
            f"transform:rotate({angle:.1f}deg);'></div>"
        )
        if sign != 0:
            symbol = "+" if sign > 0 else "-"
            sign_class = "edge-sign-positive" if sign > 0 else "edge-sign-negative"
            mid_x = (start_x + end_x) / 2.0
            mid_y = (start_y + end_y) / 2.0
            edge_label_parts.append(
                "<div "
                f"class='edge-sign {sign_class}' "
                f"style='left:{mid_x - 2.2:.1f}px; top:{mid_y - 8.0:.1f}px;'>"
                f"{symbol}</div>"
            )

    proposition_main_max_chars = max(18, int((target_width - 16.0) / 5.6))
    proposition_meta_max_chars = max(30, int((target_width - 14.0) / 3.4))
    proposition_lines = _wrap_text_lines_no_truncation(
        proposition, max_chars=proposition_main_max_chars
    )
    proposition_meta_lines = _wrap_text_lines_no_truncation(
        proposition_meta, max_chars=proposition_meta_max_chars
    )
    prop_parts = [
        "<div "
        "class='graph-node graph-node-prop' "
        f"style='left:{target_x:.1f}px; top:{target_y:.1f}px; "
        f"width:{target_width:.1f}px; height:{target_height:.1f}px;'>",
        "<div class='graph-prop-id graph-inline-highlight-muted'>proposition</div>",
    ]
    for line in proposition_lines:
        prop_parts.append(
            f"<div class='graph-prop-main graph-inline-highlight'>{html.escape(line)}</div>"
        )
    if proposition_meta_lines:
        prop_parts.append("<div class='graph-prop-gap'></div>")
    for line in proposition_meta_lines:
        prop_parts.append(
            f"<div class='graph-prop-meta graph-inline-highlight-muted'>"
            f"{html.escape(line)}</div>"
        )
    prop_parts.append("</div>")

    belief_parts: list[str] = []
    for node_id in node_ids:
        human_value = human_nodes.get(node_id)
        sim_value = sim_nodes.get(node_id)
        node_x, node_y, node_width, node_height = node_boxes[node_id]
        bin_source = (
            (human_value + sim_value) / 2.0
            if human_value is not None and sim_value is not None
            else human_value if human_value is not None else sim_value
        )
        bin_value = belief_bin(bin_source) if bin_source is not None else "n/a"
        belief_parts.append(
            "<div "
            "class='graph-node graph-node-belief' "
            f"style='left:{node_x:.1f}px; top:{node_y:.1f}px; "
            f"width:{node_width:.1f}px; height:{node_height:.1f}px;'>"
            "<div class='graph-node-head'>"
            f"<span class='graph-node-id graph-inline-highlight-muted'>"
            f"{html.escape(_display_node_id(node_id))}</span>"
            f"<span class='graph-node-values graph-inline-highlight-muted'>"
            f"Human={_fmt(human_value)} | Sim={_fmt(sim_value)}</span>"
            "</div>"
            + f"<div class='graph-node-text-full'>{html.escape(node_texts[node_id])}</div>"
            + f"<div class='graph-node-bin'>{html.escape(bin_value)}</div>"
            "</div>"
        )

    graph_html = (
        "<div class='related-graph-wrap'>"
        "<div class='related-graph-html' "
        f"style='width:{graph_width:.1f}px; height:{graph_height:.1f}px;'>"
        "<div class='graph-beliefs-cluster' "
        f"style='left:{belief_cluster_x:.1f}px; top:{belief_cluster_y:.1f}px; "
        f"width:{belief_cluster_width:.1f}px; height:{belief_cluster_height:.1f}px;'>"
        "<div class='graph-cluster-title-pill'>related beliefs</div>"
        "</div>"
        + "".join(edge_parts)
        + "".join(edge_label_parts)
        + "".join(belief_parts)
        + "".join(prop_parts)
        + "</div>"
        + "</div>"
    )
    return graph_html


def _extract_rhetorical_modes(atom: dict[str, Any]) -> dict[str, float]:
    """Extract rhetorical mode vector from an atom."""
    raw = atom.get("rhetorical_modes")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("logos", "ethos", "pathos"):
        parsed = _float(raw.get(key))
        if parsed is not None:
            out[key] = parsed
    return out


def _format_rhetorical_vector(vector: dict[str, float]) -> str:
    """Format rhetorical vector in a compact fixed-order string."""
    if not vector:
        return "N/A"
    return " ".join(
        [
            f"L={vector.get('logos', 0.0):.2f}",
            f"E={vector.get('ethos', 0.0):.2f}",
            f"P={vector.get('pathos', 0.0):.2f}",
        ]
    )


def _format_belief_targets(atom: dict[str, Any]) -> str:
    """Format targeted belief ids from an atom payload."""
    raw_targets = atom.get("belief_targets")
    if not isinstance(raw_targets, list):
        return "N/A"
    ids: list[str] = []
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        belief_id = item.get("belief_id")
        if isinstance(belief_id, str) and belief_id:
            display_id = _display_node_id(belief_id)
            if display_id not in ids:
                ids.append(display_id)
    if not ids:
        return "N/A"
    return ", ".join(ids)


def _build_atom_records(
    atom_list: Any, *, step_index: int, max_items: int = 3
) -> list[dict[str, Any]]:
    """Build normalized atom records for one simulator step."""
    if not isinstance(atom_list, list):
        return []
    records: list[dict[str, Any]] = []
    for atom_index, atom in enumerate(atom_list[:max_items], start=1):
        if not isinstance(atom, dict):
            continue
        text_span = _clean_whitespace(str(atom.get("text_span") or ""))
        if not text_span:
            continue
        p_support = _float(atom.get("p_support"))
        rhet_vector = _extract_rhetorical_modes(atom)
        target_ids = _format_belief_targets(atom)
        records.append(
            {
                "atom_index": atom_index,
                "text_span": text_span,
                "p_support": p_support,
                "rhetorical_vector": rhet_vector,
                "target_ids": target_ids,
                "dom_id": f"atom-s{step_index + 1}-a{atom_index}",
            }
        )
    return records


def _find_non_overlapping_span(
    text: str, needle: str, occupied: list[tuple[int, int]]
) -> tuple[int, int] | None:
    """Find first non-overlapping case-insensitive span of needle in text."""
    text_lower = text.lower()
    needle_lower = needle.lower()
    start = 0
    while True:
        pos = text_lower.find(needle_lower, start)
        if pos < 0:
            return None
        end = pos + len(needle)
        overlaps = False
        for left, right in occupied:
            if not (end <= left or pos >= right):
                overlaps = True
                break
        if not overlaps:
            return pos, end
        start = pos + 1


def _highlight_text_with_atoms(text: str, atom_records: list[dict[str, Any]]) -> str:
    """Highlight atom spans in message text and link them with stable ids."""
    occupied: list[tuple[int, int]] = []
    located: list[tuple[int, int, dict[str, Any]]] = []
    for record in atom_records:
        location = _find_non_overlapping_span(text, record["text_span"], occupied)
        if location is None:
            continue
        start, end = location
        occupied.append((start, end))
        located.append((start, end, record))
    located.sort(key=lambda item: item[0])

    if not located:
        return html.escape(text)

    out_parts: list[str] = []
    cursor = 0
    for start, end, record in located:
        if start > cursor:
            out_parts.append(html.escape(text[cursor:start]))
        chunk = text[start:end]
        atom_class = f"atom-{record['atom_index']}"
        p_support = record.get("p_support")
        p_text = f"{p_support:.2f}" if isinstance(p_support, float) else "N/A"
        rhet_text = _format_rhetorical_vector(record.get("rhetorical_vector") or {})
        targets_text = str(record.get("target_ids") or "N/A")
        title = f"support={p_text}; targets={targets_text}; rhetorical={rhet_text}"
        out_parts.append(
            "<span "
            f"id=\"{record['dom_id']}\" "
            f'class="atom-highlight {atom_class}" '
            f'title="{html.escape(title)}">'
            f"{html.escape(chunk)}"
            "</span>"
        )
        cursor = end
    if cursor < len(text):
        out_parts.append(html.escape(text[cursor:]))
    return "".join(out_parts)


def _simulator_rhetorical_vector(
    *, step_rows: list[dict[str, Any]], trace_payload: dict[str, Any]
) -> dict[str, float]:
    """Aggregate mean rhetorical vector over shown simulator atoms."""
    atom_history = trace_payload.get("atom_history")
    atom_history_list = atom_history if isinstance(atom_history, list) else []
    sums = {"logos": 0.0, "ethos": 0.0, "pathos": 0.0}
    count = 0
    for row in step_rows:
        idx = int(row.get("step_index", 0))
        if idx >= len(atom_history_list):
            continue
        atoms = atom_history_list[idx]
        if not isinstance(atoms, list):
            continue
        for atom in atoms:
            if not isinstance(atom, dict):
                continue
            vec = _extract_rhetorical_modes(atom)
            if not vec:
                continue
            for key in ("logos", "ethos", "pathos"):
                sums[key] += vec.get(key, 0.0)
            count += 1
    if count == 0:
        return {}
    return {key: value / count for key, value in sums.items()}


def _render_human_conversation(
    *,
    messages: list[dict[str, Any]],
    serial_questions: list[Any],
    initial: float | None,
) -> str:
    """Render interleaved human conversation with per-turn deltas."""
    rows: list[str] = []
    serial_values = [_float(value) for value in serial_questions]
    serial_index = 0
    previous = initial
    pending_delta: float | None = None

    for message in messages:
        role = str(message.get("role") or "")
        content = _clean_whitespace(str(message.get("content") or ""))

        if role == "persuader":
            rows.append(
                "<div data-v-4f62d463 class='message message-left'>"
                f"<span data-v-4f62d463>{html.escape(content)}</span>"
                "</div>"
            )
            if serial_index < len(serial_values):
                current = serial_values[serial_index]
                if current is not None and previous is not None:
                    pending_delta = current - previous
                    previous = current
                serial_index += 1

        elif role == "target":
            rows.append(
                "<div data-v-4f62d463 class='message message-right'>"
                f"<span data-v-4f62d463>{html.escape(content)}</span>"
                "</div>"
            )
            if pending_delta is not None:
                rows.append(
                    "<div class='delta-line'>"
                    f"delta: {_fmt_delta(pending_delta)}"
                    "</div>"
                )
                pending_delta = None

    if pending_delta is not None:
        rows.append(
            "<div class='delta-line'>" f"delta: {_fmt_delta(pending_delta)}" "</div>"
        )

    if not rows:
        return "<div class='delta-line'>No messages available.</div>"
    return "\n".join(rows)


def _render_sim_conversation(
    *,
    step_rows: list[dict[str, Any]],
    trace_payload: dict[str, Any],
    show_rhetorical_dimensions: bool,
) -> tuple[str, float | None]:
    """Render simulator conversation with linked atom badges and highlights."""
    rows: list[str] = []
    atom_history = trace_payload.get("atom_history")
    atom_history_list = atom_history if isinstance(atom_history, list) else []

    ordered = sorted(step_rows, key=lambda row: int(row.get("step_index", 0)))
    shown_final: float | None = None

    for row in ordered:
        idx = int(row.get("step_index", 0))
        persuader_text = _clean_whitespace(str(row.get("persuader_text") or ""))
        target_text = _clean_whitespace(str(row.get("target_text") or ""))
        before = _float(row.get("belief_before"))
        after = _float(row.get("belief_after"))
        shown_final = after if after is not None else shown_final

        atom_tags_raw = atom_history_list[idx] if idx < len(atom_history_list) else []
        atom_records = _build_atom_records(atom_tags_raw, step_index=idx)
        atom_badges_html = ""
        if atom_records:
            badge_parts: list[str] = []
            for record in atom_records:
                atom_class = f"atom-{record['atom_index']}"
                p_support = record.get("p_support")
                p_text = f"{p_support:.2f}" if isinstance(p_support, float) else "N/A"
                rhet_text = _format_rhetorical_vector(
                    record.get("rhetorical_vector") or {}
                )
                targets_text = str(record.get("target_ids") or "N/A")
                support_prefix = "+"
                if isinstance(p_support, float):
                    compact_prob = f"{p_support:.2f}"
                    if compact_prob.startswith("0"):
                        compact_prob = compact_prob[1:]
                    support_prefix = f"+{compact_prob}"
                support_text = (
                    f"{support_prefix} {targets_text}"
                    if targets_text != "N/A"
                    else support_prefix
                )
                rhet_line = ""
                if show_rhetorical_dimensions and rhet_text != "N/A":
                    vec = record.get("rhetorical_vector") or {}
                    rhet_line = (
                        "logos "
                        f"{vec.get('logos', 0.0):.2f} | "
                        "pathos "
                        f"{vec.get('pathos', 0.0):.2f} | "
                        "ethos "
                        f"{vec.get('ethos', 0.0):.2f}"
                    )
                title = (
                    f"support={p_text}; targets={targets_text}; rhetorical={rhet_text}"
                )
                badge_parts.append(
                    f"<a href=\"#{record['dom_id']}\" class=\"atom-chip {atom_class}\" "
                    f'title="{html.escape(title)}">'
                    f"<span class='chip-support'>{html.escape(support_text)}</span>"
                    + (
                        f"<span class='chip-rhet'>{html.escape(rhet_line)}</span>"
                        if rhet_line
                        else ""
                    )
                    + "</a>"
                )
            atom_badges_html = (
                "<div class='atom-row'>"
                "<span class='atom-arrow'>-&gt;</span>"
                + "".join(badge_parts)
                + "</div>"
            )

        highlighted_persuader = _highlight_text_with_atoms(persuader_text, atom_records)

        rows.append(
            f"{atom_badges_html}"
            "<div data-v-4f62d463 class='message message-left sim-persuader'>"
            f"<span data-v-4f62d463>{highlighted_persuader}</span>"
            "</div>"
        )

        rows.append(
            "<div data-v-4f62d463 class='message message-right'>"
            f"<span data-v-4f62d463>{html.escape(target_text)}</span>"
            "</div>"
        )

        delta = after - before if before is not None and after is not None else None
        rows.append("<div class='delta-line'>" f"delta: {_fmt_delta(delta)}" "</div>")

    if not rows:
        return (
            "<div class='delta-line'>No simulator steps available.</div>",
            shown_final,
        )
    return "\n".join(rows), shown_final


def _write_css_for_output(output_html: Path) -> str:
    """Write renderer CSS next to output HTML and return relative href."""
    css_name = f"{output_html.stem}.css"
    css_output = output_html.with_name(css_name)
    css_text = FIGURE_CSS_PATH.read_text(encoding="utf-8")
    css_output.write_text(css_text, encoding="utf-8")
    return css_name


def _render_from_template(values: dict[str, str]) -> str:
    """Render figure HTML from a standalone template file."""
    template_text = FIGURE_TEMPLATE_PATH.read_text(encoding="utf-8")
    return Template(template_text).substitute(values)


def render_html(
    *,
    match: dict[str, Any],
    human_round: dict[str, Any],
    sim_episode: dict[str, Any],
    sim_steps: list[dict[str, Any]],
    sim_round_snapshot: dict[str, Any],
    sim_trace: dict[str, Any],
    output_html: Path,
    show_rhetorical_dimensions: bool,
) -> None:
    """Render and write side-by-side compact HTML."""
    proposition = str(match.get("proposition") or human_round.get("proposition") or "")
    stance_supports = bool(match.get("stance_supports_proposition"))
    stance_label = "supports" if stance_supports else "opposes"

    human_initial = _float(human_round.get("target_initial_belief"))
    human_final = _float(human_round.get("target_final_belief"))
    sim_initial = _float(sim_episode.get("target_initial_belief"))

    human_nodes = _node_map(human_round.get("target_initial_node_beliefs"))
    sim_nodes = _node_map(sim_round_snapshot.get("target_initial_node_beliefs"))
    if not sim_nodes:
        sim_nodes = node_probs_from_trace(sim_trace)

    node_label_map = _node_labels_from_trace(sim_trace)
    policy_model = str(sim_episode.get("policy_model") or "N/A")
    stance_summary = "supports prop." if stance_supports else "opposes prop."
    proposition_meta = f"persuader={policy_model} | {stance_summary}"
    human_bn_payload = human_round.get("bayesian_network")
    related_beliefs_html = _render_related_beliefs(
        human_nodes=human_nodes,
        sim_nodes=sim_nodes,
        node_label_map=node_label_map,
        bn_payload=human_bn_payload if isinstance(human_bn_payload, dict) else {},
        proposition=proposition,
        proposition_meta=proposition_meta,
    )

    human_messages_raw = human_round.get("messages")
    human_messages = human_messages_raw if isinstance(human_messages_raw, list) else []
    human_serial_raw = human_round.get("serial_questions")
    human_serial = human_serial_raw if isinstance(human_serial_raw, list) else []

    human_persuader_turns = _count_persuader_turns(human_messages)
    ordered_sim_steps = sorted(sim_steps, key=lambda row: int(row.get("step_index", 0)))
    sim_steps_cut = ordered_sim_steps[:human_persuader_turns]

    human_body = _render_human_conversation(
        messages=human_messages,
        serial_questions=human_serial,
        initial=human_initial,
    )
    sim_body, sim_shown_final = _render_sim_conversation(
        step_rows=sim_steps_cut,
        trace_payload=sim_trace,
        show_rhetorical_dimensions=show_rhetorical_dimensions,
    )
    sim_persona_text = str(sim_episode.get("persona") or "N/A")

    human_total_delta = (
        (human_final - human_initial)
        if human_initial is not None and human_final is not None
        else None
    )
    sim_final_for_display = (
        sim_shown_final
        if sim_shown_final is not None
        else _float(sim_episode.get("target_final_belief"))
    )
    sim_total_delta = (
        (sim_final_for_display - sim_initial)
        if sim_initial is not None and sim_final_for_display is not None
        else None
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    css_href = _write_css_for_output(output_html)
    html_text = _render_from_template(
        {
            "css_href": css_href,
            "proposition": html.escape(proposition),
            "proposition_meta": html.escape(proposition_meta),
            "stance_label": html.escape(stance_label),
            "policy_model": html.escape(policy_model),
            "related_beliefs_html": related_beliefs_html,
            "human_initial": _fmt(human_initial),
            "human_final": _fmt(human_final),
            "human_body": human_body,
            "human_total_delta": _fmt_delta(human_total_delta),
            "sim_initial": _fmt(sim_initial),
            "sim_final_for_display": _fmt(sim_final_for_display),
            "sim_persona_text": html.escape(sim_persona_text),
            "sim_body": sim_body,
            "sim_total_delta": _fmt_delta(sim_total_delta),
        }
    )
    output_html.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matches-json",
        type=Path,
        required=True,
        help="Path to JSON list output from find_human_sim_overlap.py.",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        default=0,
        help="0-based index into matches JSON list.",
    )
    parser.add_argument(
        "--sim-episodes",
        type=Path,
        default=Path("results/rl_baseline/sim_compare/episodes.jsonl"),
        help="Simulator episodes JSONL path.",
    )
    parser.add_argument(
        "--sim-steps",
        type=Path,
        default=Path("results/rl_baseline/sim_compare/steps.jsonl"),
        help="Simulator steps JSONL path.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=DEFAULT_OUTPUT_HTML,
        help="Output HTML path.",
    )
    parser.add_argument(
        "--show-rhetorical-dimensions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to show per-atom rhetorical vectors in atom badges.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    matches_payload = json.loads(args.matches_json.read_text(encoding="utf-8"))
    if not isinstance(matches_payload, list) or not matches_payload:
        raise ValueError("matches-json is empty or invalid.")
    if args.match_index < 0 or args.match_index >= len(matches_payload):
        raise ValueError("match-index out of range.")

    match = matches_payload[args.match_index]
    if not isinstance(match, dict):
        raise ValueError("Selected match row is not an object.")

    human_round = load_human_round(match)
    sim_episode, sim_steps, sim_round_snapshot, sim_trace = load_sim_round(
        match=match,
        episodes_path=args.sim_episodes,
        steps_path=args.sim_steps,
    )

    render_html(
        match=match,
        human_round=human_round,
        sim_episode=sim_episode,
        sim_steps=sim_steps,
        sim_round_snapshot=sim_round_snapshot,
        sim_trace=sim_trace,
        output_html=args.output_html,
        show_rhetorical_dimensions=bool(args.show_rhetorical_dimensions),
    )
    print(f"Wrote HTML figure to {args.output_html}")


if __name__ == "__main__":
    main()
