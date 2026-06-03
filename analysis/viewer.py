"""
Interactive data viewer generator for serial-questions and mouse-trace graphs.

Usage
-----
python -m analysis.viewer                 # writes HTML pages under analysis/figures/viewer/
python -m analysis.viewer --min 2025-08-08  # only files on/after this date

This produces one HTML page per condition for each instrument type available:
- serial_<condition>.html  (serial-question trajectories)
- mouse_<condition>.html   (mouse-trace trajectories; segment-aligned)

Hovering or clicking a participant’s line shows that round’s full conversation
below the graph. A simple index.html links to all generated pages.

Environment note: activate the venv first
- source env-continuouspersuasion/bin/activate
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Any

import markdown

from experiment import Condition, Round, load_round_results
from experiment.cli_utils import add_min_date_arg
from experiment.condition_filters import (
    add_condition_filter_args,
    condition_matches_filters,
    filters_from_args,
)

from .plot_blocks import DEFAULT_FIG_DIR, SEG_OFFSET

FIG_DIR = DEFAULT_FIG_DIR

# Alignment constant used in analysis.analysis; keep in sync.


VIEW_DIR = FIG_DIR / "viewer"
VIEW_DIR.mkdir(exist_ok=True, parents=True)


def _safe_name(s: str) -> str:
    """Return a filesystem-safe name for a condition label."""
    return s.replace(" ", "_").replace("/", "_")


def _esc(s: str) -> str:
    """HTML-escape a string, handling None as empty string."""
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


def _round_label(idx: int, _rd: Round) -> str:
    """Compact label for a participant/round in a condition page."""
    return f"Participant {idx + 1}"


def _messages_html(rd: Round, *, include_timestamps: bool) -> str:
    """Render a round's summary (__str__) and messages as HTML blocks.

    The summary (pretty string from Round.__str__) is shown inside a <pre> to
    preserve formatting. Below it, each message is rendered as a block suitable
    for highlighting via CSS.
    """
    # Pretty summary string from Round.__str__ (includes proposition, beliefs,
    # persuader info, and formatted transcript). Render markdown to HTML here so
    # the viewer does not need a client-side markdown implementation.
    summary_md = markdown.markdown(
        str(rd),
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )

    parts: list[str] = [
        '<div class="conv">',
        f'<div class="summary-md">{summary_md}</div>',
        '<dl class="msgs">',
    ]
    recv_idx = 0
    for i, msg in enumerate(rd.messages):
        role = _esc(msg.get("role", ""))
        content_raw = msg.get("content", "") or ""

        # If this is a received (persuader) message, increment received index.
        data_rec = ""
        if role == "persuader":
            recv_idx += 1
            data_rec = f' data-rec-idx="{recv_idx}"'

        # Render content with or without word timing.
        rendered_content: list[str] = []
        dur_attr = ""
        if include_timestamps:
            transcript = rd.transcripts[i] if i < len(rd.transcripts) else None
            duration = None
            words: list[dict[str, Any]] | None = None
            if isinstance(transcript, dict):
                duration = transcript.get("duration")
                words = (
                    transcript.get("words")
                    if isinstance(transcript.get("words"), list)
                    else None
                )
            if words:
                for w in words:
                    try:
                        wstart = float(w.get("start"))
                        wend = float(w.get("end"))
                    except (TypeError, ValueError):
                        wstart, wend = None, None
                    text = _esc(str(w.get("text", w.get("word", ""))))
                    if wstart is not None and wend is not None:
                        rendered_content.append(
                            f'<span class="tok" data-wstart="{wstart}" '
                            f'data-wend="{wend}">'
                            f"{text}</span>"
                        )
                    else:
                        rendered_content.append(f'<span class="tok">{text}</span>')
            else:
                rendered_content.append(_esc(content_raw))
            if duration is not None:
                dur_attr = f' data-duration="{duration}"'
        else:
            rendered_content.append(_esc(content_raw))

        # Display numbering by 'received' order for persuader messages only
        if role == "persuader":
            dt_label = f"{recv_idx}. {role}"
        else:
            dt_label = role

        parts.append(
            f'<dt class="role">{dt_label}</dt>'
            f'<dd class="msg" data-idx="{i+1}"{data_rec}{dur_attr}>'
            f"<div class=\"content\">{''.join(rendered_content)}</div>"
            "</dd>"
        )
    if not rd.messages:
        parts.append(
            '<dt class="role">&nbsp;</dt><dd class="msg"><em>No messages exchanged.</em></dd>'
        )
    parts.append("</dl></div>")
    return "".join(parts)


def _serial_points_both(
    rd: Round,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return (raw_pts, rel_pts) including initial and final for a round's serial trace.

    Values are clamped to [0,1] for rendering safety.
    """
    serial_raw = rd.get_serial_questions(persuader_relative=False) or []
    serial_rel = rd.get_serial_questions(persuader_relative=True) or []

    init_raw = rd.target_initial_belief
    fin_raw = rd.target_final_belief
    init_rel = rd.persuader_relative_belief(init_raw)
    fin_rel = rd.persuader_relative_belief(fin_raw)

    def sanitize(v: float | None) -> float | None:
        if v is None:
            return None
        try:
            y = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(y):
            return None
        return max(0.0, min(1.0, y))

    yvals_raw = [
        sanitize(init_raw),
        *[sanitize(v) for v in serial_raw],
        sanitize(fin_raw),
    ]
    yvals_rel = [
        sanitize(init_rel),
        *[sanitize(v) for v in serial_rel],
        sanitize(fin_rel),
    ]

    def to_pts(yvals: list[float | None]) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for i, y in enumerate(yvals):
            if y is None:
                continue
            pts.append((float(i), y))
        return pts

    return to_pts(yvals_raw), to_pts(yvals_rel)


def _mouse_segments(
    rd: Round, persuader_relative: bool
) -> list[list[tuple[float, float]]]:
    """Return list of segments; each segment is list of (x, y) in segment-aligned X.

    Segment k occupies x in [k+1, k+2] (aligned with analysis.SEG_OFFSET==1).
    Within a segment, time is normalized to [0, 1] based on that segment's own span.
    """
    trace = rd.get_mouse_traces(persuader_relative=persuader_relative) or []
    segments: list[list[tuple[float, float]]] = []
    for k, seg in enumerate(trace):
        if not seg:
            segments.append([])
            continue
        try:
            seg_sorted = sorted(seg, key=lambda pt: float(pt["timestamp"]))
        except (KeyError, TypeError, ValueError):
            segments.append([])
            continue
        xs = [float(pt["timestamp"]) for pt in seg_sorted]
        ys = [float(pt["position"]) for pt in seg_sorted]
        if not xs:
            segments.append([])
            continue
        t0, t1 = xs[0], xs[-1]
        dt = t1 - t0
        if dt <= 0:
            xs_scaled = [SEG_OFFSET + k + 0.5] * len(xs)
        else:
            xs_scaled = [SEG_OFFSET + k + (t - t0) / dt for t in xs]
        segments.append(list(zip(xs_scaled, ys)))
    return segments


def _build_serial_page(
    cond: Condition,
    rounds: list[Round],
) -> str:
    """Return full HTML for a serial-question viewer page for one condition."""
    # Collect participant traces
    participants: list[dict[str, Any]] = []
    max_x = 0.0
    for idx, rd in enumerate(rounds):
        pts_raw, pts_rel = _serial_points_both(rd)
        # Default to raw points for initial render; JS toggles to relative.
        pts = pts_raw
        if not pts:
            continue
        max_x = max(max_x, pts[-1][0] if pts else 0.0)
        # Initial and final for export overlay (raw by default)
        init = rd.target_initial_belief
        fin = rd.target_final_belief

        participants.append(
            {
                "id": idx,
                "label": _round_label(idx, rd),
                "points": pts,
                "points_raw": pts_raw,
                "points_rel": pts_rel,
                "messages_html": _messages_html(rd, include_timestamps=False),
                "init": init,
                "final": fin,
            }
        )

    title = f"Serial-question trajectories — {html.escape(str(cond.as_non_id_role()))}"
    ylabel = "Support (0..1)"

    return _render_svg_page(
        title=title,
        subtitle="Hover or click a line to show conversation.",
        ylabel=ylabel,
        participants=participants,
        x_domain=(0.0, max_x if max_x > 0 else 1.0),
        y_domain=(0.0, 1.0),
        connect_across_segments=True,
    )


def _build_mouse_page(
    cond: Condition,
    rounds: list[Round],
) -> str:
    """Return full HTML for a mouse-trace viewer page for one condition."""
    participants: list[dict[str, Any]] = []
    max_x = float(SEG_OFFSET)
    for idx, rd in enumerate(rounds):
        segs_raw = _mouse_segments(rd, persuader_relative=False)
        segs_rel = _mouse_segments(rd, persuader_relative=True)
        # Default to raw segments for initial render; JS toggles to relative.
        segs = segs_raw
        if not segs:
            continue
        if segs_raw or segs_rel:
            # Use the larger of the two for domain width
            max_x = max(max_x, float(SEG_OFFSET) + max(len(segs_raw), len(segs_rel)))
        # Initial and final values for markers (raw by default)
        init = rd.target_initial_belief
        fin = rd.target_final_belief
        # Also compute persuader-relative forms for init/final so markers can toggle in JS
        init_rel_val = rd.persuader_relative_belief(init)
        fin_rel_val = rd.persuader_relative_belief(fin)
        participants.append(
            {
                "id": idx,
                "label": _round_label(idx, rd),
                "segments": segs,
                "segments_raw": segs_raw,
                "segments_rel": segs_rel,
                "messages_html": _messages_html(rd, include_timestamps=True),
                "init": init,
                "final": fin,
                "init_rel": init_rel_val,
                "final_rel": fin_rel_val,
            }
        )

    title = f"Mouse-trace trajectories — {html.escape(str(cond.as_non_id_role()))}"
    ylabel = "Support (0..1)"

    return _render_svg_page(
        title=title,
        subtitle=(
            "Hover or click any segment of a participant’s trace to show conversation."
        ),
        ylabel=ylabel,
        participants=participants,
        x_domain=(0.0, max_x),
        y_domain=(0.0, 1.0),
        connect_across_segments=False,
    )


def _render_svg_page(
    *,
    title: str,
    subtitle: str,
    ylabel: str,
    participants: list[dict[str, Any]],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    connect_across_segments: bool,
) -> str:
    """Render a self-contained HTML page with inline SVG and minimal JS.

    participants: each item must have `id`, `label`, `messages_html` and either:
      - `points`: list[(x,y)]  if connect_across_segments is True
      - `segments`: list[list[(x,y)]] otherwise
    """
    width, height = 900, 420
    margin_l, margin_r, margin_t, margin_b = 60, 20, 30, 90
    plot_w, plot_h = width - margin_l - margin_r, height - margin_t - margin_b
    x0, x1 = x_domain
    y0, y1 = y_domain

    def sx(x: float) -> float:
        if x1 == x0:
            return float(margin_l)
        return margin_l + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        if y1 == y0:
            return float(height - margin_b)
        # SVG y grows downward
        return margin_t + (1.0 - (y - y0) / (y1 - y0)) * plot_h

    # Build SVG shapes
    svg_parts: list[str] = []

    # Palette (distinct colors per participant)
    def palette_color(i: int, n: int) -> str:
        if n <= 0:
            return "#6b7280"
        # Simple HSL rainbow
        h = int((360.0 * i) / max(1, n))
        return f"hsl({h},70%,45%)"

    # Embed minimal SVG-local styles so exports keep appearance
    svg_local_css = (
        ".axis{stroke:#333;stroke-width:1}"
        ".grid{stroke:#ddd;stroke-width:1}"
        ".ytick{font-size:10px;fill:#333}"
        ".trace{fill:none;stroke-width:1.6;opacity:0.7}"
        ".trace.active{opacity:1;stroke-width:2.6}"
        ".trace.dim{opacity:0.15}"
        ".hit{fill:none;stroke:transparent;stroke-width:12;pointer-events: stroke}"
        ".pt{pointer-events:none}"
        ".pt.dim{opacity:0.15}"
    )
    # Define a clipping path for the plotting region to mimic Matplotlib's clipping
    clip_id = "clip-plot"
    svg_parts.append(
        "<defs>"
        f"<style>{svg_local_css}</style>"
        f'<clipPath id="{clip_id}">'
        f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}"/>'
        "</clipPath>"
        "</defs>"
    )

    # Axes
    # X axis line
    svg_parts.append(
        f'<line x1="{sx(x0):.2f}" y1="{sy(y0):.2f}" '
        f'x2="{sx(x1):.2f}" y2="{sy(y0):.2f}" class="axis"/>'
    )
    # Y axis line
    svg_parts.append(
        f'<line x1="{sx(x0):.2f}" y1="{sy(y0):.2f}" '
        f'x2="{sx(x0):.2f}" y2="{sy(y1):.2f}" class="axis"/>'
    )
    # Y ticks 0..1 at 0.0, 0.25, 0.5, 0.75, 1.0
    for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(y0 + f * (y1 - y0))
        svg_parts.append(
            f'<line x1="{sx(x0):.2f}" y1="{y:.2f}" '
            f'x2="{sx(x1):.2f}" y2="{y:.2f}" class="grid"/>'
        )
        svg_parts.append(
            f'<text x="{sx(x0)-6:.2f}" y="{y+4:.2f}" class="ytick" text-anchor="end">{f:.2f}</text>'
        )

    # X ticks at integer message indices
    max_x_int = int(round(x1))
    for xi in range(int(x0), max_x_int + 1):
        xx = sx(float(xi))
        # short tick
        svg_parts.append(
            f'<line x1="{xx:.2f}" y1="{sy(y0):.2f}" '
            f'x2="{xx:.2f}" y2="{sy(y0)-6:.2f}" class="axis"/>'
        )
        svg_parts.append(
            f'<text x="{xx:.2f}" y="{sy(y0)+18:.2f}" '
            f'class="ytick" text-anchor="middle">{xi}</text>'
        )

    # Axis labels
    # X label centered beneath
    svg_parts.append(
        f'<text x="{(sx(x0)+sx(x1))/2:.2f}" '
        f'y="{sy(y0)+36:.2f}" class="ytick" text-anchor="middle">'
        f"Messages received</text>"
    )
    # Y label rotated
    svg_parts.append(
        f'<text x="{sx(x0)-46:.2f}" '
        f'y="{(sy(y0)+sy(y1))/2:.2f}" '
        f'transform="rotate(-90 {sx(x0)-46:.2f} {(sy(y0)+sy(y1))/2:.2f})" '
        f'class="ytick" text-anchor="middle">{html.escape(ylabel)}</text>'
    )

    # Participant traces and markers
    n_parts = len(participants)
    for p in participants:
        pid = p["id"]
        color = palette_color(pid, n_parts)

        # circles for each polyline point will be added inline below

        if connect_across_segments:
            pts = p.get("points", [])
            if len(pts) >= 2:
                d = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for (x, y) in pts)
                # Provide raw and relative variants for the toggle
                pts_raw = p.get("points_raw") or []
                pts_rel = p.get("points_rel") or []

                d_raw = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for (x, y) in pts_raw)
                d_rel = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for (x, y) in pts_rel)

                # Hit polyline
                svg_parts.append(
                    f'<polyline class="hit" data-pid="{pid}" points="{d}"></polyline>'
                )
                # Visible trace polyline
                svg_parts.append(
                    f'<polyline class="trace" clip-path="url(#{clip_id})" '
                    f'data-pid="{pid}" points="{d}" style="stroke:{color}" '
                    f'data-raw="{d_raw}" data-rel="{d_rel}"></polyline>'
                )
                for x_pt, y_pt in pts:
                    svg_parts.append(
                        f'<circle class="pt" clip-path="url(#{clip_id})" '
                        f'data-pid="{pid}" '
                        f'cx="{sx(x_pt):.2f}" cy="{sy(y_pt):.2f}" r="2.2" '
                        f'fill="{color}" fill-opacity="0.7" />'
                    )
                # Initial square and final diamond
                x_init, y_init = pts[0]
                xf, yf = pts[-1]
                # Square (initial) – centered at point
                size = 5
                # Prepare raw/rel center y for toggle
                cy_init_raw = sy((p.get("points_raw") or [(x_init, y_init)])[0][1])
                cy_init_rel = sy((p.get("points_rel") or [(x_init, y_init)])[0][1])
                cx_init = sx(x_init)
                svg_parts.append(
                    f'<rect class="mk mk-init" data-pid="{pid}" '
                    f'data-cx="{cx_init:.2f}" '
                    f'data-cy-raw="{cy_init_raw:.2f}" '
                    f'data-cy-rel="{cy_init_rel:.2f}" '
                    f'data-size="{size}" '
                    f'x="{cx_init - size / 2:.2f}" '
                    f'y="{cy_init_raw - size / 2:.2f}" '
                    f'width="{size}" height="{size}" fill="#fff" '
                    f'stroke="{color}" stroke-width="1.2" />'
                )
                # Diamond (final) – polygon rotated
                cy_fin_raw = sy((p.get("points_raw") or [(xf, yf)])[-1][1])
                cy_fin_rel = sy((p.get("points_rel") or [(xf, yf)])[-1][1])
                cx_fin = sx(xf)
                dmd_raw = [
                    (cx_fin, cy_fin_raw - size / 2),
                    (cx_fin + size / 2, cy_fin_raw),
                    (cx_fin, cy_fin_raw + size / 2),
                    (cx_fin - size / 2, cy_fin_raw),
                ]
                dmd_s = " ".join(f"{x:.2f},{y:.2f}" for x, y in dmd_raw)
                svg_parts.append(
                    f'<polygon class="mk mk-final" data-pid="{pid}" '
                    f'data-cx="{cx_fin:.2f}" '
                    f'data-cy-raw="{cy_fin_raw:.2f}" '
                    f'data-cy-rel="{cy_fin_rel:.2f}" '
                    f'data-size="{size}" '
                    f'points="{dmd_s}" fill="{color}" stroke="#111" '
                    f'stroke-width="0.8" />'
                )
        else:
            # Mouse trace – one polyline per segment with points
            segs = p.get("segments", []) or []
            segs_raw = p.get("segments_raw", []) or []
            segs_rel = p.get("segments_rel", []) or []
            max_len = max(len(segs), len(segs_raw), len(segs_rel))
            for i in range(max_len):
                seg = segs[i] if i < len(segs) else []
                if len(seg) >= 2:
                    d = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for (x, y) in seg)
                    d_raw = " ".join(
                        f"{sx(x):.2f},{sy(y):.2f}"
                        for (x, y) in (segs_raw[i] if i < len(segs_raw) else [])
                    )
                    d_rel = " ".join(
                        f"{sx(x):.2f},{sy(y):.2f}"
                        for (x, y) in (segs_rel[i] if i < len(segs_rel) else [])
                    )
                    svg_parts.append(
                        f'<polyline class="hit" data-pid="{pid}" points="{d}"></polyline>'
                    )
                    svg_parts.append(
                        f'<polyline class="trace" clip-path="url(#{clip_id})" '
                        f'data-pid="{pid}" points="{d}" style="stroke:{color}" '
                        f'data-raw="{d_raw}" data-rel="{d_rel}"></polyline>'
                    )
                    for x_pt, y_pt in seg:
                        svg_parts.append(
                            f'<circle class="pt" clip-path="url(#{clip_id})" '
                            f'data-pid="{pid}" '
                            f'cx="{sx(x_pt):.2f}" cy="{sy(y_pt):.2f}" r="2.2" '
                            f'fill="{color}" fill-opacity="0.7" />'
                        )

            # Initial and final markers for mouse trace
            init_y = p.get("init")
            final_y = p.get("final")
            size = 6
            if init_y is not None:
                xi = float(SEG_OFFSET) - 1.0
                cx_i = sx(xi)
                cy_i_raw = sy(init_y)
                # Prefer precomputed persuader-relative values if present
                init_rel_val = p.get("init_rel")
                cy_i_rel = sy(init_rel_val) if init_rel_val is not None else cy_i_raw
                svg_parts.append(
                    f'<rect class="mk mk-init" data-pid="{pid}" '
                    f'data-cx="{cx_i:.2f}" '
                    f'data-cy-raw="{cy_i_raw:.2f}" '
                    f'data-cy-rel="{cy_i_rel:.2f}" '
                    f'data-size="{size}" '
                    f'x="{cx_i - size / 2:.2f}" '
                    f'y="{cy_i_raw - size / 2:.2f}" '
                    f'width="{size}" height="{size}" fill="#fff" '
                    f'stroke="{color}" stroke-width="1.2" />'
                )
            if final_y is not None:
                xf = float(SEG_OFFSET) + float(len(segs))
                cx_f = sx(xf)
                cy_f_raw = sy(final_y)
                fin_rel_val = p.get("final_rel")
                cy_f_rel = sy(fin_rel_val) if fin_rel_val is not None else cy_f_raw
                const_dmd = [
                    (cx_f, cy_f_raw - size / 2),
                    (cx_f + size / 2, cy_f_raw),
                    (cx_f, cy_f_raw + size / 2),
                    (cx_f - size / 2, cy_f_raw),
                ]
                dmd_s2 = " ".join(f"{x:.2f},{y:.2f}" for x, y in const_dmd)
                svg_parts.append(
                    f'<polygon class="mk mk-final" data-pid="{pid}" '
                    f'data-cx="{cx_f:.2f}" '
                    f'data-cy-raw="{cy_f_raw:.2f}" '
                    f'data-cy-rel="{cy_f_rel:.2f}" '
                    f'data-size="{size}" '
                    f'points="{dmd_s2}" fill="{color}" stroke="#111" '
                    f'stroke-width="0.8" />'
                )

    # For mouse trace, draw vertical boundaries at integer bands
    if not connect_across_segments:
        k = int(x1) + 2
        for i in range(int(x0), k):
            x = sx(float(i))
            svg_parts.append(
                f'<line x1="{x:.2f}" y1="{sy(y0):.2f}" '
                f'x2="{x:.2f}" y2="{sy(y1):.2f}" class="grid"/>'
            )

    # Build templates for conversations
    templates: list[str] = []
    for p in participants:
        pid = p["id"]
        templates.append(f"<template id=\"conv-{pid}\">{p['messages_html']}</template>")

    # Render full HTML (external CSS/JS; keep SVG styling inline for exports)
    html_parts: list[str] = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        '<link rel="stylesheet" href="../../static/viewer.css">',
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="sub">{html.escape(subtitle)} — ' f"Y: {html.escape(ylabel)}</p>",
        '<div id="chartrow" style="display:flex;gap:12px;align-items:flex-start">',
        (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'data-x0="{x0}" data-x1="{x1}" data-y0="{y0}" data-y1="{y1}" '
            f'data-ml="{margin_l}" data-mr="{margin_r}" data-mt="{margin_t}" data-mb="{margin_b}" '
            f'data-w="{plot_w}" data-h="{plot_h}" '
            f'data-connect="{str(connect_across_segments).lower()}" '
            f'data-rel="false"'
            f' data-offset="{SEG_OFFSET}">'
        ),
        *svg_parts,
        "</svg>",
        '<div id="legend" style="min-width:160px"></div>',
        "</div>",
        '<div id="toolbar" style="display:flex;gap:8px;align-items:center;margin:6px 0">',
        '  <a id="btn-back" href="index.html">Back to index</a>',
        '  <label style="font-size:12px;margin-left:8px">'
        '    <input type="checkbox" id="toggle-rel"> '
        "    Relative to persuader"
        "  </label>",
        '  <span style="flex:1 1 auto"></span>',
        '  <button id="btn-reset">Reset selection</button>',
        '  <button id="btn-svg">Export SVG</button>',
        '  <button id="btn-png">Export PNG</button>',
        '  <button id="btn-pdf">Export PDF</button>',
        "</div>",
        '<div id="panel">',
        '  <div id="who">Hover a line to see that conversation.</div>',
        '  <div id="conv"></div>',
        '  <hr class="sep">',
        '  <div id="md"></div>',
        "</div>",
        *templates,
        '<script src="../../static/viewer.js"></script>',
    ]
    return "\n".join(html_parts)


def _write(path: Path, content: str) -> None:
    """Write text to a file, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _generate_pages(
    min_date: str | None, condition_filters: dict[str, object]
) -> list[Path]:
    """Generate HTML pages and return their paths."""
    cond_to_rounds = load_round_results(min_date)
    created: list[Path] = []

    for cond, list_of_round_lists in cond_to_rounds.items():
        if condition_filters and not condition_matches_filters(cond, condition_filters):
            continue
        # list_of_round_lists: list[list[Round]]; each inner list belongs to a participant
        # (If needed, filter to completed rounds here.)

        # Serial pages per condition
        serial_rounds: list[Round] = []
        mouse_rounds: list[Round] = []
        for round_list in list_of_round_lists:
            for rd in round_list:
                if rd.serial_questions:
                    serial_rounds.append(rd)
                if rd.mouse_traces:
                    mouse_rounds.append(rd)

        if serial_rounds:
            html_page = _build_serial_page(cond, serial_rounds)
            name = f"serial_{_safe_name(str(cond.as_non_id_role()))}.html"
            out = VIEW_DIR / name
            _write(out, html_page)
            created.append(out)

        if mouse_rounds:
            html_page = _build_mouse_page(cond, mouse_rounds)
            name = f"mouse_{_safe_name(str(cond.as_non_id_role()))}.html"
            out = VIEW_DIR / name
            _write(out, html_page)
            created.append(out)

    # Index
    links = [f'<li><a href="{p.name}">{html.escape(p.name)}</a></li>' for p in created]
    index = "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            "<title>Analysis Viewer</title>",
            "<h1>Analysis Viewer</h1>",
            "<p>Self-contained pages generated under analysis/figures/viewer/.</p>",
            "<ul>",
            *links,
            "</ul>",
        ]
    )
    _write(VIEW_DIR / "index.html", index)
    # External CSS/JS are linked relatively from analysis/static to avoid duplication.
    return created


def main() -> None:
    """CLI entrypoint to generate interactive viewer pages."""
    parser = argparse.ArgumentParser()
    add_min_date_arg(parser)
    add_condition_filter_args(parser)
    args = parser.parse_args()

    condition_filters = filters_from_args(args)
    pages = _generate_pages(args.min_date, condition_filters)
    if not pages:
        print("No pages generated (no data found).")
        return
    print(f"Wrote {len(pages)} page(s) to {VIEW_DIR.resolve()}.")


if __name__ == "__main__":
    main()
