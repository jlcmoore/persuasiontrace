"""
src/tests/experiment/test_flagged_messages_vad.py
~~~~~~~~~~
- Reads src/data/flagged_messages.jsonl.
- For each item, picks a data:audio/... URI from original_audio (preferred) or audio.
- Calls gpt-4o-transcribe with include=["logprobs"] to regenerate VAD logprobs.
- Checks is_speech_like at a configurable threshold (default 0.6).
- If a failure occurs and interactive debugging is enabled, it immediately:
  - Writes a small HTML file that can play the failing audio and
    shows the exact below-threshold tokens.
  - Optionally opens the HTML in your browser.
- If interactive debugging is off, it collects all failures and reports them at the end.
- Skips gracefully if the file is missing or no items contain a usable data:audio URI.

Environment toggles:
- VAD_INTERACTIVE_DEBUG=1 fail fast on first failure and emit audio HTML.
- VAD_DEBUG_OPEN_BROWSER=1 open the debug HTML in your default browser.
- VAD_THRESHOLD=0.6 set a custom probability threshold.
"""

import io
import json
import math
import os
import pathlib
import tempfile
import textwrap
import webbrowser
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote as urlquote

import pytest
from openai import OpenAI  # lazy import

from experiment.endpoints import THRESHOLD_PROB, decode_audio_data_uri, is_speech_like
from experiment.utils import get_data_file_path

DATA_POS_PATH = pathlib.Path(get_data_file_path("flagged_messages.jsonl"))
DATA_NEG_PATH = pathlib.Path(get_data_file_path("non_flagged_messages.jsonl"))


def _truthy_env(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val not in ("", "0", "false", "no", "off")


def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {i}: {e}") from e


def _pick_audio_data_uri(obj: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit original_audio, then fall back to audio
    for key in ("original_audio", "audio"):
        val = obj.get(key)
        if isinstance(val, str) and val.startswith("data:audio/"):
            return val
    return None


def _compute_vad_from_audio_uri(audio_data_uri: str) -> Dict[str, Any]:
    """
    Recomputes VAD logprobs using gpt-4o-transcribe. Returns a plain dict
    containing at least 'logprobs' suitable for is_speech_like.
    Raises on errors so tests surface issues instead of silently passing.
    """
    audio_bytes, subtype = decode_audio_data_uri(audio_data_uri)
    buf = io.BytesIO(audio_bytes)
    buf.name = f"audio.{(subtype or 'bin').lower()}"

    client = OpenAI()
    resp = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=buf,
        include=["logprobs"],
    )
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    # Fallback: extract what we need if SDK object is not pydantic-like
    return {"logprobs": getattr(resp, "logprobs", [])}


def _tokens_below_threshold(
    vad_response: Dict[str, Any], threshold: float
) -> List[Tuple[int, str, float, float]]:
    out: List[Tuple[int, str, float, float]] = []
    entries = vad_response.get("logprobs") or []
    log_thresh = math.log(threshold)
    for idx, e in enumerate(entries):
        lp = e.get("logprob")
        tok = e.get("token")
        if isinstance(lp, (float, int)) and lp < log_thresh:
            out.append(
                (idx, str(tok) if tok is not None else "", float(lp), math.exp(lp))
            )
    return out


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    q = min(max(q, 0.0), 1.0)
    s = sorted(vals)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    w = pos - lo
    return s[lo] * (1 - w) + s[hi] * w


def _contiguous_spans(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    spans = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        spans.append((start, prev))
        start = prev = i
    spans.append((start, prev))
    return spans


def _build_csv(vad_response: Dict[str, Any], threshold: float) -> str:
    rows = ["index,token,logprob,prob,below_threshold"]
    log_thresh = math.log(threshold)
    entries = vad_response.get("logprobs") or []
    for idx, e in enumerate(entries):
        lp = e.get("logprob")
        tok = e.get("token")
        if not isinstance(lp, (float, int)):
            continue
        p = math.exp(lp)
        below = int(lp < log_thresh)
        token_escaped = (str(tok) if tok is not None else "").replace('"', '""')
        rows.append(f'{idx},"{token_escaped}",{lp:.6f},{p:.6f},{below}')
    return "\n".join(rows)


def _try_get_transcript(audio_data_uri: str) -> Optional[str]:
    try:
        audio_bytes, subtype = decode_audio_data_uri(audio_data_uri)
    except ValueError:
        return None
    buf = io.BytesIO(audio_bytes)
    buf.name = f"audio.{(subtype or 'bin').lower()}"
    client = OpenAI()
    # Whisper gives consistent transcripts; use minimal settings.
    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        response_format="json",
    )
    # Pydantic-like or dict-like:
    if isinstance(resp, dict):
        return resp.get("text", "")
    if hasattr(resp, "text"):
        return resp.text
    if hasattr(resp, "model_dump"):
        return resp.model_dump().get("text", "")
    return None


def _write_debug_html(
    audio_data_uri: str,
    vad_response: Dict[str, Any],
    threshold: float,
    pos_neg: str,
    *,
    lineno: Optional[int] = None,
    keys: Optional[List[str]] = None,
    transcript: Optional[str] = None,
) -> pathlib.Path:
    # Compute stats
    entries = vad_response.get("logprobs") or []
    lps = [
        float(e["logprob"])
        for e in entries
        if isinstance(e.get("logprob"), (float, int))
    ]
    tokens = [str(e.get("token") or "") for e in entries]
    n = len(lps)
    log_thresh = math.log(threshold)
    below_idx = [i for i, lp in enumerate(lps) if lp < log_thresh]
    below_count = len(below_idx)
    frac_below = (below_count / n) if n else 0.0
    min_lp = min(lps) if lps else float("nan")
    max_lp = max(lps) if lps else float("nan")
    mean_lp = sum(lps) / n if n else float("nan")
    med_lp = _percentile(lps, 0.5)
    p05 = _percentile(lps, 0.05)
    p25 = _percentile(lps, 0.25)
    p75 = _percentile(lps, 0.75)
    worst_order = sorted(range(n), key=lambda i: lps[i] if i < len(lps) else 0.0)
    worst_top = worst_order[: min(32, len(worst_order))]

    # Build offenders table rows
    def fmt_token(t: str) -> str:
        return (
            t.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )

    offender_rows = []
    for i in worst_top:
        lp = lps[i]
        p = math.exp(lp)
        t = fmt_token(tokens[i]) if i < len(tokens) else ""
        cls = "bad" if lp < log_thresh else "ok"
        offender_rows.append(
            f"<tr class='{cls}'><td>{i}</td><td><code>{t}</code></td>"
            + f"<td>{lp:.4f}</td><td>{p:.4f}</td></tr>"
        )

    # All tokens table
    all_rows = []
    for i, lp in enumerate(lps):
        p = math.exp(lp)
        t = fmt_token(tokens[i])
        cls = "bad" if lp < log_thresh else "ok"
        bar_w = max(1, int(min(100, p * 100)))
        all_rows.append(
            f"<tr class='{cls}'>"
            f"<td>{i}</td>"
            f"<td><code>{t}</code></td>"
            f"<td>{lp:.4f}</td>"
            f"<td>{p:.4f}</td>"
            f"<td><div class='bar' style='width:{bar_w}px'></div></td>"
            f"</tr>"
        )

    spans = _contiguous_spans(below_idx)
    spans_preview = (
        "none"
        if not spans
        else ", ".join([f"[{a}-{b}] (len {b-a+1})" for a, b in spans[:10]])
    )

    # Audio details
    subtype = "unknown"
    a_bytes_len = 0
    try:
        raw_bytes, subtype = decode_audio_data_uri(audio_data_uri)
        a_bytes_len = len(raw_bytes)
    except ValueError:
        pass

    # CSV data URI
    csv = _build_csv(vad_response, threshold)

    csv_data_uri = "data:text/csv;charset=utf-8," + urlquote(csv)

    # Optional transcript block
    transcript_block = ""
    if transcript:
        t_html = fmt_token(transcript)
        transcript_block = f"""
        <div class="row">
          <h2>Transcript (debug)</h2>
          <pre class="mono">{t_html}</pre>
        </div>
        """

    # Keys and meta
    meta_lines = []
    meta_lines.append(f"Example type: {pos_neg}")
    if lineno is not None:
        meta_lines.append(f"JSONL line: {lineno}")
    if keys:
        meta_lines.append(f"Top-level keys: {keys}")
    meta_html = "<br>".join(meta_lines)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VAD Failure Debug</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{ --bad: #b00020; --ok: #0a6; --muted: #666; }}
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }}
    h1, h2, h3 {{ margin: 0.4rem 0; }}
    .row {{ margin: 1rem 0; }}
    .muted {{ color: var(--muted); }}
    pre.mono {{ white-space: pre-wrap; word-break: break-word; background: #f6f8fa; padding: 0.75rem; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }}
    thead th {{ background: #f3f4f6; position: sticky; top: 0; }}
    tr.bad {{ background: #ffecec; }}
    tr.ok {{ background: #f4fff7; }}
    .bar {{ height: 10px; background: linear-gradient(90deg, var(--bad), var(--ok)); }}
    .wrap {{ max-height: 360px; overflow: auto; border: 1px solid #eee; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    details > summary {{ cursor: pointer; }}
  </style>
</head>
<body>
  <h1>VAD Failure Debug</h1>
  <div class="row">
    <audio controls preload="metadata" src="{audio_data_uri}"></audio>
    <div class="muted">Audio: audio/{subtype} • {a_bytes_len} bytes</div>
  </div>

  <div class="row">
    <h2>Summary</h2>
    <pre class="mono">
Threshold p >= {threshold:.3f} (log-threshold {log_thresh:.4f})
Tokens: {n}, Below-threshold: {below_count} ({frac_below:.1%})
LogProb: min {min_lp:.4f}, median {med_lp:.4f}, mean {mean_lp:.4f}, max {max_lp:.4f}
Percentiles: p05 {p05:.4f}, p25 {p25:.4f}, p75 {p75:.4f}
Failing spans (first 10): {spans_preview}
{meta_html}
    </pre>
  </div>

  <div class="row">
    <h2>Top offenders (lowest probabilities)</h2>
    <div class="wrap">
      <table>
        <thead><tr><th>#</th><th>token</th><th>logprob</th><th>prob</th></tr></thead>
        <tbody>
          {"".join(offender_rows)}
        </tbody>
      </table>
    </div>
    <a download="vad_logprobs.csv" href="{csv_data_uri}">Download CSV</a>
  </div>

  <div class="row">
    <h2>All tokens</h2>
    <div class="wrap">
      <table>
        <thead><tr><th>#</th><th>token</th><th>logprob</th><th>prob</th><th>bar</th></tr></thead>
        <tbody>
          {"".join(all_rows)}
        </tbody>
      </table>
    </div>
    <div class="muted">Rows highlighted red are below threshold.</div>
  </div>

  {transcript_block}
</body>
</html>"""
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="vad_debug_"))
    out = tmpdir / "failure.html"
    out.write_text(html, encoding="utf-8")
    return out


def _run_dataset(
    name: str,
    path: pathlib.Path,
    expect_speech_like: bool,
    *,
    threshold: float,
    interactive: bool,
    open_browser: bool,
) -> None:
    """
    Runs VAD validation on one dataset. If interactive, fail fast on
    the first mismatch and emit a rich HTML artifact.
    Otherwise, aggregate all mismatches and fail at the end.
    """
    if not path.exists():
        pytest.skip(f"{name}: {path} not present; nothing to validate")

    failures: List[str] = []
    checked = 0
    skipped_no_audio = 0

    for lineno, obj in enumerate(_iter_jsonl(path), start=1):
        audio_uri = _pick_audio_data_uri(obj)
        if not audio_uri:
            skipped_no_audio += 1
            continue

        vad = _compute_vad_from_audio_uri(audio_uri)
        checked += 1

        ok = is_speech_like(vad, threshold_prob=threshold)
        match = (ok is True) if expect_speech_like else (ok is False)

        if match:
            continue

        # Build readable reason and handle interactive debug
        verdict = "speech-like" if ok else "not speech-like"
        expected = "speech-like" if expect_speech_like else "not speech-like"
        below = _tokens_below_threshold(vad, threshold)
        head = below[:10]
        summary_lines = [
            f"[{name}] Line {lineno}: expected {expected}, ",
            f"got {verdict} (threshold p >= {threshold:.3f})",
            f"Below-threshold tokens: {len(below)}",
            "",
            "First offenders (index, token, logprob, prob):",
        ]
        for idx, tok, lp, p in head:
            token_repr = tok.replace("\n", "\\n")
            summary_lines.append(
                f"  {idx:4d}  {token_repr!r:20s}  lp={lp: .4f}  p={p:.4f}"
            )
        if len(below) > len(head):
            summary_lines.append(f"... and {len(below) - len(head)} more.")
        reason = "\n".join(summary_lines)

        if interactive:
            transcript = None
            if _truthy_env("VAD_DEBUG_GET_TRANSCRIPT", "0"):
                transcript = _try_get_transcript(audio_uri)

            html_path = _write_debug_html(
                audio_uri,
                vad,
                threshold,
                name,
                lineno=lineno,
                keys=list(obj.keys()),
                transcript=transcript,
            )
            if open_browser:
                webbrowser.open(html_path.as_uri())
            pytest.fail(
                textwrap.dedent(
                    f"""
                    [{name}] VAD mismatch (interactive mode):
                    - Expected: {expected}, Got: {verdict}
                    - Debug HTML: {html_path}
                    - Details: See Summary, Top offenders, and All tokens sections.
                    """
                ).strip()
            )

        failures.append(reason)
        # For non-interactive, continue collecting all failures

    if checked == 0:
        pytest.skip(
            f"[{name}] No entries with usable data:audio/... URIs "
            f"(skipped_no_audio={skipped_no_audio})."
        )

    if failures:
        pytest.fail(
            f"[{name}] {len(failures)} item(s) mismatched:\n\n" + "\n\n".join(failures)
        )


@pytest.mark.expensive
def test_vad_regenerated_is_speech_like_on_pos_and_neg():
    """
    Validates VAD on both datasets:
      - flagged_messages.jsonl (pos): expected speech-like
      - non_flagged_messages.jsonl (neg): expected speech-like by default
        (set VAD_NEG_EXPECT_SPEECH=0 to expect not-speech-like).

    Env:
      - OPENAI_API_KEY must be set
      - VAD_THRESHOLD (default 0.6)
      - VAD_INTERACTIVE_DEBUG=1 (fail fast with HTML on first mismatch)
      - VAD_DEBUG_OPEN_BROWSER=1 (auto-open HTML)
      - VAD_DATASET=pos|neg to limit to a single dataset
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping live VAD regeneration")

    threshold = THRESHOLD_PROB
    interactive = _truthy_env("VAD_INTERACTIVE_DEBUG", "0")
    open_browser = _truthy_env("VAD_DEBUG_OPEN_BROWSER", "0")
    dataset_sel = os.getenv("VAD_DATASET", "").strip().lower()

    # Build selected datasets
    datasets = []
    if dataset_sel in ("", "pos"):
        datasets.append(("pos", DATA_POS_PATH, False))
    if dataset_sel in ("", "neg"):
        datasets.append(("neg", DATA_NEG_PATH, True))

    if not datasets:
        pytest.skip("VAD_DATASET env filtered out all datasets")

    # Run each dataset; if interactive, the first mismatch will fail immediately
    for name, path, expect_speech in datasets:
        _run_dataset(
            name=name,
            path=path,
            expect_speech_like=expect_speech,
            threshold=threshold,
            interactive=interactive,
            open_browser=open_browser,
        )
