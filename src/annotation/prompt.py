"""Prompt templates and helpers for rhetoric annotation scoring."""

from typing import Mapping, Sequence

from annotation.schema import ANNOTATION_CODES


def format_feature_list(codes: Sequence[str]) -> str:
    """Return a bullet list of annotation codes for prompt text."""
    return "\n".join(f"   - {code}" for code in codes)


def format_schema_block(codes: Sequence[str]) -> str:
    """Return a JSON schema block for annotation codes."""
    lines = ["{"]
    for index, code in enumerate(codes):
        lines.append(f'  "{code}": {{')
        lines.append('    "rationale": "<short justification>",')
        lines.append('    "score": <number from 0 to 2>')
        if index == len(codes) - 1:
            lines.append("  }")
        else:
            lines.append("  },")
    lines.append("}")
    return "\n".join(lines)


RHETORIC_PROMPT = f"""\
You are an expert annotator of persuasive strategies in multi-turn dialogues.

Your task: given a dialogue and one FOCUS message in that dialogue, you will:
1. Carefully read the whole dialogue for context.
2. Evaluate ONLY the FOCUS message on {len(ANNOTATION_CODES)} persuasion-related features:
{format_feature_list(ANNOTATION_CODES)}
3. For EACH feature:
   - Briefly explain (1-3 sentences) why you assigned the score, referring to
     specific aspects of the FOCUS message.
   - Then assign an integer score from 0 to 2.

SCORING SCALE (0-2):
- 0 = absent (feature does not appear in the FOCUS message).
- 1 = somewhat present (feature appears but is not dominant).
- 2 = very present (feature is a dominant part of the FOCUS message).

LOGOS
- What to capture:
  - Use of facts, logic, or reasoning to persuade.
  - Includes causal explanations, conditional "if...then" arguments,
    comparisons, and generalizations that appeal to rational evaluation.
- Examples of cues:
  - Explicit reasoning ("because...", "therefore...", "if X then Y").
  - References to statistics, probabilities, logical consequences, or
    trade-offs.
- Exclude:
  - Purely emotional statements without reasoning.
  - Mere assertions of opinion without explanation.

PATHOS
- What to capture:
  - Emotional or affective appeals, where the message tries to persuade by
    arousing feelings (e.g., fear, anger, empathy, pride, guilt, hope).
  - Narrative or vivid storytelling primarily used to move the reader
    emotionally.
- Examples of cues:
  - Strong emotional adjectives/adverbs.
  - First-person or third-person stories whose main function is to evoke
    emotion rather than to provide factual detail or technical explanation.
- Note: A message can be both logos and pathos if it mixes reasoning with
  emotional framing.

ETHOS
- What to capture:
  - Attempts to build the speaker's credibility, trustworthiness, or
    authority.
  - The speaker presents themselves (or a close identity they speak for) as
    expert, experienced, high-status, or morally reliable.
- Examples of cues:
  - Stating professional or lived expertise ("As a doctor...", "I've worked in
    this field for 20 years...").
  - Emphasizing fairness, honesty, or reputation ("I have no stake in this...",
    "I've always been honest about...").
- Exclude:
  - Mentions of other people's expertise as mere support, unless clearly used
    to boost the speaker's own credibility.

GENERAL GUIDELINES
- Focus only on the FOCUS message, but use the prior turns for context (e.g.,
  to know what is being claimed or who the speaker is).
- A single sentence can contribute to multiple features (e.g., a personal story
  that is both logos and pathos).
- Be conservative:
  - Do NOT infer features that are not clearly supported by the text.
  - For ethos, do NOT assume the speaker is credible unless they actively build
    that impression in the message.
- If a feature is truly absent, assign 0 and explain briefly why.

INPUT FORMAT
You will receive a formatted context block followed by the FOCUS message.

Format:

## Context (earlier messages, oldest first):
```
speaker: message text
speaker: message text
...
```

## Focus message (to annotate):
```
speaker: message text
```

- If there is no earlier context, the context block will say "(none)".
- The focus message appears only in the focus block, not in the context.

OUTPUT FORMAT (STRICT JSON)
- Output MUST be a single valid JSON object.
- Use only double quotes for keys and string values.
- Do NOT include any text before or after the JSON (no markdown, no comments).
- Keys must appear exactly as specified below.

Schema:

{format_schema_block(ANNOTATION_CODES)}

- Scores must be integers.
- Rationales should be concise (one sentence each).
"""

# from \citep{xia_persua_2022}
FEW_SHOT_EXAMPLE_1 = """\
--------------------
EXAMPLE 1 (logos)

Input:

## Context (earlier messages, oldest first):
```
(none)
```

## Focus message (to annotate):
```
user: If it is so much trouble to get dates, maintain a relationship, and not \
be yourself, why are you still chasing these goals
```

Expected output:

{
  "logos": {
    "rationale": "The message poses a conditional-style challenge that reasons \
about the costs and benefits of pursuing relationships, using logical \
questioning rather than describing specific past events.",
    "score": 2
  },
  "pathos": {
    "rationale": "The tone is mildly critical or exasperated, but it does not \
strongly try to arouse emotion through vivid or affective language.",
    "score": 1
  },
  "ethos": {
    "rationale": "The speaker does not present credentials, status, or moral \
character; they only question the logic of the behavior.",
    "score": 0
  }
}
"""

# from \citep{modzelewski_pcot_2025}
FEW_SHOT_EXAMPLE_2 = """\
--------------------
EXAMPLE 2 (slogan / Call strategy, mostly pathos)

Input:

## Context (earlier messages, oldest first):
```
(none)
```

## Focus message (to annotate):
```
user: Make America Great Again!
```

Expected output:

{
  "logos": {
    "rationale": "The slogan asserts a desired goal but does not provide \
reasons, causal explanations, or logical argumentation.",
    "score": 0
  },
  "pathos": {
    "rationale": "The phrase appeals to nostalgia and national pride, aiming \
to evoke positive emotions rather than reasoned analysis.",
    "score": 2
  },
  "ethos": {
    "rationale": "The speaker does not explicitly present their own \
credibility or expertise, so there is no clear credibility appeal in the \
wording itself.",
    "score": 0
  }
}
"""

FEW_SHOT_EXAMPLES = (
    "FEW-SHOT EXAMPLES\n\n"
    "Below are examples to illustrate how to apply these definitions.\n\n"
    f"{FEW_SHOT_EXAMPLE_1}\n\n"
    f"{FEW_SHOT_EXAMPLE_2}\n\n"
    "END OF INSTRUCTIONS.\n"
    "Respond to future inputs using ONLY the JSON format specified above.\n"
)


def format_context_block(
    turns: Sequence[Mapping[str, str]], target_turn_index: int
) -> str:
    """Return a formatted context block from earlier dialogue turns."""

    if target_turn_index < 0:
        raise ValueError("target_turn_index must be non-negative.")
    if target_turn_index > len(turns):
        raise ValueError("target_turn_index exceeds number of turns.")

    lines = ["## Context (earlier messages, oldest first):", "```"]
    has_context = False
    for item in turns[:target_turn_index]:
        speaker = str(item.get("speaker") or "unknown").strip()
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{speaker}: {text}")
        has_context = True
    if not has_context:
        lines.append("(none)")
    lines.append("```")
    return "\n".join(lines) + "\n\n"


def format_target_block(
    turns: Sequence[Mapping[str, str]], target_turn_index: int
) -> str:
    """Return a formatted focus message block for annotation."""

    if target_turn_index < 0 or target_turn_index >= len(turns):
        raise ValueError("target_turn_index must point to an existing turn.")

    target = turns[target_turn_index]
    speaker = str(target.get("speaker") or "unknown").strip()
    text = str(target.get("text") or "").strip()

    lines = ["## Focus message (to annotate):", "```"]
    if text:
        lines.append(f"{speaker}: {text}")
    else:
        lines.append("(missing focus message)")
    lines.append("```")
    return "\n".join(lines) + "\n"


def format_dialogue_for_prompt(
    turns: Sequence[Mapping[str, str]], target_turn_index: int
) -> str:
    """Return the full formatted prompt input from turns and target index."""

    context_block = format_context_block(turns, target_turn_index)
    target_block = format_target_block(turns, target_turn_index)
    return f"{context_block}{target_block}"
