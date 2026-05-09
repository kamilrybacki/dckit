"""LLM prompts used by discovery + oracle. Centralised for testability."""

from __future__ import annotations

MICRO_NAME_PROMPT = """You name topics for a knowledge taxonomy.

Below are 3 short excerpts from a knowledge corpus that semantic search clustered together. Output a SINGLE noun phrase (3-8 words) naming the unifying topic. Be specific but not narrow. No preamble, no quotation marks, no trailing period.

[1] {c1}
[2] {c2}
[3] {c3}

Topic:"""


MACRO_PROPOSE_PROMPT = """Design EXACTLY {k_macro} top-level subject areas to cover a heterogeneous knowledge corpus.

The corpus contains these {n_micro} micro-topics (for context only — do NOT assign them):
{micro_topics_listing}

Pick {k_macro} two-digit codes from: {codes_csv} (use {k_macro} of these, ascending order).

Each area:
- One 2-digit code (ascending order across the output)
- A 2-4 word title (broad subject domain, not a narrow project)
- Examples: "Software Development", "References & Dictionaries", "Spirituality & Philosophy"

Output ONLY a JSON array of EXACTLY {k_macro} objects, no prose, no fences:
[
  {{"code": "00", "title": "System & Meta"}},
  {{"code": "10", "title": "Knowledge & Learning"}}
]

JSON:"""


REFINE_EMPTY_PROMPT = """The previous taxonomy proposal had {n_empty} areas with zero matching topics in the corpus. They must be REPLACED with new areas that match populated content.

EMPTY areas to replace:
{empty_listing}

POPULATED areas (DO NOT change — for context only):
{populated_listing}

Output EXACTLY {n_empty} replacement (code, title) pairs:
- Pick {n_empty} codes from these AVAILABLE codes (must not collide with populated): {available_codes}
- Each title: 2-4 words, broad subject domain, not covered by populated areas
- Output codes in ASCENDING order

Output ONLY a JSON array of EXACTLY {n_empty} objects:
[{{"code": "XX", "title": "..."}}, ...]

JSON:"""


ORACLE_JUDGE_PROMPT = """You are evaluating a discrete-codebook taxonomy assignment.

Below is a passage from a knowledge corpus, followed by {k} candidate area labels.

=== BEGIN PASSAGE ===
{passage}
=== END PASSAGE ===

=== BEGIN AREAS ===
{area_listing}
=== END AREAS ===

Pick the SINGLE area that best fits the passage's primary topic. Output JSON:
{{"area_index": <0-based integer>, "confidence": <"high"|"medium"|"low">}}

JSON:"""
