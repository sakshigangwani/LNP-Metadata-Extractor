"""Structured LNP metadata extraction with OpenAI.

Uses OpenAI Structured Outputs: the LNPMetadata Pydantic model is passed as the
`response_format`, so the model is constrained to return JSON matching the schema,
which the SDK parses and validates back into an LNPMetadata instance.
"""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from .config import settings
from .schema import LNPMetadata

_SYSTEM = (
    "You are a meticulous scientific data curator specializing in lipid "
    "nanoparticle (LNP) formulations. You extract structured metadata from a "
    "research paper into a fixed schema.\n\n"
    "Rules:\n"
    "- Only record values explicitly stated or unambiguously derivable from the paper.\n"
    "- Use null for any field not reported. Never guess or fabricate values.\n"
    "- Record values EXACTLY as the paper reports them. For string measurement fields, "
    "copy the reported form verbatim — preserve ranges ('90-110 nm'), inequalities "
    "('<0.2', '>95%'), approximations ('~100 nm'), and qualitative descriptors "
    "('slightly negative'). Do NOT round, average, or collapse a range/inequality/"
    "qualitative value into a single number. e.g. if the paper says diameter is "
    "'~90-110 nm', record '~90-110 nm', not '100'.\n"
    "- If a paper reports multiple distinct formulations, extract the primary / lead "
    "formulation and note the others in `notes`.\n"
    "- List every field you inferred or were unsure about in `uncertain_fields`."
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # Reads OPENAI_API_KEY from the environment.
    return OpenAI()


def extract_metadata(paper_text: str) -> LNPMetadata:
    """Extract LNP metadata from full paper text. Returns a validated LNPMetadata."""
    text = paper_text[: settings.max_extraction_chars]

    completion = _client().beta.chat.completions.parse(
        model=settings.openai_extraction_model,
        max_tokens=settings.extraction_max_tokens,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    "Extract the LNP formulation metadata from the following paper.\n\n"
                    "<paper>\n" + text + "\n</paper>"
                ),
            },
        ],
        response_format=LNPMetadata,
    )

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused to extract: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError(
            f"No parsed output returned (finish_reason="
            f"{completion.choices[0].finish_reason!r})."
        )
    return message.parsed
