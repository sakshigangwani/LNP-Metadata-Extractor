"""Extract plain text from a PDF."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Union

from pypdf import PdfReader


def _clean(text: str) -> str:
    # Join hyphenated line breaks: "nano-\nparticle" -> "nanoparticle"
    text = re.sub(r"-\n(?=\w)", "", text)
    # Collapse runs of whitespace but keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_text(source: Union[str, Path, bytes]) -> str:
    """Read a PDF from a path or raw bytes and return cleaned text.

    Raises ValueError if no extractable text is found (e.g. a scanned PDF with no
    text layer) so the caller can surface a clear message instead of indexing
    empty content.
    """
    if isinstance(source, (bytes, bytearray)):
        import io

        reader = PdfReader(io.BytesIO(source))
    else:
        reader = PdfReader(str(source))

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    text = _clean("\n\n".join(pages))
    if len(text) < 50:
        raise ValueError(
            "No extractable text found in this PDF. It may be a scanned image "
            "without a text layer (OCR would be required)."
        )
    return text
