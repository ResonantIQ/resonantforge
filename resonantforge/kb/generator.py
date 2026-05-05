"""
KB fixture generator — writes chunks to the corpus output directory.

Public interface
----------------
generate_kb(profile_name, output_dir) -> (chunks, content_hash)
    Generates KB fixtures for a named profile and writes them to
    ``output_dir/knowledge_base/``.  Returns the chunk list and a
    deterministic SHA-256 content hash for the manifest.

load_kb_chunks(corpus_dir, profile_name) -> list[KBChunk]
    Loads previously-generated KB chunks from a corpus directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from resonantforge.kb.ps_content import get_ps_kb_chunks
from resonantforge.kb.saas_content import get_saas_kb_chunks
from resonantforge.schemas import KBChunk


def generate_kb(profile_name: str, output_dir: Path) -> tuple[list[KBChunk], str]:
    """
    Generate KB fixtures for a profile and write them to ``output_dir/knowledge_base/``.

    Two artefacts are written:

    - One Markdown file per unique ``document_path``, with each chunk's text
      preceded by an HTML comment carrying ``chunk_id`` and ``constraint_type``
      for human readability.
    - A single ``chunks.jsonl`` file containing one ``KBChunk`` JSON record per
      line, used by downstream layers (retrieval, validation, manifest hashing).

    All chunk IDs are deterministic strings — they are defined in the content
    modules, not generated here — so repeated calls with the same profile
    produce identical output and identical hashes.

    Args:
        profile_name: ``"saas"`` or ``"ps"`` / ``"professional_services"``.
        output_dir: Root output directory for this profile's corpus data.
                    ``knowledge_base/`` is created as a subdirectory.

    Returns:
        A tuple of ``(chunks, content_hash)`` where ``content_hash`` is a
        ``sha256:<hex>`` string suitable for embedding in the corpus manifest.

    Raises:
        ValueError: If ``profile_name`` is not a recognised profile key.
    """
    if profile_name == "saas":
        chunks = get_saas_kb_chunks()
    elif profile_name in ("ps", "professional_services"):
        chunks = get_ps_kb_chunks()
    else:
        raise ValueError(
            f"Unknown profile: {profile_name!r}. Expected 'saas' or 'ps'."
        )

    kb_dir = output_dir / "knowledge_base"

    # -----------------------------------------------------------------
    # Group chunks by document_path for Markdown output.
    # We preserve insertion order so repeated generations produce
    # identical files (important for hash determinism).
    # -----------------------------------------------------------------
    doc_files: dict[str, list[KBChunk]] = {}
    for chunk in chunks:
        doc_files.setdefault(chunk.document_path, []).append(chunk)

    for doc_path, doc_chunks in doc_files.items():
        full_path = kb_dir / doc_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as fh:
            fh.write(f"# {doc_path}\n\n")
            for chunk in doc_chunks:
                fh.write(
                    f"<!-- chunk_id: {chunk.chunk_id} "
                    f"constraint_type: {chunk.constraint_type.value} -->\n"
                )
                fh.write(chunk.chunk_text + "\n\n")

    # -----------------------------------------------------------------
    # Write the JSONL index — one record per chunk.
    # -----------------------------------------------------------------
    chunks_meta_path = kb_dir / "chunks.jsonl"
    chunks_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_meta_path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")

    # -----------------------------------------------------------------
    # Compute a deterministic content hash over all chunk IDs + texts,
    # sorted by chunk_id so insertion order doesn't affect the digest.
    # -----------------------------------------------------------------
    content = "".join(
        c.chunk_id + c.chunk_text
        for c in sorted(chunks, key=lambda x: x.chunk_id)
    )
    content_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()

    return chunks, content_hash


def load_kb_chunks(corpus_dir: Path, profile_name: str) -> list[KBChunk]:
    """
    Load KB chunks from a previously-generated corpus directory.

    Reads ``corpus_dir/<profile_name>/knowledge_base/chunks.jsonl`` and
    deserialises each line into a ``KBChunk``.  Returns an empty list if the
    JSONL file does not exist (e.g., the corpus has not been generated yet).

    Args:
        corpus_dir: Root corpus output directory (parent of profile directories).
        profile_name: The profile subdirectory name (e.g., ``"saas"``).

    Returns:
        List of ``KBChunk`` objects in the order they appear in the JSONL file.
    """
    chunks_path = corpus_dir / profile_name / "knowledge_base" / "chunks.jsonl"
    if not chunks_path.exists():
        return []

    chunks: list[KBChunk] = []
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(KBChunk.model_validate_json(line))
    return chunks
