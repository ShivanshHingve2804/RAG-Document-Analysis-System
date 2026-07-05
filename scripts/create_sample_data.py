"""
create_sample_data.py
----------------------
Generates two small sample PDFs into data/pdfs/ so the pipeline can be
run and evaluated end-to-end without needing to source real documents
first. Uses reportlab (pip install reportlab).

Run once:  python scripts/create_sample_data.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from src import config

DOC_1_TITLE = "Transformer Architecture Overview"
DOC_1_PARAGRAPHS = [
    "The Transformer architecture, introduced in the 2017 paper "
    "'Attention Is All You Need', replaced recurrence and convolutions "
    "with a mechanism called self-attention.",

    "Self-attention allows each token in a sequence to weigh the "
    "importance of every other token when building its representation, "
    "which enables parallel processing across the entire sequence "
    "instead of the step-by-step processing required by RNNs.",

    "A standard Transformer has an encoder stack and a decoder stack. "
    "The encoder maps an input sequence to a sequence of continuous "
    "representations, while the decoder generates an output sequence "
    "one token at a time, attending both to its own previous outputs "
    "and to the encoder's representations.",

    "GPT-style models use a decoder-only architecture: they drop the "
    "encoder entirely and use masked self-attention so each token can "
    "only attend to previous tokens, which makes the model naturally "
    "suited to autoregressive text generation.",

    "BERT, in contrast, uses only the encoder stack with bidirectional "
    "self-attention, allowing every token to attend to every other "
    "token in the sequence. This makes BERT strong at understanding "
    "tasks like classification, but it is not designed for free-form "
    "text generation the way GPT is.",

    "Positional encodings are added to token embeddings because "
    "self-attention itself has no built-in notion of token order. "
    "The original paper used fixed sinusoidal functions, while many "
    "later models use learned positional embeddings instead.",

    "Multi-head attention runs several attention operations in "
    "parallel, each with its own learned projection, allowing the "
    "model to jointly attend to information from different "
    "representation subspaces at different positions.",
]

DOC_2_TITLE = "Company Remote Work Policy"
DOC_2_PARAGRAPHS = [
    "This policy applies to all full-time employees who have been "
    "approved for remote or hybrid work arrangements by their manager "
    "and Human Resources.",

    "Employees working remotely are expected to be available during "
    "core collaboration hours, defined as 10:00 AM to 4:00 PM in their "
    "assigned office's local timezone, unless otherwise agreed with "
    "their manager.",

    "The company will provide a one-time home office stipend of $500 "
    "for approved remote employees to cover equipment such as a "
    "monitor, chair, or desk, subject to submission of receipts within "
    "60 days of approval.",

    "All remote employees must use company-issued VPN software when "
    "accessing internal systems, and must complete the annual security "
    "awareness training within 30 days of their remote work approval.",

    "Remote employees are required to attend in-person team gatherings "
    "at least twice per year, with travel expenses covered by the "
    "company in accordance with the standard travel policy.",

    "Performance for remote employees is evaluated using the same "
    "criteria and review cycle as in-office employees; remote work "
    "status has no bearing on promotion or compensation decisions.",
]


def _build_pdf(path: Path, title: str, paragraphs):
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                             leftMargin=inch, rightMargin=inch,
                             topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.3 * inch)]
    for para in paragraphs:
        story.append(Paragraph(para, styles["BodyText"]))
        story.append(Spacer(1, 0.15 * inch))
    doc.build(story)
    print(f"Created {path}")


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _build_pdf(config.DATA_DIR / "transformer_architecture.pdf",
               DOC_1_TITLE, DOC_1_PARAGRAPHS)
    _build_pdf(config.DATA_DIR / "remote_work_policy.pdf",
               DOC_2_TITLE, DOC_2_PARAGRAPHS)


if __name__ == "__main__":
    main()
