
"""Document text extraction for the Group 13 AI career match project."""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from docx import Document


MAX_FILE_SIZE = 10 * 1024 * 1024


def extract_pdf_text(file_bytes):
    """Extract selectable text from a PDF document."""
    reader = PdfReader(BytesIO(file_bytes))

    if reader.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported.")

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""

        if text.strip():
            pages.append(text.strip())

    return "\n".join(pages)


def extract_docx_text(file_bytes):
    """Extract paragraphs and table-cell text from DOCX."""
    document = Document(BytesIO(file_bytes))
    parts = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text.strip())

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())

    return "\n".join(parts)


def extract_document_text(file_bytes, filename):
    """
    Extract text from PDF, DOCX, or TXT documents.

    Parameters
    ----------
    file_bytes : bytes
        Uploaded document content.
    filename : str
        Original filename.

    Returns
    -------
    str
        Extracted document text.
    """
    if not isinstance(file_bytes, bytes):
        raise TypeError("file_bytes must be bytes.")

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("Maximum supported file size is 10 MB.")

    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        text = extract_pdf_text(file_bytes)

    elif suffix == ".docx":
        text = extract_docx_text(file_bytes)

    elif suffix == ".txt":
        text = file_bytes.decode("utf-8-sig")

    else:
        raise ValueError(
            "Unsupported file type. Use PDF, DOCX, or TXT."
        )

    if not text.strip():
        raise ValueError(
            "No readable text was found. "
            "Scanned PDFs may require OCR."
        )

    return text.strip()
