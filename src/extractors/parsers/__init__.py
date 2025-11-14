"""
Document parsers for converting various formats to markdown.
"""

from .vision import VisionParser
from .pymupdf import PyMuPDFParser

try:
    from .docling import DoclingParser
    _HAS_DOCLING = True
except ImportError:
    _HAS_DOCLING = False

try:
    from .docx import DocxParser
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

__all__ = [
    "VisionParser",
    "PyMuPDFParser",
]

if _HAS_DOCLING:
    __all__.append("DoclingParser")

if _HAS_DOCX:
    __all__.append("DocxParser")
