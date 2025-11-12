"""
Document Parsers Package

A collection of specialized document parsers for various formats.
"""

# Import all parser classes and convenience functions for easy access
from .vision_parser import VisionParser, get_openai_config
from .docling_parser import DoclingParser, parse_document as parse_document_docling
from .pymupdf_parser import PyMuPDFParser, parse_pdf as parse_pdf_pymupdf
from .docx_parser import DocxParser, parse_docx

__all__ = [
    # Vision Parser
    'VisionParser',
    'get_openai_config',

    # Docling Parser
    'DoclingParser',
    'parse_document_docling',

    # PyMuPDF Parser
    'PyMuPDFParser',
    'parse_pdf_pymupdf',

    # DOCX Parser
    'DocxParser',
    'parse_docx',
]
