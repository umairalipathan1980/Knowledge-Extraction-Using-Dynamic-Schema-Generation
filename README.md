# Knowledge Extraction Using Dynamic Schema Generation

A document data extraction system that automatically generates Pydantic schemas from natural language requirements and extracts structured data from PDFs and text documents.

## Features

- **Dynamic Schema Generation**: Describe what you want to extract in plain English
- **Smart Structure Detection**: Automatically detects flat vs nested data structures
- **PDF Processing**: Convert PDFs to markdown with vision models
- **Multi-Provider Support**: Works with Azure OpenAI or OpenAI
- **Type Safety**: Generates strict Pydantic models with validation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with your credentials:

```env
# Azure OpenAI
AZURE_API_KEY=your_azure_key_here

# Standard OpenAI (alternative)
OPENAI_API_KEY=your_openai_key_here
```

## Quick Start

### Simple Extraction

```python
from schema_generator import SchemaGenerator

# Initialize (defaults to Azure OpenAI)
generator = SchemaGenerator(use_azure=True)

# Define what to extract
requirements = """
Extract project information:
- Project title (string)
- Budget (number)
- Start date (date)
- Status (enum: active or completed)
"""

# Extract from documents
results = generator.extract(
    user_requirements=requirements,
    documents=["your document text here"]
)
```

### PDF Extraction

```python
from parsers.vision_parser import VisionParser, get_openai_config

# Initialize parser
config = get_openai_config(use_azure=True)
parser = VisionParser(
    openai_config=config,
    use_context=True,
    dpi=300,
    clean_output=True
)

# Parse PDF
markdown_pages = parser.convert_pdf("input/document.pdf")
```

### PO+BOM Extraction

```python
from extractor import main, extract_po_bom_data

# Step 1: Parse PDFs and classify documents
combined_path = main()

# Step 2: Extract and match PO+BOM data
results = extract_po_bom_data(combined_path)
```

## API Providers

### Azure OpenAI (Default)

```python
generator = SchemaGenerator(use_azure=True)
# Uses deployment: 'gpt-4.1'
```

### OpenAI

```python
generator = SchemaGenerator(use_azure=False)
# Uses model: 'gpt-4.1-2025-04-14'
```

### Custom Configuration

```python
# Azure with custom deployment
generator = SchemaGenerator(use_azure=True, model='my-deployment')

# OpenAI with different model
generator = SchemaGenerator(use_azure=False, model='gpt-4.1-2025-04-14')
```

## Project Structure

```
extractors/
├── extractor.py              # Main PO+BOM extraction workflow
├── schema_generator.py       # Core dynamic schema extraction
├── extraction_example.py     # Simple usage example
├── parsers/
│   ├── vision_parser.py      # PDF → Markdown conversion
│   ├── docling_parser.py     # Alternative parser
│   └── pymupdf_parser.py     # PyMuPDF-based parser
└── input/                    # Input PDFs
```

## Key Concepts

### User Requirements

Describe extraction needs in plain English:

```python
requirements = """
For each invoice item, extract:
- Item number (string)
- Description (string)
- Quantity (integer)
- Unit price (decimal)
- Total (decimal)
"""
```

### Structure Detection

The system automatically detects:
- **Flat**: One record per document (e.g., invoice header)
- **Nested**: Multiple items per document (e.g., invoice line items)


## Advanced Features

### Schema Validation

```python
# Automatic field validation
requirements = """
- Email (string, pattern: ^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$)
- Status (enum: draft, pending, approved, rejected)
- Amount (decimal)
- Date (date, format: iso-date)
"""
```

## Examples

See:
- `extraction_example.py` - Basic extraction workflow
- `extractor.py` - Complete PO+BOM matching pipeline

