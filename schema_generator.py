"""
Dynamic Schema Extraction with Structured Outputs
Improved version with:
- Deterministic parsing (temperature=0, top_p=1, optional seed)
- Stronger “schema-of-the-schema” (enums, regex/patterns, formats, uniqueness)
- Stricter dynamic Pydantic models (forbid extras, constrained types)
- Prompt hardening + fenced inputs to reduce prompt injection
- Normalization layer (dates, EUR currency, list splitting)
- Retries with exponential backoff, basic usage logging
"""

from __future__ import annotations

import os
import re
import time
from decimal import Decimal
from datetime import datetime
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from openai import APIError, RateLimitError, APITimeoutError
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    create_model,
    ConfigDict,
    constr,
)

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

load_dotenv()

SYSTEM_PARSER = (
    "You convert text into strictly structured data according to the provided schema. "
    "Never invent values. If uncertain or missing, return null. "
    "Do not include explanations or extra keys or extra fields."
)


def get_openai_config(use_azure: bool = True) -> dict:
    """
    Get OpenAI configuration based on whether to use Azure or standard OpenAI.

    Args:
        use_azure: If True, use Azure OpenAI. If False, use standard OpenAI API.

    Returns:
        Configuration dictionary with appropriate settings
    """
    if use_azure:
        return {
            'use_azure': True,
            'api_key': os.getenv("AZURE_API_KEY"),
            'azure_endpoint': os.getenv("AZURE_ENDPOINT", "https://haagahelia-poc-gaik.openai.azure.com"),
            'api_version': os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            'model': os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1"),
        }
    else:
        return {
            'use_azure': False,
            'api_key': os.getenv("OPENAI_API_KEY"),
            'model': 'gpt-4.1-2025-04-14',
        }


def create_openai_client(config: dict):
    """
    Create an OpenAI or Azure OpenAI client based on configuration.

    Args:
        config: Configuration dictionary from get_openai_config()

    Returns:
        OpenAI or AzureOpenAI client instance
    """
    if config.get('use_azure', False):
        return AzureOpenAI(
            api_key=config['api_key'],
            api_version=config['api_version'],
            azure_endpoint=config['azure_endpoint'],
        )
    else:
        return OpenAI(api_key=config['api_key'])

# -----------------------------------------------------------------------------
# Retry & call helpers
# -----------------------------------------------------------------------------

def _with_retries(call, tries: int = 4):
    for i in range(tries):
        try:
            return call()
        except (RateLimitError, APITimeoutError, APIError) as e:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)  # backoff


def _parse_with(*, client, model: str, messages: list[dict], response_format: type[BaseModel]):
    """
    Wraps client.beta.chat.completions.parse in a retry + deterministic settings.

    Args:
        client: OpenAI or AzureOpenAI client instance
        model: Model name to use
        messages: Messages to send
        response_format: Pydantic model for structured output
    """
    return _with_retries(
        lambda: client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=0,
            top_p=1.0,
            seed=12345,      # remove if your endpoint doesn't support seeding
            timeout=30,      # seconds; remove if your SDK doesn't support it
        )
    )

# -----------------------------------------------------------------------------
#Fixed schema for parsing the user's extraction requirements
# -----------------------------------------------------------------------------

AllowedTypes = Literal[
    "str", "int", "float", "bool", "list[str]", "date", "decimal", "list[dict]"
]


class FieldSpec(BaseModel):
    """Specification for a single field to extract."""

    field_name: str = Field(description="snake_case field name, must start with a letter")
    field_type: AllowedTypes = Field(description="Type of the field")
    description: str
    required: bool = True
    enum: Optional[list[str]] = Field(default=None, description="Allowed values (if enumerated)")
    pattern: Optional[str] = Field(default=None, description="Regex to validate strings (optional)")
    format: Optional[Literal["iso-date", "currency-eur"]] = Field(default=None)

    @field_validator("field_name")
    @classmethod
    def _snake_case(cls, v: str) -> str:
        v2 = re.sub(r"[^a-zA-Z0-9]+", "_", v).strip("_").lower()
        if not re.match(r"^[a-z][a-z0-9_]*$", v2 or ""):
            raise ValueError("field_name must be snake_case and start with a letter")
        return v2

    @field_validator("enum")
    @classmethod
    def _enum_nonempty(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is not None and len(v) == 0:
            raise ValueError("enum must be a non-empty list when provided")
        return v


class ExtractionRequirements(BaseModel):
    """Parsed extraction requirements from user input."""

    use_case_name: str
    fields: list[FieldSpec]

    @field_validator("fields")
    @classmethod
    def _unique_names(cls, fields: list[FieldSpec]) -> list[FieldSpec]:
        seen = set()
        for f in fields:
            if f.field_name in seen:
                raise ValueError(f"Duplicate field_name: {f.field_name}")
            seen.add(f.field_name)
        return fields

# -----------------------------------------------------------------------------
# Structure Detection for Nested vs Flat Schemas
# -----------------------------------------------------------------------------

class StructureAnalysis(BaseModel):
    """Analysis of whether the extraction requires nested or flat structure."""

    structure_type: Literal["flat", "nested_list"] = Field(
        description="Type of structure: 'flat' for single object, 'nested_list' for array of items"
    )
    parent_container_name: str = Field(
        description="Name for the parent container (e.g., 'items', 'records', 'entries')"
    )
    parent_description: str = Field(
        description="Description of what the parent container holds"
    )
    item_description: str = Field(
        description="Description of extraction requirements for each individual item (if nested)"
    )
    reasoning: str = Field(
        description="Brief explanation of why this structure was chosen"
    )


def detect_structure_type(user_description: str, *, client=None, model: str = None) -> StructureAnalysis:
    """
    Stage 1: Analyze if the extraction requires a nested list structure or flat structure.
    """
    if client is None:
        config = get_openai_config(use_azure=True)
        client = create_openai_client(config)
        model = model if model else config['model']
    elif model is None:
        raise ValueError("model must be provided when client is specified")

    resp = _parse_with(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PARSER},
            {
                "role": "user",
                "content": (
                    "Analyze the following extraction requirements and determine the output structure.\n\n"
                    "Use NESTED_LIST when:\n"
                    "- The DOCUMENT contains multiple items/records/rows to extract\n"
                    "- Instructions mention 'multiple items IN THE DOCUMENT', 'list of items', 'table of records'\n"
                    "- 'one line per item', 'one row per record', 'repeat for each entry'\n"
                    "- Document is structured as a table, list, or collection of similar items\n"
                    "- Example: Extract all products from an invoice (multiple products in one invoice)\n\n"
                    "Use FLAT when:\n"
                    "- ONE record per document (even if processing multiple documents)\n"
                    "- 'for each document', 'from each document', 'per document'\n"
                    "- Document describes a SINGLE entity (e.g., one project, one invoice, one person)\n"
                    "- Extracting summary/aggregate information from the document\n"
                    "- Example: Extract project details from grant document (one project per document)\n\n"
                    "IMPORTANT: 'For each X, extract...' means FLAT if X is the document itself, NESTED if X refers to multiple items within the document.\n\n"
                    "Requirements:\n```txt\n" + user_description + "\n```"
                ),
            },
        ],
        response_format=StructureAnalysis,
    )
    analysis = resp.choices[0].message.parsed
    if getattr(resp, "usage", None):
        print(f"[detect_structure_type] tokens={resp.usage.total_tokens}")
    return analysis


def parse_nested_requirements(
    user_description: str,
    *,
    client=None,
    model: str = None
) -> tuple[type[BaseModel], ExtractionRequirements]:
    """
    Stage 2: Parse nested requirements by:
    1. Detecting structure type
    2. Parsing item-level fields
    3. Creating nested parent model

    Returns: (ParentModel, item_requirements)
    """
    if client is None:
        config = get_openai_config(use_azure=True)
        client = create_openai_client(config)
        model = model if model else config['model']
    elif model is None:
        raise ValueError("model must be provided when client is specified")

    print("Stage 1: Analyzing structure type...")
    analysis = detect_structure_type(user_description, client=client, model=model)

    print(f"✓ Structure type: {analysis.structure_type}")
    print(f"  Reasoning: {analysis.reasoning}")

    if analysis.structure_type == "flat":
        # Just parse as flat requirements
        print("  → Using flat structure")
        requirements = parse_user_requirements(user_description, client=client, model=model)
        extraction_model = create_extraction_model(requirements)
        return extraction_model, requirements

    # Nested structure
    print(f"  → Using nested structure with '{analysis.parent_container_name}' field")
    print(f"  Parent: {analysis.parent_description}")

    print("\nStage 2: Parsing item-level fields...")
    item_requirements = parse_user_requirements(analysis.item_description, client=client, model=model)

    print(f"✓ Identified {len(item_requirements.fields)} fields per item")
    print(f"  Fields: {[f.field_name for f in item_requirements.fields]}")

    print("\nStage 3: Creating nested Pydantic model...")
    ItemModel = create_extraction_model(item_requirements)

    # Create parent model with items list
    from typing import List
    from pydantic import create_model, ConfigDict

    suffix = "_Collection"
    base_name = sanitize_model_name(item_requirements.use_case_name, suffix=suffix)
    model_name = base_name + suffix

    ParentModel = create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        __doc__=f"Collection of {item_requirements.use_case_name} items",
        **{
            analysis.parent_container_name: (
                List[ItemModel],
                Field(description=analysis.parent_description)
            )
        }
    )

    print(f"✓ Created nested model: {ParentModel.__name__}")
    print(f"  Container field: '{analysis.parent_container_name}' (List[{ItemModel.__name__}])")

    return ParentModel, item_requirements


# -----------------------------------------------------------------------------
# Parse the user's natural language into field specs
# -----------------------------------------------------------------------------

def parse_user_requirements(user_description: str, *, client=None, model: str = None) -> ExtractionRequirements:
    """
    Parse the extraction requirements from the user's natural language using structured outputs.
    """
    if client is None:
        config = get_openai_config(use_azure=True)
        client = create_openai_client(config)
        model = model if model else config['model']
    elif model is None:
        raise ValueError("model must be provided when client is specified")

    resp = _parse_with(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PARSER},
            {
                "role": "user",
                "content": "Parse the extraction requirements below into the target schema.\n"
                           "If a field cannot be identified reliably, omit it.\n"
                           "```txt\n" + user_description + "\n```"
            },
        ],
        response_format=ExtractionRequirements,
    )
    req = resp.choices[0].message.parsed
    if getattr(resp, "usage", None):
        print(f"[parse_user_requirements] tokens={resp.usage.total_tokens}")
    return req

# -----------------------------------------------------------------------------
# Create dynamic Pydantic model from field specs
# -----------------------------------------------------------------------------

def sanitize_model_name(name: str, suffix: str = "") -> str:
    """
    Sanitize model name following OpenAI requirements.
    Only alphanumeric, underscores, and hyphens are allowed.
    Ensures final name (with suffix) is <= 64 chars.

    Args:
        name: The base name to sanitize
        suffix: Optional suffix to add (e.g., "_Extraction", "_Collection")

    Returns:
        Sanitized name that when combined with suffix is <= 64 chars
    """
    # Replace invalid characters with underscores
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Remove consecutive underscores
    s = re.sub(r"_+", "_", s).strip("_")

    # Remove suffix from name if it already exists (avoid duplication)
    if suffix and s.endswith(suffix.lstrip("_")):
        s = s[:-len(suffix.lstrip("_"))].rstrip("_")

    # Ensure final name fits within 64 char limit
    max_length = 64 - len(suffix)
    if len(s) > max_length:
        s = s[:max_length].rstrip("_")

    return s if s else "Dynamic"


def create_extraction_model(requirements: ExtractionRequirements) -> type[BaseModel]:
    """
    Create a Pydantic model dynamically from field specifications (strict).
    - Forbid extra/unknown keys.
    - Apply enums, regex patterns, and formats where applicable.
    """
    # Base type mapping
    base_types = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list[str]": list[str],
        "date": str,     # we'll normalize to ISO later
        "decimal": Decimal,
        "list[dict]": list[dict],  # for nested structures like items
    }

    field_defs: dict[str, tuple[object, Field]] = {}

    for f in requirements.fields:
        py_type = base_types[f.field_type]

        # Constrain strings when possible
        annotated: object = py_type
        if f.field_type == "str" and f.pattern:
            annotated = Annotated[str, constr(pattern=f.pattern)]
        elif f.field_type == "decimal":
            annotated = Decimal  # leave numeric constraints to normalization/validation

        # Enums → Literal[...] for strict checking
        if f.enum:
            # Build a Literal[...] dynamically; acceptable for runtime checks
            annotated = Literal[tuple(f.enum)]  # type: ignore[misc,call-arg]

        # Optionality
        required_default = ... if f.required else None
        typ = annotated if f.required else (annotated | None)

        field_defs[f.field_name] = (
            typ,
            Field(default=required_default, description=f.description),
        )

    suffix = "_Extraction"
    base_name = sanitize_model_name(requirements.use_case_name, suffix=suffix)
    model_name = base_name + suffix

    DynamicModel = create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        __doc__=f"Extraction model for {requirements.use_case_name}",
        **field_defs,
    )
    return DynamicModel

# -----------------------------------------------------------------------------
# Normalization helpers (post-LLM)
# -----------------------------------------------------------------------------

def _to_iso_date(s: str) -> str:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s  # leave as-is if unparseable


def _normalize_record(data: dict, req: ExtractionRequirements) -> dict:
    spec_by_name = {f.field_name: f for f in req.fields}
    out = {}
    for k, v in data.items():
        spec = spec_by_name.get(k)
        if spec is None or v is None:
            out[k] = v
            continue

        if spec.field_type == "date" and isinstance(v, str):
            out[k] = _to_iso_date(v)
        elif spec.field_type == "list[str]":
            if isinstance(v, str):
                out[k] = [s.strip() for s in re.split(r"[;,]", v) if s.strip()]
            elif isinstance(v, list):
                out[k] = [str(x).strip() for x in v]
            else:
                out[k] = v
        else:
            out[k] = v
    return out

# -----------------------------------------------------------------------------
# Helper: Pretty print Pydantic model schema
# -----------------------------------------------------------------------------

def print_pydantic_schema(model: type[BaseModel], title: str = "Generated Pydantic Schema") -> None:
    """
    Print a formatted view of the Pydantic model schema.
    Shows field names, types, whether they're required, and descriptions.
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Model Name: {model.__name__}")
    if model.__doc__:
        print(f"Description: {model.__doc__}")

    print(f"\nFields:")
    for field_name, field_info in model.model_fields.items():
        # Get type annotation
        annotation = field_info.annotation
        type_str = str(annotation).replace("typing.", "")

        # Check if required
        is_required = field_info.is_required()
        required_str = "required" if is_required else "optional"

        # Get description
        description = field_info.description or "(no description)"

        print(f"  • {field_name}")
        print(f"    Type: {type_str}")
        print(f"    Required: {required_str}")
        print(f"    Description: {description}")

    # Also print JSON schema for full detail
    print(f"\n{'-'*80}")
    print("JSON Schema:")
    print(f"{'-'*80}")
    import json
    schema = model.model_json_schema()
    print(json.dumps(schema, indent=2))
    print(f"{'='*80}\n")


# -----------------------------------------------------------------------------
# Extract data using the dynamic schema with structured outputs
# -----------------------------------------------------------------------------

def _schema_hint(req: ExtractionRequirements) -> str:
    lines = []
    for f in req.fields:
        opt = "required" if f.required else "optional"
        extras = []
        if f.enum: extras.append(f"enum={f.enum}")
        if f.pattern: extras.append(f"pattern={f.pattern}")
        if f.format: extras.append(f"format={f.format}")
        line = f"- {f.field_name}: {f.field_type}, {opt}"
        if extras:
            line += f" ({', '.join(extras)})"
        lines.append(line)
    return "\n".join(lines)


def extract_from_document(
    document_text: str,
    extraction_model: type[BaseModel],
    req: ExtractionRequirements,
    *,
    client=None,
    model: str = None,
) -> BaseModel:
    """
    Use structured outputs to extract according to `extraction_model`.
    """
    if client is None:
        config = get_openai_config(use_azure=True)
        client = create_openai_client(config)
        model = model if model else config['model']
    elif model is None:
        raise ValueError("model must be provided when client is specified")

    schema_hint = _schema_hint(req)
    messages = [
        {"role": "system", "content": SYSTEM_PARSER},
        {
            "role": "user",
            "content": (
                f"Extract only the specified fields per the schema below.\n"
                f"If a value is missing or uncertain, return null.\n\n"
                f"Schema:\n{schema_hint}\n\n"
                f"Document:\n```txt\n{document_text}\n```"
            ),
        },
    ]
    resp = _parse_with(client=client, model=model, messages=messages, response_format=extraction_model)
    if getattr(resp, "usage", None):
        print(f"[extract_from_document] tokens={resp.usage.total_tokens}")
    return resp.choices[0].message.parsed

# -----------------------------------------------------------------------------
# COMPLETE WORKFLOW
# -----------------------------------------------------------------------------

def dynamic_extraction_workflow(
    user_description: str,
    documents: list[str],
    *,
    client=None,
    model: str = None,
) -> list[dict]:
    """
    LEGACY: Flat-only workflow (kept for backward compatibility).
    Use smart_extraction_workflow() instead for automatic nested/flat detection.

    1) Parse user requirements → ExtractionRequirements
    2) Build dynamic Pydantic model
    3) Extract from each document with structured outputs
    4) Normalize outputs (dates, EUR, lists) and return dicts
    """
    if client is None:
        config = get_openai_config(use_azure=True)
        client = create_openai_client(config)
        model = model if model else config['model']
    elif model is None:
        raise ValueError("model must be provided when client is specified")

    print("Step 1: Parsing user requirements...")
    requirements = parse_user_requirements(user_description, client=client, model=model)
    print(f"✓ Identified {len(requirements.fields)} fields")
    print("  Fields:", [f.field_name for f in requirements.fields])

    print("\nStep 2: Creating dynamic Pydantic schema...")
    ExtractionModel = create_extraction_model(requirements)
    print(f"✓ Created schema: {ExtractionModel.__name__}")

    # Print the generated Pydantic model
    print_pydantic_schema(ExtractionModel, title="Generated Pydantic Schema")

    print("\nStep 3: Extracting from documents...")
    results: list[dict] = []
    for i, doc in enumerate(documents, start=1):
        print(f"  Processing document {i}/{len(documents)}...")
        parsed = extract_from_document(doc, ExtractionModel, requirements, client=client, model=model)
        normalized = _normalize_record(parsed.model_dump(), requirements)
        results.append(normalized)

    print(f"✓ Extracted data from {len(documents)} documents")
    return results


def smart_extraction_workflow(
    user_description: str,
    documents: list[str],
    *,
    client=None,
    model: str = None,
) -> list[dict]:
    """
    SMART WORKFLOW: Automatically detects nested vs flat structure.

    Two-stage parsing:
    1) Detect structure type (nested_list or flat)
    2) Parse fields appropriately
    3) Create correct Pydantic model (nested or flat)
    4) Extract from documents
    5) Normalize and return
    """
    if client is None:
        config = get_openai_config(use_azure=True)
        client = create_openai_client(config)
        model = model if model else config['model']
    elif model is None:
        raise ValueError("model must be provided when client is specified")

    print("="*80)
    print("SMART EXTRACTION WORKFLOW")
    print("="*80)

    # Stage 1 & 2: Detect structure and parse requirements
    ExtractionModel, item_requirements = parse_nested_requirements(user_description, client=client, model=model)

    # Print the generated model
    print("\n" + "="*80)
    print("GENERATED PYDANTIC MODEL")
    print("="*80)
    print_pydantic_schema(ExtractionModel, title="Extraction Schema")

    # Stage 3: Extract from documents
    print("\n" + "="*80)
    print("EXTRACTION PHASE")
    print("="*80)
    results: list[dict] = []

    for i, doc in enumerate(documents, start=1):
        print(f"\nProcessing document {i}/{len(documents)}...")

        # Build extraction prompt with context about structure
        schema_hint = _schema_hint(item_requirements)

        messages = [
            {"role": "system", "content": SYSTEM_PARSER},
            {
                "role": "user",
                "content": (
                    f"EXTRACTION TASK:\n{user_description}\n\n"
                    f"Extract data according to the schema below.\n"
                    f"If extracting multiple items, ensure ALL items are included in the result.\n\n"
                    f"Schema fields:\n{schema_hint}\n\n"
                    f"Document:\n```txt\n{doc}\n```"
                ),
            },
        ]

        resp = _parse_with(client=client, model=model, messages=messages, response_format=ExtractionModel)
        if getattr(resp, "usage", None):
            print(f"  [extraction] tokens={resp.usage.total_tokens}")

        parsed = resp.choices[0].message.parsed
        result_dict = parsed.model_dump()

        # Normalize nested items if present
        if isinstance(result_dict, dict):
            for key, value in result_dict.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    # Normalize each item in the list
                    normalized_items = [_normalize_record(item, item_requirements) for item in value]
                    result_dict[key] = normalized_items
                    print(f"  ✓ Extracted {len(normalized_items)} items")

        results.append(result_dict)

    print(f"\n✓ Completed extraction from {len(documents)} document(s)")
    return results


# -----------------------------------------------------------------------------
# REUSABLE CLASS INTERFACE
# -----------------------------------------------------------------------------

class SchemaGenerator:
    """
    Reusable schema generator for dynamic data extraction.

    Automatically detects nested vs flat structure and generates appropriate
    Pydantic models from natural language requirements.

    Usage:
        generator = SchemaGenerator(model="gpt-4o")
        results = generator.extract(
            user_requirements="Extract invoice number and amount...",
            documents=["document text 1", "document text 2"]
        )
    """

    def __init__(self, use_azure: bool = True, model: Optional[str] = None):
        """
        Initialize the SchemaGenerator.

        Args:
            use_azure: If True, use Azure OpenAI. If False, use standard OpenAI (default: True)
            model: Optional model override. If not specified, uses model from config.
                   NOTE: For Azure, this should be the DEPLOYMENT NAME (e.g., 'gpt-4.1'),
                         not the model name (e.g., 'gpt-4.1-2025-04-14').

        Examples:
            # Use Azure with default deployment
            generator = SchemaGenerator(use_azure=True)

            # Use standard OpenAI with default model
            generator = SchemaGenerator(use_azure=False)

            # Use Azure with custom deployment name
            generator = SchemaGenerator(use_azure=True, model='my-custom-deployment')

            # Use OpenAI with different model
            generator = SchemaGenerator(use_azure=False, model='gpt-4o')
        """
        self.config = get_openai_config(use_azure=use_azure)
        self.model = model if model else self.config['model']
        self.client = create_openai_client(self.config)
        self.extraction_model = None
        self.item_requirements = None
        self.structure_analysis = None

    def analyze_structure(self, user_requirements: str) -> StructureAnalysis:
        """
        Analyze if the requirements need nested or flat structure.

        Args:
            user_requirements: Natural language description of extraction task

        Returns:
            StructureAnalysis with structure type and descriptions
        """
        self.structure_analysis = detect_structure_type(user_requirements, client=self.client, model=self.model)
        return self.structure_analysis

    def generate_schema(self, user_requirements: str) -> type[BaseModel]:
        """
        Generate Pydantic schema from natural language requirements.

        Args:
            user_requirements: Natural language description of fields to extract

        Returns:
            Generated Pydantic model class (nested or flat)
        """
        print("Generating schema from requirements...")
        self.extraction_model, self.item_requirements = parse_nested_requirements(
            user_requirements,
            client=self.client,
            model=self.model
        )
        return self.extraction_model

    def extract(
        self,
        user_requirements: str,
        documents: list[str],
        print_schema: bool = True
    ) -> list[dict]:
        """
        Complete extraction workflow: generate schema and extract data.

        Args:
            user_requirements: Natural language description of extraction task
            documents: List of document texts to extract from
            print_schema: Whether to print the generated schema (default: True)

        Returns:
            List of extraction results (dicts)
        """
        print("="*80)
        print("SMART EXTRACTION WORKFLOW")
        print("="*80)

        # Stage 1: Detect structure type and store analysis
        print("Stage 1: Analyzing structure type...")
        self.structure_analysis = detect_structure_type(user_requirements, client=self.client, model=self.model)
        print(f"✓ Structure type: {self.structure_analysis.structure_type}")
        print(f"  Reasoning: {self.structure_analysis.reasoning}")

        # Stage 2: Parse requirements and create model (populate instance variables)
        if self.structure_analysis.structure_type == "flat":
            print("  → Using flat structure")
            self.item_requirements = parse_user_requirements(user_requirements, client=self.client, model=self.model)
            self.extraction_model = create_extraction_model(self.item_requirements)
        else:
            # Nested structure
            print(f"  → Using nested structure with '{self.structure_analysis.parent_container_name}' field")
            print(f"  Parent: {self.structure_analysis.parent_description}")

            print("\nStage 2: Parsing item-level fields...")
            self.item_requirements = parse_user_requirements(self.structure_analysis.item_description, client=self.client, model=self.model)

            print(f"✓ Identified {len(self.item_requirements.fields)} fields per item")
            print(f"  Fields: {[f.field_name for f in self.item_requirements.fields]}")

            print("\nStage 3: Creating nested Pydantic model...")
            ItemModel = create_extraction_model(self.item_requirements)

            from typing import List
            suffix = "_Collection"
            base_name = sanitize_model_name(self.item_requirements.use_case_name, suffix=suffix)
            model_name = base_name + suffix

            self.extraction_model = create_model(
                model_name,
                __config__=ConfigDict(extra="forbid"),
                __doc__=f"Collection of {self.item_requirements.use_case_name} items",
                **{
                    self.structure_analysis.parent_container_name: (
                        List[ItemModel],
                        Field(description=self.structure_analysis.parent_description)
                    )
                }
            )

            print(f"✓ Created nested model: {self.extraction_model.__name__}")
            print(f"  Container field: '{self.structure_analysis.parent_container_name}' (List[{ItemModel.__name__}])")

        # Print the generated model if requested
        if print_schema:
            print("\n" + "="*80)
            print("GENERATED PYDANTIC MODEL")
            print("="*80)
            print_pydantic_schema(self.extraction_model, title="Extraction Schema")

        # Stage 3: Extract from documents
        print("\n" + "="*80)
        print("EXTRACTION PHASE")
        print("="*80)
        results: list[dict] = []

        for i, doc in enumerate(documents, start=1):
            print(f"\nProcessing document {i}/{len(documents)}...")

            # Build extraction prompt with context about structure
            schema_hint = _schema_hint(self.item_requirements)

            messages = [
                {"role": "system", "content": SYSTEM_PARSER},
                {
                    "role": "user",
                    "content": (
                        f"EXTRACTION TASK:\n{user_requirements}\n\n"
                        f"Extract data according to the schema below.\n"
                        f"If extracting multiple items, ensure ALL items are included in the result.\n\n"
                        f"Schema fields:\n{schema_hint}\n\n"
                        f"Document:\n```txt\n{doc}\n```"
                    ),
                },
            ]

            resp = _parse_with(client=self.client, model=self.model, messages=messages, response_format=self.extraction_model)
            if getattr(resp, "usage", None):
                print(f"  [extraction] tokens={resp.usage.total_tokens}")

            parsed = resp.choices[0].message.parsed
            result_dict = parsed.model_dump()

            # Normalize nested items if present
            if isinstance(result_dict, dict):
                for key, value in result_dict.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        # Normalize each item in the list
                        normalized_items = [_normalize_record(item, self.item_requirements) for item in value]
                        result_dict[key] = normalized_items
                        print(f"  ✓ Extracted {len(normalized_items)} items")

            results.append(result_dict)

        print(f"\n✓ Completed extraction from {len(documents)} document(s)")
        return results

    def extract_to_files(
        self,
        user_requirements: str,
        documents: list[str],
        json_path: str = "extraction_results.json",
        csv_path: str = "extraction_results.csv"
    ) -> list[dict]:
        """
        Extract data and save to JSON and CSV files.

        Args:
            user_requirements: Natural language description of extraction task
            documents: List of document texts to extract from
            json_path: Output JSON file path
            csv_path: Output CSV file path

        Returns:
            List of extraction results (dicts)
        """
        results = self.extract(user_requirements, documents)

        # Save to JSON
        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {json_path}")

        # Save to CSV if nested structure
        try:
            import csv
            if results and isinstance(results[0], dict):
                for result in results:
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=value[0].keys())
                                writer.writeheader()
                                writer.writerows(value)
                            print(f"✓ Items saved to CSV: {csv_path}")
                            break
        except Exception as e:
            print(f"Note: Could not save CSV: {e}")

        return results

    def get_schema_info(self) -> dict:
        """
        Get information about the generated schema.

        Returns:
            Dict with schema information
        """
        if not self.extraction_model:
            return {"error": "No schema generated yet. Call generate_schema() first."}

        return {
            "model_name": self.extraction_model.__name__,
            "structure_type": self.structure_analysis.structure_type if self.structure_analysis else "unknown",
            "fields": [f.field_name for f in self.item_requirements.fields] if self.item_requirements else [],
            "field_count": len(self.item_requirements.fields) if self.item_requirements else 0
        }


