# Extraction Workflow

```mermaid
flowchart TD
    A[User Requirements<br/>+ Documents] --> B{Detect Structure}

    B -->|Flat| C1[Generate Single Model]
    B -->|Nested| C2[Generate Parent + Item Models]

    C1 --> D1[Parse Field Specs]
    C2 --> D2[Parse Parent Fields]
    C2 --> D3[Parse Item Fields]

    D1 --> E1[Create Pydantic Model]
    D2 --> E2[Create Parent Model]
    D3 --> E3[Create Item Model]

    E1 --> F[Extract Data<br/>with LLM]
    E2 --> F
    E3 --> F

    F --> G[Validate with Pydantic]
    G --> H[Save JSON/CSV]

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style F fill:#ffe1f5
    style H fill:#e1ffe1
```

## Key Components

### Phase 1: Structure Detection
- **Input:** User requirements text
- **Process:** LLM analyzes if data is flat or nested
- **Output:** "flat" or "nested_list"

### Phase 2: Schema Generation
- **Parse Specifications:** Extract field names, types, constraints
- **Build Pydantic Models:** Dynamic model creation with validation rules
- **Handle Relationships:** Parent-child for nested structures

### Phase 3: Data Extraction
- **Structured Output:** Uses OpenAI's Pydantic response format
- **Deterministic:** Temperature=0, seed for reproducibility
- **Retry Logic:** Exponential backoff for failures

### Phase 4: Output
- **JSON:** Full structured data with metadata
- **CSV:** Flattened tabular format (optional)
