# -----------------------------------------------------------------------------
# EXAMPLE USAGE
# -----------------------------------------------------------------------------

from schema_generator import SchemaGenerator


if __name__ == "__main__":
    # Simple example showing how to use the SchemaGenerator class
    print("="*80)
    print("SCHEMA GENERATOR - EXAMPLE USAGE")
    print("="*80)

    # Example: Extract project information
    user_requirements = """
    Extract project information from research grant documents.
    For each project, extract:
    - Project title (string)
    - Project acronym (string)
    - Lead institution (string)
    - Total funding in EUR (number)
    - Start date (date)
    - Project status (enum: ongoing or completed)
    """

    sample_document = [
        """
        Project Report
        
        Title: Advanced AI Research Initiative
        Acronym: AIRI
        Lead Institution: University of Helsinki
        Total Funding: 2500000 euros
        Partners: Finland, Sweden, Norway, Denmark
        Status: Ongoing
        Start Date: 2024-01-15
        """,
        """
        Project Summary
        
        Project Name: Green Energy Solutions
        Short Name: GES
        Lead: Technical University of Munich
        Budget: 1800000 EUR
        Participating Countries: Germany, Austria, Switzerland
        Current Status: Completed
        Started: 2023-06-01
        """,
    ]
    # Create generator and extract
    # Use Azure OpenAI with default deployment 'gpt-4.1'
    generator = SchemaGenerator(use_azure=True)

    # Or use standard OpenAI:
    # generator = SchemaGenerator(use_azure=False)

    print("\nUsing SchemaGenerator class for extraction...")
    print("-" * 80)

    results = generator.extract(
        user_requirements=user_requirements,
        documents=sample_document  # Already a list, don't wrap again
    )

    # Display results
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)
    import json
    print(json.dumps(results, indent=2, default=str))

