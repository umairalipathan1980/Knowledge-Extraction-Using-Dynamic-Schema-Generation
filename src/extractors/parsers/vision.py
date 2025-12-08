import os
import base64
from io import BytesIO
from typing import List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from PIL import Image
import fitz  # PyMuPDF for PDF processing

load_dotenv()


class VisionParser:
    """
    A PDF parser that converts PDFs to images and uses OpenAI Vision API
    (Azure or standard) to extract content in markdown format.
    """

    # Default prompt optimized for table preservation
    DEFAULT_TABLE_PROMPT = """
Convert this document page to accurate markdown format. Follow these rules STRICTLY:

**CRITICAL RULES:**
1. **NO HALLUCINATION**: Only output content that is actually visible on the page
2. **NO EMPTY ROWS**: Do NOT create empty table rows. If you see a table, only include rows with actual data
3. **STOP when content ends**: When you reach the end of visible content, STOP. Do not continue with empty rows

**Formatting Requirements:**
- Tables: Use markdown table syntax with | separators
- Multi-row cells: Keep item descriptions/notes in the same row as the item data
- Table continuations: If a table continues from a previous page, continue it without repeating headers
- Images: Interpret images, charts, graphs at at their proper places
- Preserve ALL visible text: headers, data, footers, page numbers, everything
- Keep numbers, dates, and text exactly as shown
- Maintain document structure and layout
- Keep the items at the same location as they are in the original document

**What to include:**
- All table data
- All section headings, text paragraphs
- Interpretation of all images (if any) 

Return ONLY the markdown content, no explanations.
"""

    def __init__(
        self,
        openai_config: dict,
        custom_prompt: str = None,
        use_context: bool = True,
        dpi: int = 300,
        clean_output: bool = True
    ):
        """
        Initialize the VisionParser with all configuration options.

        Args:
            openai_config: Dictionary containing OpenAI configuration
                For Azure:
                    - use_azure: True
                    - api_key: Azure OpenAI API key
                    - azure_endpoint: Azure OpenAI endpoint URL
                    - api_version: API version
                    - model: Deployment name
                For Standard OpenAI:
                    - use_azure: False
                    - api_key: OpenAI API key
                    - model: Model name (e.g., 'gpt-4o-2024-11-20')
            custom_prompt: Optional custom instructions for the vision model (uses DEFAULT_TABLE_PROMPT if None)
            use_context: Whether to provide previous page context for multi-page documents (default: True)
            dpi: Image resolution for PDF conversion (default: 300)
            clean_output: Enable LLM post-processing to clean and merge tables (default: True)
        """
        self.config = openai_config
        self.custom_prompt = custom_prompt or self._get_default_prompt()
        self.use_context = use_context
        self.dpi = dpi
        self.clean_output = clean_output
        self.use_azure = openai_config.get('use_azure', True)

        # Initialize appropriate OpenAI client
        if self.use_azure:
            # Extract base endpoint (remove deployment path if present)
            endpoint = openai_config['azure_endpoint']
            if '/openai/deployments/' in endpoint:
                # Extract base endpoint before /openai/deployments/
                endpoint = endpoint.split('/openai/deployments/')[0]

            # Initialize Azure OpenAI client
            self.client = AzureOpenAI(
                api_key=openai_config['api_key'],
                api_version=openai_config['api_version'],
                azure_endpoint=endpoint
            )
        else:
            # Initialize standard OpenAI client
            self.client = OpenAI(
                api_key=openai_config['api_key']
            )

        self.model = openai_config['model']

    def _get_default_prompt(self) -> str:
        """Get default prompt for markdown extraction."""
        return """
Please convert this document page to markdown format with the following requirements:

1. Preserve ALL content exactly as it appears
2. Maintain the document structure and hierarchy
3. For tables:
   - Use proper markdown table syntax with | separators
   - If this page continues a table from the previous page, continue the table seamlessly
   - Do NOT repeat table headers unless they appear on this page
   - Preserve multi-row cells by repeating content or using appropriate formatting
   - Maintain column alignment
   - Keep all headers and data intact
   - For item descriptions or notes within table cells, keep them in the same row
4. Preserve formatting like bold, italic, lists, etc.
5. For images or charts, describe them briefly in [Image: description] format
6. Maintain the reading order and layout flow
7. Keep numbers, dates, and special characters exactly as shown

Return ONLY the markdown content, no explanations.
"""

    def _pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF pages to images using PyMuPDF (fitz).
        Images are kept in memory as PIL Image objects.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image conversion (default: 300)

        Returns:
            List of PIL Image objects
        """
        print(f"Converting PDF to images using PyMuPDF (DPI: {dpi})...")
        images = []
        doc = fitz.open(pdf_path)  # Open the PDF

        # Calculate zoom factor from DPI (72 is the default DPI in PDFs)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            # Render page to pixmap with specified DPI
            pix = doc[page_num].get_pixmap(matrix=mat)
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        print(f"Converted {len(images)} pages to images")
        return images

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    ##compress before encoding
    # def _image_to_base64(self, image: Image.Image) -> str:
    #     buffered = BytesIO()
    #     # Reduce quality/size while preserving readability
    #     image = image.resize((int(image.width * 0.8), int(image.height * 0.8)), Image.LANCZOS)
    #     image.save(buffered, format="JPEG", quality=85, optimize=True)
    #     return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _parse_image_with_vision(self, image: Image.Image, page_num: int, previous_context: str = None) -> str:
        """
        Parse a single image using Azure OpenAI Vision API.

        Args:
            image: PIL Image object
            page_num: Page number (for logging)
            previous_context: Context from previous page(s) to help with continuations

        Returns:
            Markdown content extracted from the image
        """
        print(f"Parsing page {page_num} with vision model...")

        # Convert image to base64
        base64_image = self._image_to_base64(image)

        # Build the prompt with context if available
        prompt_text = self.custom_prompt
        if previous_context and self.use_context:
            prompt_text = f"""
{self.custom_prompt}

CONTEXT FROM PREVIOUS PAGE:
The previous page has the following content:
```
{previous_context}
```

If this page continues a table or section from the previous page, continue it seamlessly without repeating headers.
"""

        # Create the vision API request
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=16000,  # Increased to capture all content
            temperature=0
        )

        markdown_content = response.choices[0].message.content
        return markdown_content

    def _clean_markdown_with_llm(self, markdown_pages: List[str]) -> str:
        """
        Use LLM to clean up and merge markdown from multiple pages.

        Args:
            markdown_pages: List of markdown strings from each page

        Returns:
            Cleaned and merged markdown
        """
        print("Cleaning and merging markdown ...")

        # Combine pages with separators
        combined = "\n\n---PAGE_BREAK---\n\n".join(markdown_pages)

        cleanup_prompt = """
You are a document processing expert. Clean up and merge this multi-page markdown document.

TASKS:
1. **Remove artifacts**: Delete any empty table rows or hallucinated content (rows with only pipe separators and no data)
2. **Merge broken tables**: When a table continues across pages (separated by ---PAGE_BREAK---):
   - Keep only ONE table header
   - Merge all data rows into a single continuous table
   - Remove page break markers within tables
3. **Handle incomplete rows**: If a table row is split across pages, merge it into a complete row
4. **Preserve all real content**: Keep all actual data, headers, footers, and text
5. **Clean up formatting**: Ensure proper markdown syntax throughout
6. **Do NOT hallucinate**: Only output what you see in the input

INPUT MARKDOWN:
```markdown
{markdown}
```

OUTPUT: Return ONLY the cleaned, merged markdown. No explanations, no code blocks wrapper.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": cleanup_prompt.format(markdown=combined)
                }
            ],
            max_completion_tokens=16000,
            temperature=0
        )

        cleaned_markdown = response.choices[0].message.content

        # Remove any markdown code block wrappers if present
        if cleaned_markdown.startswith("```markdown"):
            cleaned_markdown = cleaned_markdown.replace("```markdown", "", 1)
        if cleaned_markdown.startswith("```"):
            cleaned_markdown = cleaned_markdown.replace("```", "", 1)
        if cleaned_markdown.endswith("```"):
            cleaned_markdown = cleaned_markdown.rsplit("```", 1)[0]

        return cleaned_markdown.strip()

    def convert_pdf(self, pdf_path: str) -> List[str]:
        """
        Convert PDF to markdown using vision API.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of markdown strings, one per page (or single cleaned string if clean_output=True)
        """
        # Convert PDF to images using configured DPI
        images = self._pdf_to_images(pdf_path, self.dpi)

        # Parse each image with context from previous page
        markdown_pages = []
        for i, image in enumerate(images, 1):
            # Get context from previous page if available
            previous_context = markdown_pages[-1] if markdown_pages and self.use_context else None

            # Parse with context
            markdown = self._parse_image_with_vision(image, i, previous_context)
            markdown_pages.append(markdown)

        # Post-process with LLM to clean up and merge if requested
        if self.clean_output and len(markdown_pages) > 1:
            cleaned = self._clean_markdown_with_llm(markdown_pages)
            return [cleaned]  # Return as single-item list for consistency

        return markdown_pages

    def save_markdown(self, markdown_pages: List[str], output_path: str,
                     separator: str = "\n\n---\n\n"):
        """
        Save markdown pages to a file.

        Args:
            markdown_pages: List of markdown strings
            output_path: Path to save the markdown file
            separator: Separator between pages (default: horizontal rule, only used if multiple pages)
        """
        # If only one page (e.g., already cleaned), save directly
        if len(markdown_pages) == 1:
            combined_markdown = markdown_pages[0]
        else:
            # Combine all pages with separator
            combined_markdown = separator.join(markdown_pages)

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_markdown)

        print(f"Markdown saved to: {output_path}")




