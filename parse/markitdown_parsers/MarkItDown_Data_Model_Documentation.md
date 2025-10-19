# MarkItDown Data Model Documentation

## Overview

**MarkItDown** is a Python library developed by Microsoft for converting various document formats into Markdown. It is designed to preserve important document structure and content, making it particularly suitable for use with Large Language Models (LLMs) and text analysis pipelines.

**Official Repository**: [https://github.com/microsoft/markitdown](https://github.com/microsoft/markitdown)

## Table of Contents

- [Installation](#installation)
- [Core Result Object](#core-result-object)
- [Supported File Formats](#supported-file-formats)
- [Usage Examples](#usage-examples)
- [Optional Features](#optional-features)
- [Document Structure Preservation](#document-structure-preservation)
- [Best Practices](#best-practices)
- [Limitations](#limitations)
- [Advanced Features](#advanced-features)

---

## Installation

### Basic Installation

```bash
pip install markitdown
```

### Full Installation (All Optional Dependencies)

```bash
pip install 'markitdown[all]'
```

### Requirements

- **Python**: 3.10 or higher
- **Recommended**: Use a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install 'markitdown[all]'
```

---

## Core Result Object

The `MarkItDown.convert()` method returns a **DocumentConverterResult** object with the following attributes:

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `markdown` | `str` | The complete document converted to Markdown format |
| `title` | `str` or `None` | The document title, if extractable from the source file |
| `text_content` | `str` | Plain text content extracted from the document |

### Basic Structure

```python
class DocumentConverterResult:
    markdown: str        # Full Markdown conversion
    title: str | None    # Document title (if available)
    text_content: str    # Plain text version
```

### Example

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("document.pdf")

# Access the attributes
print(f"Title: {result.title}")
print(f"Markdown length: {len(result.markdown)} characters")
print(f"Text content length: {len(result.text_content)} characters")

# Display markdown content
print(result.markdown)
```

---

## Supported File Formats

MarkItDown supports conversion from a wide variety of file formats:

### Document Formats

| Format | Extensions | Description |
|--------|-----------|-------------|
| **PDF** | `.pdf` | Extracts text, tables, and images |
| **Microsoft Word** | `.docx`, `.doc` | Converts using Mammoth library to HTML, then to Markdown |
| **PowerPoint** | `.pptx` | Extracts slide content, preserving headings and bullet points |
| **Excel** | `.xlsx`, `.xls` | Converts spreadsheets to Markdown tables (multi-tab support) |

### Web & Structured Formats

| Format | Extensions | Description |
|--------|-----------|-------------|
| **HTML** | `.html`, `.htm` | Converts HTML to Markdown (special handling for Wikipedia) |
| **CSV** | `.csv` | Converts to Markdown tables (requires UTF-8 encoding) |
| **JSON** | `.json` | Structures JSON data in Markdown format |
| **XML** | `.xml`, `.rss`, `.atom` | Parses XML-based formats |

### Media Formats

| Format | Extensions | Description |
|--------|-----------|-------------|
| **Images** | `.jpg`, `.jpeg`, `.png` | Extracts EXIF metadata, optional OCR with EasyOCR |
| **Audio** | `.mp3`, `.wav`, etc. | Extracts metadata, transcribes speech using speech_recognition |

### Other Formats

| Format | Extensions | Description |
|--------|-----------|-------------|
| **EPUB** | `.epub` | Converts eBook content to Markdown |
| **ZIP Archives** | `.zip` | Iterates through contents, converts each file |
| **YouTube URLs** | URLs | Processes video content from YouTube links |

---

## Usage Examples

### 1. Command-Line Interface

#### Basic Conversion (Output to stdout)

```bash
markitdown document.pdf > output.md
```

#### Specify Output File

```bash
markitdown document.pdf -o output.md
```

#### Convert Multiple Files

```bash
markitdown file1.pdf file2.docx file3.xlsx
```

### 2. Python API - Basic Usage

```python
from markitdown import MarkItDown

# Initialize converter
md = MarkItDown()

# Convert a document
result = md.convert("document.docx")

# Access markdown content
print(result.markdown)

# Access title
if result.title:
    print(f"Document Title: {result.title}")

# Save to file
with open("output.md", "w", encoding="utf-8") as f:
    f.write(result.markdown)
```

### 3. Convert Different File Types

```python
from markitdown import MarkItDown

md = MarkItDown()

# PDF
pdf_result = md.convert("report.pdf")
print(pdf_result.markdown)

# Word Document
docx_result = md.convert("document.docx")
print(docx_result.markdown)

# Excel Spreadsheet
xlsx_result = md.convert("data.xlsx")
print(xlsx_result.markdown)

# PowerPoint
pptx_result = md.convert("presentation.pptx")
print(pptx_result.markdown)

# Image (with EXIF metadata)
img_result = md.convert("photo.jpg")
print(img_result.markdown)
```

### 4. Handle Title and Metadata

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("document.pdf")

# Use title or default to filename
title = result.title or "Untitled Document"
print(f"Processing: {title}")

# Create markdown with front matter
output = f"""---
title: {title}
source: document.pdf
---

{result.markdown}
"""

with open("output.md", "w", encoding="utf-8") as f:
    f.write(output)
```

---

## Optional Features

### 1. LLM Integration for Enhanced Image Descriptions

MarkItDown can integrate with Large Language Models (LLMs) to generate detailed descriptions of images within documents.

#### With OpenAI

```python
from markitdown import MarkItDown
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key")

# Initialize MarkItDown with LLM support
md = MarkItDown(llm_client=client, llm_model="gpt-4o")

# Convert document with images
result = md.convert("document_with_images.pdf")

# Images will have AI-generated descriptions
print(result.markdown)
```

#### Benefits of LLM Integration

- **Enhanced Image Descriptions**: Detailed, contextual descriptions of images, charts, and diagrams
- **Chart Understanding**: Interprets data visualizations and graphs
- **Visual Content Analysis**: Describes complex visual elements in text form
- **Accessibility**: Makes visual content accessible through text descriptions

### 2. Azure Document Intelligence Integration

For enhanced document processing capabilities:

```python
from markitdown import MarkItDown

# Initialize with Azure Document Intelligence endpoint
md = MarkItDown(docintel_endpoint="<your_azure_endpoint>")

result = md.convert("complex_document.pdf")
print(result.markdown)
```

### 3. Plugin System

MarkItDown supports third-party plugins:

```python
from markitdown import MarkItDown

# Enable plugins
md = MarkItDown(enable_plugins=True)

result = md.convert("document.pdf")
```

---

## Document Structure Preservation

MarkItDown is designed to preserve important document structure when converting to Markdown:

### Text Elements

#### Headers

Source formats with headings are converted to Markdown headers:

```markdown
# Heading 1
## Heading 2
### Heading 3
```

#### Paragraphs

Regular text is preserved as paragraphs with appropriate spacing.

#### Lists

- **Bulleted lists** are converted to Markdown unordered lists
- **Numbered lists** are converted to Markdown ordered lists

```markdown
- Item 1
- Item 2
  - Nested item

1. First item
2. Second item
```

#### Links

Hyperlinks are preserved in Markdown format:

```markdown
[Link Text](https://example.com)
```

### Tables

Excel spreadsheets and document tables are converted to Markdown table syntax:

```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
```

#### Multi-Sheet Excel Files

For Excel files with multiple sheets, MarkItDown typically outputs each sheet with headers:

```markdown
## Sheet 1

| Column A | Column B | Column C |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

## Sheet 2

| Column X | Column Y |
|----------|----------|
| Data X   | Data Y   |
```

### Images

Images are referenced in Markdown format:

```markdown
![Image Description](image_path_or_reference)
```

**With LLM Integration**: Image descriptions are enhanced with AI-generated contextual descriptions.

**With OCR**: Text within images can be extracted and included.

### Code and Formatting

- **Bold text**: `**bold**` or `__bold__`
- **Italic text**: `*italic*` or `_italic_`
- **Code blocks**: Preserved with appropriate syntax
- **Block quotes**: `> Quote text`

---

## Best Practices

### 1. File Format Selection

- **For Text-Heavy Documents**: PDF or DOCX work well
- **For Data**: Excel (XLSX) for structured tabular data
- **For Presentations**: PPTX maintains structure and hierarchy
- **For Web Content**: HTML with automatic Wikipedia optimization

### 2. Encoding and Character Sets

- Ensure **UTF-8 encoding** for CSV files
- MarkItDown handles Unicode characters in most formats
- Test with special characters if working with international content

### 3. Large Document Processing

```python
from markitdown import MarkItDown
import os

md = MarkItDown()

# Process in chunks or monitor memory
file_size = os.path.getsize("large_document.pdf")
if file_size > 50_000_000:  # 50 MB
    print("Warning: Large file detected. Processing may take time.")

result = md.convert("large_document.pdf")
```

### 4. Error Handling

```python
from markitdown import MarkItDown

md = MarkItDown()

try:
    result = md.convert("document.pdf")
    
    if result.markdown:
        print("Conversion successful")
    else:
        print("Warning: Empty markdown output")
        
except Exception as e:
    print(f"Conversion failed: {e}")
```

### 5. Batch Processing

```python
from markitdown import MarkItDown
from pathlib import Path

md = MarkItDown()

# Process multiple files
input_dir = Path("documents")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

for file_path in input_dir.glob("*.pdf"):
    try:
        result = md.convert(str(file_path))
        
        # Save with same name, .md extension
        output_path = output_dir / f"{file_path.stem}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)
            
        print(f"✓ Converted: {file_path.name}")
        
    except Exception as e:
        print(f"✗ Failed: {file_path.name} - {e}")
```

### 6. Quality Validation

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("document.pdf")

# Validate output
validation = {
    "has_content": len(result.markdown) > 0,
    "has_title": result.title is not None,
    "word_count": len(result.text_content.split()),
    "has_headers": "#" in result.markdown,
    "has_tables": "|" in result.markdown
}

print(f"Validation Results: {validation}")
```

---

## Limitations

### 1. Document Fidelity

- **Not for High-Fidelity Conversion**: MarkItDown is optimized for LLM consumption, not pixel-perfect document reproduction
- **Formatting Loss**: Complex formatting (colors, fonts, layouts) is not preserved
- **Visual Layout**: Page-specific layouts are converted to linear markdown structure

### 2. Table Limitations

- **Complex Tables**: Merged cells, nested tables may be simplified
- **Table Styling**: Colors, borders, and cell formatting are lost
- **Large Tables**: Very wide tables may be difficult to represent in Markdown

### 3. Image Handling

- **Without LLM**: Only image references and basic metadata
- **With LLM**: Good descriptions but requires API access and incurs costs
- **Image Extraction**: Images are referenced, not typically extracted as files
- **OCR**: Optional and requires EasyOCR installation

### 4. Format-Specific Limitations

#### Excel
- **Formulas**: Converted to calculated values, not formulas
- **Charts**: Converted to text descriptions (with LLM) or omitted
- **Cell Formatting**: Colors, borders, conditional formatting lost
- **Macros**: Not processed or preserved

#### PowerPoint
- **Animations**: Not preserved
- **Slide Transitions**: Not included
- **Speaker Notes**: May or may not be included
- **Complex Layouts**: Simplified to text and bullet points

#### PDF
- **Scanned PDFs**: Require OCR for text extraction
- **Complex Layouts**: Multi-column layouts may be linearized
- **Forms**: Interactive form fields become static text

### 5. Performance Considerations

- **Large Files**: May require significant processing time and memory
- **LLM Integration**: Adds latency and API costs for image processing
- **Concurrent Processing**: Not explicitly designed for parallel execution

---

## Advanced Features

### 1. Custom Document Processing Pipeline

```python
from markitdown import MarkItDown
import re

md = MarkItDown()

def clean_markdown(markdown_text):
    """Post-process markdown output"""
    # Remove excessive blank lines
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    
    # Add table of contents
    headers = re.findall(r'^(#{1,6})\s+(.+)$', markdown_text, re.MULTILINE)
    toc = "\n## Table of Contents\n\n"
    for level, title in headers:
        indent = "  " * (len(level) - 1)
        toc += f"{indent}- [{title}](#{title.lower().replace(' ', '-')})\n"
    
    return toc + "\n" + markdown_text

# Convert and clean
result = md.convert("document.pdf")
cleaned_markdown = clean_markdown(result.markdown)

with open("output.md", "w", encoding="utf-8") as f:
    f.write(cleaned_markdown)
```

### 2. Metadata Extraction

```python
from markitdown import MarkItDown
from datetime import datetime

md = MarkItDown()
result = md.convert("document.pdf")

# Create rich metadata
metadata = {
    "title": result.title or "Untitled",
    "processed_date": datetime.now().isoformat(),
    "word_count": len(result.text_content.split()),
    "character_count": len(result.text_content),
    "has_tables": "|" in result.markdown,
    "header_count": result.markdown.count("\n#")
}

print(f"Document Metadata: {metadata}")
```

### 3. Integration with LangChain

```python
from markitdown import MarkItDown
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Convert document
md = MarkItDown()
result = md.convert("document.pdf")

# Split into semantic chunks
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

splits = splitter.split_text(result.markdown)

for i, split in enumerate(splits):
    print(f"Chunk {i}:")
    print(f"Metadata: {split.metadata}")
    print(f"Content: {split.page_content[:100]}...")
    print()
```

### 4. Vector Database Integration

```python
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer
import chromadb

# Convert document
md = MarkItDown()
result = md.convert("document.pdf")

# Split into paragraphs
paragraphs = [p.strip() for p in result.markdown.split("\n\n") if p.strip()]

# Create embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(paragraphs)

# Store in ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="documents")

collection.add(
    documents=paragraphs,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(paragraphs))]
)

print(f"Stored {len(paragraphs)} chunks in vector database")
```

---

## Comparison with Other Tools

| Feature | MarkItDown | textract | pypandoc | mammoth |
|---------|------------|----------|----------|---------|
| **Multiple Formats** | ✓ Extensive | ✓ Good | ✓ Excellent | ✗ DOCX only |
| **Structure Preservation** | ✓✓ Excellent | ✓ Moderate | ✓✓ Excellent | ✓ Good |
| **LLM Integration** | ✓ Built-in | ✗ None | ✗ None | ✗ None |
| **Tables** | ✓ Markdown | ✓ Text | ✓ Various | ✓ HTML |
| **Images** | ✓ With LLM | ✗ Limited | ✓ References | ✗ Limited |
| **Dependencies** | Moderate | Heavy | External (pandoc) | Light |
| **Speed** | Fast | Moderate | Fast | Fast |
| **Use Case** | LLM pipelines | Text extraction | Format conversion | DOCX to HTML |

### When to Use MarkItDown

✓ **Best For:**
- LLM and RAG applications
- Document structure preservation
- Multi-format support with single API
- Integration with AI pipelines
- Clean, readable Markdown output

✗ **Not Ideal For:**
- High-fidelity document reproduction
- Preserving complex formatting
- Professional document conversion for human consumption
- Real-time streaming conversions

---

## Version Information

- **Library**: MarkItDown by Microsoft
- **Python Requirement**: 3.10+
- **GitHub**: [https://github.com/microsoft/markitdown](https://github.com/microsoft/markitdown)
- **License**: MIT

---

## Additional Resources

- **Official Repository**: [https://github.com/microsoft/markitdown](https://github.com/microsoft/markitdown)
- **PyPI Package**: [https://pypi.org/project/markitdown/](https://pypi.org/project/markitdown/)
- **Issue Tracker**: [https://github.com/microsoft/markitdown/issues](https://github.com/microsoft/markitdown/issues)

---

**Last Updated**: October 2024  
**Based On**: Official MarkItDown Documentation

