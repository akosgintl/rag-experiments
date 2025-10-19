# Docling DoclingDocument Model Documentation

## Overview

**Docling** is a document parsing library developed by IBM Research (DS4SD) that converts various document formats into structured, AI-ready representations. The `DoclingDocument` is the core data model that provides a unified and expressive representation of parsed documents.

**Official Resources**:
- [Docling Documentation](https://docling-project.github.io/docling/)
- [Docling GitHub Repository](https://github.com/DS4SD/docling)
- [Docling Core](https://github.com/DS4SD/docling-core)

## Table of Contents

- [Key Features](#key-features)
- [Supported Document Formats](#supported-document-formats)
- [DoclingDocument Structure](#doclingdocument-structure)
- [Element Types](#element-types)
- [Metadata](#metadata)
- [Bounding Boxes](#bounding-boxes)
- [Export and Serialization](#export-and-serialization)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Integration with AI Frameworks](#integration-with-ai-frameworks)
- [Best Practices](#best-practices)

---

## Key Features

The `DoclingDocument` provides several powerful features for document processing:

1. **Unified Representation**: Offers a consistent structure for diverse document types (PDF, DOCX, PPTX, XLSX, HTML, images, audio)

2. **Hierarchical Structure**: Captures the nested organization of documents, including sections, paragraphs, tables, and figures

3. **Rich Content Types**: Supports various elements like text, tables, images, code blocks, and formulas

4. **Structured Output**: Delivers chunked, labeled data optimized for Large Language Model (LLM) pipelines

5. **Advanced Parsing**: Extracts clean structures from complex layouts, including:
   - Multi-column formats
   - Embedded tables
   - Formulas and equations
   - Complex page layouts

6. **Metadata Inclusion**: Incorporates essential metadata including document properties and layout information

7. **Multiple Export Formats**: Supports exporting to Markdown, HTML, DocTags, and lossless JSON

8. **Provenance Tracking**: Maintains spatial information about where content appears in the original document

---

## Supported Document Formats

Docling can convert the following document formats into `DoclingDocument`:

| Format | File Extensions | Description |
|--------|----------------|-------------|
| **PDF** | `.pdf` | Portable Document Format (including scanned PDFs with OCR) |
| **Microsoft Word** | `.docx` | Word documents with text, tables, and images |
| **PowerPoint** | `.pptx` | PowerPoint presentations |
| **Excel** | `.xlsx` | Excel spreadsheets with multiple sheets |
| **HTML** | `.html`, `.htm` | Web pages and HTML documents |
| **Images** | `.jpg`, `.png`, etc. | Images with OCR support |
| **Audio** | `.mp3`, `.wav`, etc. | Audio files with automatic speech recognition (ASR) |

---

## DoclingDocument Structure

The `DoclingDocument` serves as a comprehensive container for all elements parsed from a source document. It encapsulates the document's hierarchical structure and provides a unified interface for accessing and manipulating document content.

### Core Components

The `DoclingDocument` is composed of several key components:

1. **Elements**: The building blocks of the document representing various content types
2. **Metadata**: Information about the document (title, author, language, etc.)
3. **Bounding Boxes**: Spatial coordinates defining element positions
4. **Hierarchical Organization**: Tree-like structure capturing document organization

---

## Element Types

Within a `DoclingDocument`, elements are categorized into specific types representing different aspects of the document's content:

### Text Elements

#### TextItem
Represents textual content, including paragraphs, headings, and inline text. This is the most common element type for body text.

**Properties**:
- Text content (sanitized and original)
- Formatting information
- Position in document hierarchy
- Provenance (page number and bounding box)

#### Headings and Sections
Represents document structure elements like titles and section headers, capturing the hierarchical organization of the document.

### Table Elements

#### TableItem
Represents tabular data, capturing the structure and content of tables within the document.

**Capabilities**:
- Preserves table structure (rows, columns, merged cells)
- Supports export to various formats
- Maintains cell relationships
- Includes header identification

**Export Methods**:
- Export to pandas DataFrame for data analysis
- Export to Markdown format
- Export to HTML format

### Image Elements

#### ImageItem (PictureItem)
Represents images embedded in the document, including their position and associated metadata.

**Properties**:
- Image data and format
- Bounding box coordinates
- Associated captions
- References and footnotes

### List Elements

#### ListItem
Represents both ordered and unordered lists along with their items.

**Properties**:
- List item text
- Enumeration status (numbered vs. bulleted)
- Marker symbol
- Nesting level

### Code Elements

#### CodeItem
Represents code blocks, preserving the formatting and content of code snippets.

**Properties**:
- Code content
- Programming language (if detected)
- Syntax preservation
- Associated captions

### Figure Elements

#### FigureItem
Represents figures, including diagrams, charts, and other visual elements, along with their captions.

**Properties**:
- Figure image data
- Caption text
- References and footnotes
- Spatial information

### Formula Elements

Represents mathematical formulas and equations, preserving their structure and notation.

---

## Metadata

Metadata in a `DoclingDocument` provides contextual information about the document:

| Metadata Field | Description |
|----------------|-------------|
| **Title** | The title of the document |
| **Author** | The author(s) of the document |
| **Language** | The language in which the document is written |
| **Creation Date** | The date when the document was created |
| **Modification Date** | The date when the document was last modified |
| **Document Properties** | Additional properties from the source format |

This metadata is crucial for organizing, searching, and managing documents effectively.

---

## Bounding Boxes

Bounding boxes define the spatial properties of elements within the document. They are essential for:

- Understanding document layout
- Spatial analysis of content
- Extracting content from specific regions
- Reconstructing visual appearance
- Maintaining provenance information

**Bounding Box Properties**:
- Position coordinates (left, top, right, bottom)
- Page number association
- Coordinate system (typically top-left origin)

---

## Export and Serialization

The `DoclingDocument` provides multiple export formats to facilitate interoperability and further processing:

### 1. Markdown Export

Converts the document content into Markdown format, ideal for:
- Documentation and web content
- LLM-friendly text representation
- Human-readable output
- Integration with Markdown-compatible tools

```python
markdown_output = doc.export_to_markdown()
```

**Features**:
- Preserves document structure (headings, lists, tables)
- Includes table formatting
- Handles images with references or placeholders
- Maintains text formatting (bold, italic)

### 2. HTML Export

Generates an HTML representation suitable for:
- Web-based applications
- Browser rendering
- Rich formatting preservation

**Features**:
- Full HTML structure
- CSS-compatible formatting
- Image embedding or referencing
- Table styling

### 3. DocTags Export

A format designed for representing document structures in a tag-based system, optimized for:
- Document analysis pipelines
- Structured data extraction
- Custom processing workflows

### 4. JSON Export

Provides a lossless JSON serialization that:
- Preserves all structural and content details
- Enables data storage and exchange
- Maintains complete document information
- Supports custom processing

**Use Cases**:
- Archiving parsed documents
- Data interchange between systems
- Database storage
- Further programmatic processing

---

## Usage Examples

### Example 1: Basic Document Conversion

Convert a document and access its `DoclingDocument` representation:

```python
from docling.document_converter import DocumentConverter

# Convert document
source = "document.pdf"  # Can be file path or URL
converter = DocumentConverter()
result = converter.convert(source)
doc = result.document

# Access document information
print(f"Document: {doc.name}")
print(f"Number of pages: {len(doc.pages)}")
```

### Example 2: Export to Markdown

Convert a document and export its content to Markdown:

```python
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"  # Can use URLs
converter = DocumentConverter()
doc = converter.convert(source).document

# Export to Markdown
markdown_text = doc.export_to_markdown()
print(markdown_text)  # Output: "### Docling Technical Report[...]"

# Save to file
with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)
```

### Example 3: Working with Tables

Access and export table data:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
doc = converter.convert("spreadsheet.xlsx").document

# Access tables
for table in doc.tables:
    # Export to pandas DataFrame
    df = table.export_to_dataframe()
    print(f"Table shape: {df.shape}")
    print(df.head())
    
    # Export to Markdown
    table_md = table.export_to_markdown()
    print(table_md)
    
    # Export to HTML
    table_html = table.export_to_html()
    print(table_html)
```

### Example 4: Accessing Images

Extract images from a document:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
doc = converter.convert("document.pdf").document

# Access images
for i, image in enumerate(doc.pictures):
    # Get PIL Image
    pil_image = image.get_image(doc)
    if pil_image:
        pil_image.save(f"image_{i}.png")
    
    # Get caption
    caption = image.caption_text(doc)
    if caption:
        print(f"Image {i} caption: {caption}")
```

### Example 5: Iterating Through Content

Iterate through document elements:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
doc = converter.convert("document.pdf").document

# Iterate through all items
for item, level in doc.iterate_items():
    element_type = type(item).__name__
    print(f"{'  ' * level}{element_type}")
    
    # Access text content if available
    if hasattr(item, 'text'):
        print(f"{'  ' * level}  Content: {item.text[:50]}...")
```

### Example 6: Page-by-Page Processing

Process document by pages:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
doc = converter.convert("document.pdf").document

# Process each page
for page_no, page in doc.pages.items():
    print(f"\nPage {page_no}:")
    print(f"  Size: {page.size.width} x {page.size.height}")
    
    # Get content from this page
    for item, level in doc.iterate_items(page_no=page_no):
        if hasattr(item, 'text'):
            print(f"  {item.text[:50]}...")
```

### Example 7: Batch Processing

Process multiple documents:

```python
from docling.document_converter import DocumentConverter
from pathlib import Path

converter = DocumentConverter()
input_dir = Path("documents")

for pdf_file in input_dir.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")
    
    doc = converter.convert(str(pdf_file)).document
    
    # Export to Markdown
    output_path = pdf_file.with_suffix(".md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())
    
    print(f"  Saved to {output_path}")
```

---

## Advanced Features

### 1. OCR Support

Docling enables optical character recognition for scanned PDFs and images, extracting text from non-digital documents.

**Capabilities**:
- Text extraction from scanned pages
- Layout preservation
- Multi-language support
- Integration with OCR engines

### 2. Visual Language Model Integration

Supports visual language models like **GraniteDocling**, allowing for:
- Processing of complex layouts
- Understanding visual elements
- Enhanced image and chart interpretation
- Multi-modal document understanding

### 3. Audio Processing

Incorporates automatic speech recognition (ASR) models to process audio files:
- Converts spoken content into text
- Includes in document structure
- Preserves timing information
- Supports multiple audio formats

### 4. Complex Layout Handling

Advanced parsing capabilities for:
- **Multi-column layouts**: Accurately extracts content from multi-column documents
- **Nested structures**: Handles nested tables, lists, and sections
- **Mixed content**: Processes documents with combined text, tables, and images
- **Non-standard layouts**: Adapts to various document structures

### 5. Formula and Equation Support

Extracts and preserves:
- Mathematical formulas
- Chemical equations
- Special notation
- LaTeX-compatible representations

---

## Integration with AI Frameworks

The `DoclingDocument` is designed to integrate seamlessly with popular AI frameworks:

### LangChain Integration

```python
from docling.document_converter import DocumentConverter
from langchain.text_splitters import RecursiveCharacterTextSplitter

# Convert document
converter = DocumentConverter()
doc = converter.convert("document.pdf").document

# Get markdown for LangChain
markdown_text = doc.export_to_markdown()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_text(markdown_text)
```

### LlamaIndex Integration

```python
from docling.document_converter import DocumentConverter
from llama_index.core import Document

# Convert with Docling
converter = DocumentConverter()
doc = converter.convert("document.pdf").document

# Create LlamaIndex Document
text = doc.export_to_markdown()
llama_doc = Document(text=text, metadata={"source": "document.pdf"})
```

### Haystack Integration

Compatible with Haystack pipelines for:
- Document retrieval
- Question answering
- Semantic search
- RAG (Retrieval-Augmented Generation) applications

---

## Best Practices

### 1. Document Format Selection

- **For Scanned Documents**: Use PDF with OCR enabled
- **For Structured Data**: Excel or tables in PDF/DOCX work well
- **For Presentations**: PPTX maintains structure and hierarchy
- **For Web Content**: HTML preserves links and structure

### 2. Memory Management

```python
# For large documents, process in batches or by page
for page_no in doc.pages.keys():
    items = list(doc.iterate_items(page_no=page_no))
    process_page(items)
    # Page items can be garbage collected
```

### 3. Export Format Selection

- **Markdown**: Best for LLM consumption and human readability
- **JSON**: Best for storage and data interchange
- **HTML**: Best for web display and rich formatting
- **DataFrame**: Best for tabular data analysis

### 4. Error Handling

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

try:
    result = converter.convert("document.pdf")
    doc = result.document
    
    if doc:
        markdown = doc.export_to_markdown()
        print("Conversion successful")
    else:
        print("Warning: Empty document")
        
except Exception as e:
    print(f"Conversion failed: {e}")
```

### 5. Performance Optimization

- Use appropriate conversion settings for document type
- Enable OCR only when necessary
- Process large documents page-by-page
- Cache converted documents when reusing

### 6. Quality Validation

```python
# Validate conversion quality
doc = converter.convert("document.pdf").document

validation = {
    "has_content": len(doc.texts) > 0,
    "has_tables": len(doc.tables) > 0,
    "has_images": len(doc.pictures) > 0,
    "page_count": len(doc.pages),
}

print(f"Validation: {validation}")
```

---

## Comparison with Other Tools

| Feature | Docling | MarkItDown | pypdf | PyMuPDF |
|---------|---------|------------|-------|---------|
| **Multi-format Support** | ✓✓ Extensive | ✓ Good | ✗ PDF only | ✗ PDF only |
| **Structure Preservation** | ✓✓ Excellent | ✓ Good | ✓ Basic | ✓ Good |
| **Table Extraction** | ✓✓ Advanced | ✓ Basic | ✗ Limited | ✓ Good |
| **Image Extraction** | ✓✓ Advanced | ✓ Basic | ✓ Basic | ✓✓ Excellent |
| **OCR Support** | ✓ Built-in | ✗ No | ✗ No | ✗ No |
| **AI Integration** | ✓✓ Excellent | ✓ Good | ✗ Limited | ✗ Limited |
| **Complex Layouts** | ✓✓ Excellent | ✓ Good | ✗ Poor | ✓ Good |
| **Export Formats** | ✓✓ Multiple | ✓ Markdown | ✗ Text only | ✓ Text/HTML |

### When to Use Docling

✓ **Best For**:
- RAG and LLM applications requiring structured content
- Complex document layouts (multi-column, tables, formulas)
- Documents requiring OCR
- Multi-format document processing
- AI-powered document understanding
- Production document processing pipelines

✗ **Consider Alternatives For**:
- Simple text extraction only
- Real-time streaming processing
- Minimal dependency requirements
- Very basic PDF parsing

---

## Limitations and Considerations

### 1. Processing Time
- Complex layouts require more processing time
- OCR adds significant overhead for scanned documents
- Visual language models increase processing time but improve accuracy

### 2. Document Fidelity
- Optimized for content extraction, not pixel-perfect reproduction
- Some formatting details may be simplified
- Focus on semantic content over visual appearance

### 3. Language Support
- Primary focus on English and common European languages
- OCR quality varies by language and font
- Some special characters may require attention

### 4. File Size
- Large PDF files may require significant memory
- Consider page-by-page processing for very large documents
- Optimize images before processing if possible

---

## Version and Compatibility

- **Python**: 3.9 or higher recommended
- **Core Library**: docling-core
- **Main Package**: docling

### Installation

```bash
# Basic installation
pip install docling

# With all optional dependencies
pip install "docling[all]"

# For OCR support
pip install "docling[ocr]"
```

---

## Additional Resources

### Official Documentation
- [Docling Documentation](https://docling-project.github.io/docling/)
- [Docling Document Reference](https://docling-project.github.io/docling/reference/docling_document/)
- [Getting Started Guide](https://docling-project.github.io/docling/getting_started/)

### Source Code
- [Docling GitHub](https://github.com/DS4SD/docling)
- [Docling Core](https://github.com/DS4SD/docling-core)

### Community
- [Issue Tracker](https://github.com/DS4SD/docling/issues)
- [Discussions](https://github.com/DS4SD/docling/discussions)

---

## Summary

The `DoclingDocument` provides:

1. **Unified Document Representation**: Consistent structure across multiple formats
2. **Hierarchical Organization**: Tree-like structure capturing document organization
3. **Rich Content Support**: Text, tables, images, formulas, and more
4. **AI-Ready Output**: Optimized for LLM and RAG applications
5. **Multiple Export Formats**: Markdown, HTML, JSON, and more
6. **Advanced Parsing**: Handles complex layouts, OCR, and visual understanding
7. **Framework Integration**: Works seamlessly with LangChain, LlamaIndex, Haystack

This makes Docling ideal for:
- Document parsing and analysis
- RAG (Retrieval-Augmented Generation) systems
- Document conversion pipelines
- Content extraction and indexing
- Multi-modal document processing
- AI-powered document understanding

---

**Last Updated**: October 2024  
**Based On**: Official Docling Documentation  
**Version**: Compatible with Docling 1.0+
