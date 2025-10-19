# Unstructured.io Elements Data Model Documentation

## Overview

**Unstructured.io** is a platform and open-source library for ingesting and processing unstructured documents, transforming them into structured data suitable for retrieval-augmented generation (RAG) and agentic AI applications.

The core output of Unstructured.io is a list of **Elements** - discrete components representing meaningful segments of the document - each enriched with **Metadata** that provides additional context.

**Official Resources**:
- [Unstructured.io Documentation](https://docs.unstructured.io/)
- [Unstructured GitHub Repository](https://github.com/Unstructured-IO/unstructured)
- [API Reference](https://docs.unstructured.io/api-reference/general/summary)

## Table of Contents

- [Key Features](#key-features)
- [Supported File Types](#supported-file-types)
- [Document Elements](#document-elements)
- [Element Types](#element-types)
- [Element Structure](#element-structure)
- [Metadata Fields](#metadata-fields)
- [VLM Generated HTML Elements](#vlm-generated-html-elements)
- [Data Connector Metadata](#data-connector-metadata)
- [Processing Strategies](#processing-strategies)
- [Chunking Strategy](#chunking-strategy)
- [API Parameters](#api-parameters)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

---

## Key Features

1. **Element-Based Representation**: Decomposes documents into discrete, typed elements
2. **Rich Metadata**: Each element includes contextual information (page numbers, coordinates, source info)
3. **Multi-Format Support**: Processes PDF, DOCX, XLSX, HTML, images, and more
4. **Vision Language Model (VLM) Integration**: Advanced visual understanding for complex layouts
5. **Flexible Processing Strategies**: Fast, hi_res, auto, and ocr_only modes
6. **HTML Representation**: Tables and structured content available as HTML
7. **Chunking Support**: Built-in chunking strategies for RAG applications
8. **API and Open Source**: Available as SaaS API or self-hosted

---

## Supported File Types

Unstructured.io supports a comprehensive range of file formats:

### Plaintext Formats
- `.eml` - Email messages
- `.html`, `.htm` - Web pages
- `.json` - JSON documents
- `.md` - Markdown files
- `.msg` - Outlook messages
- `.rst` - reStructuredText
- `.rtf` - Rich Text Format
- `.txt` - Plain text
- `.xml` - XML documents

### Image Formats
- `.png` - Portable Network Graphics
- `.jpg`, `.jpeg` - JPEG images
- `.tiff` - Tagged Image File Format
- `.bmp` - Bitmap images
- `.heic` - High Efficiency Image Format

### Document Formats
- `.csv` - Comma-separated values
- `.doc`, `.docx` - Microsoft Word
- `.epub` - Electronic publication
- `.odt` - OpenDocument Text
- `.pdf` - Portable Document Format
- `.ppt`, `.pptx` - Microsoft PowerPoint
- `.tsv` - Tab-separated values
- `.xlsx` - Microsoft Excel

---

## Document Elements

In the Unstructured framework, documents are decomposed into **elements**. Each element represents a meaningful segment of the document with a specific semantic purpose.

### Element Characteristics

- **Type Classification**: Each element has a specific type (Title, NarrativeText, Table, etc.)
- **Text Content**: Extracted text content of the element
- **Unique Identifier**: Each element has a unique `element_id`
- **Metadata**: Contextual information about the element
- **Hierarchical Relationships**: Elements can have parent-child relationships

---

## Element Types

Unstructured.io categorizes document content into various element types:

### Text Elements

#### Title
Represents headings and titles within the document.

**Characteristics**:
- Used for document headings and section titles
- Typically larger or bold text
- Structural significance in document hierarchy

**Example**:
```json
{
    "type": "Title",
    "element_id": "abc123...",
    "text": "Chapter 1: Introduction",
    "metadata": { ... }
}
```

#### NarrativeText
Denotes paragraphs or continuous blocks of text - the main body content.

**Characteristics**:
- Main document content
- Paragraphs and prose
- Multiple sentences of continuous text

**Example**:
```json
{
    "type": "NarrativeText",
    "element_id": "def456...",
    "text": "This section describes the methodology used...",
    "metadata": { ... }
}
```

#### ListItem
Represents individual items within ordered or unordered lists.

**Characteristics**:
- Bulleted or numbered list items
- Part of a larger list structure
- May include list markers

**Example**:
```json
{
    "type": "ListItem",
    "element_id": "ghi789...",
    "text": "First item in the list",
    "metadata": { ... }
}
```

#### UncategorizedText
Text that doesn't fit into predefined categories.

**Characteristics**:
- Miscellaneous text elements
- Fallback category for unclassified text
- May include annotations, notes, or unclear text

#### Other Text Element Types

Unstructured.io may also identify:
- **Header**: Page headers
- **Footer**: Page footers
- **FigureCaption**: Captions for figures and images
- **PageNumber**: Page number elements
- **Address**: Physical or mailing addresses
- **EmailAddress**: Email addresses
- **CodeSnippet**: Code blocks
- **Formula**: Mathematical formulas and equations

### Structured Elements

#### Table
Represents tabular data with rows and columns.

**Characteristics**:
- Structured data in table format
- Includes HTML representation in metadata
- Can be exported to DataFrame format
- Preserves table structure and relationships

**Example**:
```json
{
    "type": "Table",
    "element_id": "jkl012...",
    "text": "Product | Quantity | Price\\nOffice Desk | 2 | $249\\n...",
    "metadata": {
        "text_as_html": "<table><tr><th>Product</th>...</table>",
        "page_number": 3
    }
}
```

#### Image
Represents images, figures, charts, and diagrams embedded in documents.

**Characteristics**:
- Visual content elements
- May include extracted text (OCR)
- Can include caption text
- May have associated image data

**Example**:
```json
{
    "type": "Image",
    "element_id": "mno345...",
    "text": "Figure 1: Sales Distribution",
    "metadata": {
        "page_number": 5,
        "coordinates": {...}
    }
}
```

---

## Element Structure

Each element follows a consistent structure:

### Core Fields

```json
{
    "type": "Title",                    // Element type
    "element_id": "unique-id-string",   // Unique identifier
    "text": "Extracted text content",   // Text content
    "metadata": {                       // Metadata object
        // Various metadata fields
    }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Element type (Title, NarrativeText, Table, Image, etc.) |
| `element_id` | `str` | Unique identifier for the element |
| `text` | `str` | The extracted text content of the element |
| `metadata` | `object` | Metadata object containing additional information |

---

## Metadata Fields

Each element includes metadata that provides contextual information. The available metadata fields depend on the document type and processing options.

### Common Metadata Fields

#### page_number
```json
"page_number": 3
```
- **Type**: `int`
- **Description**: Page number where the element appears (1-indexed)
- **Available for**: PDF and multi-page documents

#### coordinates
```json
"coordinates": {
    "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "system": "PixelSpace"
}
```
- **Type**: `object`
- **Description**: Bounding box coordinates of the element on the page
- **Available when**: `coordinates=True` parameter is used
- **Format**: List of coordinate points (typically 4 corners)

#### filename
```json
"filename": "document.pdf"
```
- **Type**: `str`
- **Description**: Name of the source file
- **Available for**: All documents

#### filetype
```json
"filetype": "application/pdf"
```
- **Type**: `str`
- **Description**: MIME type of the source document
- **Examples**: `"application/pdf"`, `"application/vnd.openxmlformats-officedocument.wordprocessingml.document"`

#### languages
```json
"languages": ["eng"]
```
- **Type**: `list[str]`
- **Description**: Languages detected or specified for the document
- **Available for**: Documents processed with OCR or language detection
- **Examples**: `["eng"]`, `["hun"]`, `["eng", "spa"]`

#### page_name
```json
"page_name": "Sheet1"
```
- **Type**: `str`
- **Description**: Name of the sheet or page
- **Available for**: Excel files (sheet names), some other formats

#### category_depth
```json
"category_depth": 1
```
- **Type**: `int`
- **Description**: Depth in the document hierarchy
- **Use**: Indicates nesting level in document structure

### Processing Metadata

#### date_created
```json
"date_created": "2024-01-15T10:30:00"
```
- **Type**: `str` (ISO 8601 timestamp)
- **Description**: Creation date of the document

#### date_modified
```json
"date_modified": "2024-03-20T14:45:00"
```
- **Type**: `str` (ISO 8601 timestamp)
- **Description**: Last modification date of the document

#### date_processed
```json
"date_processed": "2024-10-19T12:00:00"
```
- **Type**: `str` (ISO 8601 timestamp)
- **Description**: Date when the document was processed by Unstructured

#### partitioner_type
```json
"partitioner_type": "hi_res"
```
- **Type**: `str`
- **Description**: Strategy used to partition the document
- **Values**: `"fast"`, `"hi_res"`, `"ocr_only"`, `"auto"`, `"vlm_partition"`

---

## VLM Generated HTML Elements

The **Vision Language Model (VLM)** partitioner generates advanced HTML representations of document elements, providing enhanced structure and relationships.

### VLM-Specific Metadata

#### text_as_html
```json
"text_as_html": "<table class=\"Table\" id=\"abc123\"><tr><th>Header</th></tr>...</table>"
```
- **Type**: `str`
- **Description**: HTML representation of the element
- **Available for**: Tables and structured elements with VLM partitioner
- **Use**: Facilitates reconstruction of document structure with proper HTML markup

#### parent_id
```json
"parent_id": "parent-element-id"
```
- **Type**: `str`
- **Description**: The `element_id` of the parent element
- **Use**: Establishes hierarchical relationships between elements
- **Purpose**: Enables reconstruction of complete document structure

### VLM Table Example

```json
{
    "type": "Table",
    "element_id": "c60aea37616e3db75660918c6d657c38",
    "text": "ITEM | QUANTITY | PRICE | TOTAL\\nOffice Desk | 2 | $249 | $498\\n...",
    "metadata": {
        "category_depth": 1,
        "page_number": 1,
        "parent_id": "8cc3b39afcd948d49d85084eaae80ff8",
        "text_as_html": "<table class=\"Table\" id=\"958308a90ccd4fcb825cb12eed20d103\"><tr><th>ITEM</th><th>QUANTITY</th><th>PRICE</th><th>TOTAL</th></tr><tr><td>Office Desk</td><td>2</td><td>$249</td><td>$498</td></tr></table>",
        "languages": ["eng"],
        "filetype": "application/pdf",
        "partitioner_type": "vlm_partition",
        "filename": "invoice.pdf"
    }
}
```

---

## Data Connector Metadata

When documents are ingested through various data connectors, additional metadata fields may be included:

### Common Connector Fields

- **date_created**: Document creation timestamp
- **date_modified**: Last modification timestamp
- **date_processed**: Processing timestamp
- **permissions_data**: Access permissions information
- **record_locator**: Unique identifier or path within source system
- **url**: Web address where document is accessible
- **version**: Document version information

### Connector-Specific Fields

Different source connectors may add specific metadata fields:

#### Astra DB
- `document_id`: Document identifier in Astra DB

#### Confluence
- `document_id`: Confluence document ID
- `space_id`: Confluence space identifier

#### Dropbox
- `file_id`: Dropbox file identifier
- `protocol`: Access protocol
- `remote_file_path`: Path in Dropbox

#### Google Drive
- `file_id`: Google Drive file identifier

#### Jira
- `id`: Jira issue ID
- `key`: Jira issue key

#### MongoDB
- `collection`: MongoDB collection name
- `database`: MongoDB database name
- `document_id`: Document identifier in MongoDB

#### Notion
- `database_id`: Notion database identifier (for databases)
- `page_id`: Notion page identifier (for pages)

#### Amazon S3
- `protocol`: Access protocol
- `remote_file_path`: S3 object path

#### SharePoint
- `server_relative_path`: SharePoint server-relative path
- `user_pname`: User principal name

---

## Processing Strategies

Unstructured.io offers multiple processing strategies, each optimized for different document types and quality requirements.

### Strategy Types

#### 1. fast
**Optimized for speed with digital-character-based documents**

```python
elements = partition_pdf(
    filename="document.pdf",
    strategy="fast"
)
```

**Characteristics**:
- **Speed**: Very fast
- **Best for**: HTML, DOCX, PPTX, XLSX, TXT with digital text
- **OCR**: Not used
- **Table Detection**: Basic
- **Image Extraction**: Limited
- **Use Case**: Quick text extraction from digital documents

#### 2. hi_res
**High-quality processing for complex documents**

```python
elements = partition_pdf(
    filename="document.pdf",
    strategy="hi_res"
)
```

**Characteristics**:
- **Speed**: Slower but more accurate
- **Best for**: PDFs with embedded images, complex layouts
- **OCR**: Available when needed
- **Table Detection**: Advanced
- **Image Extraction**: Advanced
- **Use Case**: Maximum quality for complex documents, scanned PDFs

#### 3. ocr_only
**Runs OCR using Tesseract**

```python
elements = partition_pdf(
    filename="document.pdf",
    strategy="ocr_only"
)
```

**Characteristics**:
- **Speed**: Moderate
- **Best for**: Scanned documents, images with text
- **OCR**: Always used (Tesseract)
- **Table Detection**: Limited
- **Image Extraction**: Basic
- **Use Case**: Scanned documents requiring text extraction

#### 4. auto
**Automatically determines best strategy per page**

```python
elements = partition_pdf(
    filename="document.pdf",
    strategy="auto"
)
```

**Characteristics**:
- **Speed**: Variable (optimizes per page)
- **Best for**: Mixed documents (digital and scanned pages)
- **OCR**: Used when needed
- **Table Detection**: Yes
- **Image Extraction**: Yes
- **Use Case**: Documents with varying page types, balanced approach

---

## Chunking Strategy

Unstructured provides chunking functions to divide document elements into manageable sections, particularly useful for:
- Fitting content into models with limited context windows
- Creating optimal chunks for RAG applications
- Managing large documents

### Chunking Options

#### basic
Combines elements until reaching a specified character length.

```python
elements = partition_pdf(
    filename="document.pdf",
    chunking_strategy="basic"
)
```

#### by_page
Divides elements based on page boundaries.

```python
elements = partition_pdf(
    filename="document.pdf",
    chunking_strategy="by_page"
)
```

#### by_similarity
Groups elements with similar content.

```python
elements = partition_pdf(
    filename="document.pdf",
    chunking_strategy="by_similarity"
)
```

#### by_title
Segments elements based on title headings.

```python
elements = partition_pdf(
    filename="document.pdf",
    chunking_strategy="by_title"
)
```

### Recovering Original Elements

When chunking is applied, access original elements via `orig_elements`:

```python
for chunk in chunked_elements:
    # Access original elements that formed this chunk
    original_elements = chunk.metadata.orig_elements
    
    # Original elements preserve metadata like page numbers and coordinates
    for orig_elem in original_elements:
        page = orig_elem.metadata.page_number
        coords = orig_elem.metadata.coordinates
```

**Note**: During serialization, the `orig_elements` field is compressed into Base64 gzipped format. Use `elements_from_base64_gzipped_json()` to deserialize.

---

## API Parameters

The Unstructured API provides extensive parameters for customizing document processing.

### Core Parameters

#### filename / file
```python
partition_pdf(filename="document.pdf")
# or
partition_pdf(file=file_object)
```
- **Description**: Path to file or file object to process

#### strategy
```python
partition_pdf(filename="document.pdf", strategy="hi_res")
```
- **Options**: `"fast"`, `"hi_res"`, `"ocr_only"`, `"auto"`
- **Default**: `"hi_res"`
- **Description**: Processing strategy for PDF/image files

#### output_format
```python
partition_pdf(filename="document.pdf", output_format="application/json")
```
- **Options**: `"application/json"`, `"text/csv"`
- **Default**: `"application/json"`
- **Description**: Format of the response

### Content Extraction Parameters

#### coordinates
```python
partition_pdf(filename="document.pdf", coordinates=True)
```
- **Type**: `bool`
- **Default**: `False`
- **Description**: Include coordinate bounding boxes for each element (requires OCR)

#### extract_images_in_pdf
```python
partition_pdf(filename="document.pdf", extract_images_in_pdf=True)
```
- **Type**: `bool`
- **Default**: `False`
- **Description**: Extract images from PDF documents

#### infer_table_structure
```python
partition_pdf(filename="document.pdf", infer_table_structure=True)
```
- **Type**: `bool`
- **Default**: `False`
- **Description**: Infer table structure and generate HTML representation

### Language and OCR Parameters

#### languages
```python
partition_pdf(filename="document.pdf", languages=["eng"])
# or multiple languages
partition_pdf(filename="document.pdf", languages=["eng", "hun"])
```
- **Type**: `list[str]`
- **Description**: Languages present in document for OCR
- **Examples**: `["eng"]` (English), `["hun"]` (Hungarian), `["spa"]` (Spanish)

### Metadata Parameters

#### include_page_breaks
```python
partition_pdf(filename="document.pdf", include_page_breaks=True)
```
- **Type**: `bool`
- **Default**: `False`
- **Description**: Include page break elements in output

#### include_metadata
```python
partition_pdf(filename="document.pdf", include_metadata=True)
```
- **Type**: `bool`
- **Default**: `True`
- **Description**: Include metadata with each element

### Chunking Parameters

#### chunking_strategy
```python
partition_pdf(
    filename="document.pdf",
    chunking_strategy="by_title"
)
```
- **Options**: `"basic"`, `"by_page"`, `"by_similarity"`, `"by_title"`
- **Description**: Strategy for chunking elements after partitioning

### Content Type

#### content_type
```python
partition_pdf(filename="document.pdf", content_type="application/pdf")
```
- **Type**: `str`
- **Description**: Hint about content type for processing

---

## Usage Examples

### Example 1: Basic PDF Processing

```python
from unstructured.partition.pdf import partition_pdf

# Basic partitioning
elements = partition_pdf(filename="document.pdf")

# Iterate through elements
for element in elements:
    print(f"Type: {element.type}")
    print(f"Text: {element.text[:100]}...")
    print(f"Page: {element.metadata.page_number}")
    print()
```

### Example 2: High-Quality Processing with Tables

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="report.pdf",
    strategy="hi_res",
    infer_table_structure=True,
    include_metadata=True
)

# Extract tables
for element in elements:
    if element.type == "Table":
        print(f"Found table on page {element.metadata.page_number}")
        print(f"Text representation:\\n{element.text}")
        
        # Access HTML representation
        if hasattr(element.metadata, 'text_as_html'):
            print(f"HTML:\\n{element.metadata.text_as_html}")
```

### Example 3: Multi-Language OCR

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="multilingual.pdf",
    strategy="hi_res",
    languages=["eng", "hun"],  # English and Hungarian
    coordinates=True
)

for element in elements:
    print(f"{element.type}: {element.text}")
    if hasattr(element.metadata, 'coordinates'):
        print(f"  Coordinates: {element.metadata.coordinates}")
```

### Example 4: Extract Images

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="document.pdf",
    strategy="hi_res",
    extract_images_in_pdf=True
)

# Find image elements
images = [elem for elem in elements if elem.type == "Image"]
print(f"Found {len(images)} images")

for img in images:
    print(f"Image on page {img.metadata.page_number}")
    if img.text:
        print(f"Caption: {img.text}")
```

### Example 5: Chunking for RAG

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="large_document.pdf",
    strategy="hi_res",
    chunking_strategy="by_title"
)

# Elements are now chunked
for chunk in elements:
    print(f"Chunk ID: {chunk.element_id}")
    print(f"Text: {chunk.text[:200]}...")
    
    # Access original elements
    if hasattr(chunk.metadata, 'orig_elements'):
        orig_count = len(chunk.metadata.orig_elements)
        print(f"  Combined from {orig_count} original elements")
```

### Example 6: Word Document Processing

```python
from unstructured.partition.docx import partition_docx

elements = partition_docx(
    filename="document.docx",
    infer_table_structure=True
)

for element in elements:
    print(f"{element.type}: {element.text[:100]}")
```

### Example 7: Excel Spreadsheet Processing

```python
from unstructured.partition.xlsx import partition_xlsx

elements = partition_xlsx(
    filename="spreadsheet.xlsx",
    include_metadata=True
)

for element in elements:
    sheet_name = element.metadata.page_name
    print(f"Sheet '{sheet_name}': {element.text[:100]}")
```

---

## Best Practices

### 1. Choose the Right Strategy

- **Digital documents**: Use `"fast"` for quick processing
- **Scanned PDFs**: Use `"hi_res"` or `"ocr_only"`
- **Mixed documents**: Use `"auto"` for adaptive processing
- **High accuracy needed**: Use `"hi_res"` with table structure inference

### 2. Enable Appropriate Features

```python
elements = partition_pdf(
    filename="document.pdf",
    strategy="hi_res",
    infer_table_structure=True,  # For documents with tables
    coordinates=True,  # If you need position information
    extract_images_in_pdf=True,  # For documents with images
    include_metadata=True  # Always recommended
)
```

### 3. Handle Metadata Safely

```python
# Always check for attribute existence
page_num = getattr(element.metadata, 'page_number', None)
coords = getattr(element.metadata, 'coordinates', None)

# Or use hasattr
if hasattr(element.metadata, 'text_as_html'):
    html = element.metadata.text_as_html
```

### 4. Use Appropriate Languages

```python
# Single language
elements = partition_pdf(filename="doc.pdf", languages=["eng"])

# Multiple languages
elements = partition_pdf(filename="doc.pdf", languages=["eng", "spa"])
```

### 5. Implement Error Handling

```python
try:
    elements = partition_pdf(filename="document.pdf", strategy="hi_res")
    if not elements:
        print("Warning: No elements extracted")
except Exception as e:
    print(f"Processing failed: {e}")
```

### 6. Process Large Documents Efficiently

```python
# Use chunking for large documents
elements = partition_pdf(
    filename="large_doc.pdf",
    strategy="hi_res",
    chunking_strategy="by_page"  # Or "by_title" for semantic chunking
)
```

### 7. Optimize for Your Use Case

**For RAG Applications**:
```python
elements = partition_pdf(
    filename="doc.pdf",
    strategy="hi_res",
    chunking_strategy="by_title",
    infer_table_structure=True
)
```

**For Data Extraction**:
```python
elements = partition_pdf(
    filename="doc.pdf",
    strategy="hi_res",
    infer_table_structure=True,
    coordinates=True
)
```

**For Quick Text Extraction**:
```python
elements = partition_pdf(
    filename="doc.pdf",
    strategy="fast"
)
```

---

## API Services

Unstructured.io offers multiple deployment options:

### Free API Service
- Basic access with 1,000 pages cap
- Good for testing and small projects

### Premium API Services
- **Commercial SaaS API**: Scalable cloud service
- **AWS Marketplace**: Available through AWS
- **Azure Marketplace**: Available through Azure

### Open Source
- Self-hosted deployment
- Full customization capabilities
- No usage limits

---

## Comparison with Other Tools

| Feature | Unstructured.io | MarkItDown | Docling |
|---------|----------------|------------|---------|
| **Multi-format Support** | ✓✓ Extensive | ✓ Good | ✓✓ Extensive |
| **Element Classification** | ✓✓ Detailed | ✓ Basic | ✓✓ Advanced |
| **Table Extraction** | ✓✓ Advanced HTML | ✓ Markdown | ✓✓ Structured |
| **Image Extraction** | ✓✓ Advanced | ✓ References | ✓✓ Advanced |
| **OCR Support** | ✓✓ Built-in | ✗ No | ✓ Built-in |
| **VLM Integration** | ✓✓ Yes | ✓ Optional | ✓✓ Yes |
| **API Service** | ✓✓ Yes | ✗ No | ✗ No |
| **Chunking** | ✓✓ Built-in | ✗ External | ✗ External |
| **Coordinate Tracking** | ✓ Yes | ✗ Limited | ✓ Yes |

### When to Use Unstructured.io

✓ **Best For**:
- RAG applications requiring element-level processing
- Documents with complex table structures
- Multi-format document processing pipelines
- OCR-heavy workloads (scanned documents)
- Production deployments requiring API service
- Projects needing detailed element classification
- Applications requiring coordinate tracking

✗ **Consider Alternatives For**:
- Simple markdown conversion only
- Lightweight dependency requirements
- Offline-only processing
- When SaaS API is not suitable

---

## Version and Compatibility

- **Python**: 3.8 or higher recommended
- **Main Package**: `unstructured`
- **Optional Dependencies**: Various packages for specific file types

### Installation

```bash
# Basic installation
pip install unstructured

# With PDF support
pip install "unstructured[pdf]"

# With all extras
pip install "unstructured[all-docs]"
```

---

## Additional Resources

### Official Documentation
- [Unstructured.io Documentation](https://docs.unstructured.io/)
- [Document Elements Reference](https://docs.unstructured.io/platform/document-elements)
- [API Reference](https://docs.unstructured.io/api-reference/general/summary)
- [Chunking Documentation](https://docs.unstructured.io/open-source/core-functionality/chunking)

### Source Code
- [Unstructured GitHub](https://github.com/Unstructured-IO/unstructured)
- [Documentation GitHub](https://github.com/Unstructured-IO/docs)

### Community
- [Issue Tracker](https://github.com/Unstructured-IO/unstructured/issues)
- [Discussions](https://github.com/Unstructured-IO/unstructured/discussions)

---

## Summary

The Unstructured.io Elements Data Model provides:

1. **Element-Based Architecture**: Documents decomposed into typed, semantic elements
2. **Rich Metadata**: Comprehensive contextual information for each element
3. **Multi-Format Support**: Process 30+ document formats
4. **Advanced Processing**: VLM, OCR, table structure inference
5. **Flexible Strategies**: Choose speed vs. quality tradeoffs
6. **Built-in Chunking**: Optimize for RAG and LLM applications
7. **API and Open Source**: Deploy as SaaS or self-hosted

This makes Unstructured.io ideal for:
- RAG (Retrieval-Augmented Generation) applications
- Document analysis and classification
- Multi-format document processing pipelines
- OCR and scanned document processing
- Enterprise document management
- AI-powered document understanding

---

**Last Updated**: October 2024  
**Based On**: Official Unstructured.io Documentation  
**Version**: Compatible with Unstructured 0.10+
