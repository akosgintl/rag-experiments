# Complete PDF Processing Guide: Text, Table, and Image Extraction

## Overview

This comprehensive guide provides Python code examples for extracting **TEXT**, **TABLES**, and **IMAGES** from PDF documents using three powerful frameworks: **Docling**, **Unstructured.io**, and **MarkItDown**. All examples are optimized for chunking, embedding into vector databases, and creating knowledge graphs.

## Framework Capability Matrix

| Feature | Docling | Unstructured.io | MarkItDown |
|---------|---------|-----------------|------------|
| **TEXT EXTRACTION** |
| Quality | ★★★★★ | ★★★★☆ | ★★★★☆ |
| Layout understanding | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Element classification | ★★★★☆ | ★★★★★ | ★★☆☆☆ |
| **TABLE EXTRACTION** |
| Detection | ★★★★★ | ★★★★★ | ★★☆☆☆ |
| Structure preservation | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Complex tables | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| **IMAGE EXTRACTION** |
| Detection | ★★★★★ | ★★★★★ | ★★☆☆☆ |
| File saving | ★★★★★ | ★★★★★ | ★☆☆☆☆ |
| Descriptions | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ |

## Installation Requirements

### Docling - Full Installation
```bash
pip install docling[vlm] docling-core
pip install sentence-transformers chromadb neo4j
pip install pillow pandas openpyxl
```

### Unstructured.io - Full Installation
```bash
pip install unstructured[pdf,paddleocr]
pip install sentence-transformers chromadb neo4j
pip install pillow pandas openpyxl
```

### MarkItDown - Full Installation
```bash
pip install markitdown
pip install sentence-transformers chromadb neo4j openai
pip install pandas beautifulsoup4 lxml
```

## 1. DOCLING - Advanced PDF Processing

**Best for:** Complex PDFs, scientific papers, financial reports with sophisticated layouts

### Key Features:
- ✅ **Superior table extraction** with DataFrame export
- ✅ **Advanced layout understanding** with spatial analysis
- ✅ **Complete image extraction** (pages, figures, table images)
- ✅ **Multiple output formats** (Markdown, JSON, HTML, DataFrame)

```bash
result = process_pdf_with_docling_advanced("complex_document.pdf")
```

## 2. UNSTRUCTURED.IO - Production-Ready Processing

**Best for:** Production RAG systems, enterprise document processing, mixed content types

### Key Features:
- ✅ **Element classification** with automatic categorization
- ✅ **Built-in semantic chunking** strategies
- ✅ **Base64 image extraction** with metadata
- ✅ **Multiple processing strategies** (fast/auto/hi_res)

```bash
# High-quality extraction:
result = process_pdf_with_unstructured_advanced("document.pdf", strategy="hi_res")

# Balanced approach: 
result = process_pdf_with_unstructured_advanced("document.pdf", strategy="auto")  

# Fast processing:
result = process_pdf_with_unstructured_advanced("document.pdf", strategy="fast")
```

## 3. MARKITDOWN - Lightweight Processing

**Best for:** Quick prototyping, simple PDFs, Markdown workflows, LLM-enhanced descriptions

### Key Features:
- ✅ **Simple and fast** processing with minimal setup
- ✅ **Markdown table extraction** with CSV conversion
- ✅ **LLM-enhanced image descriptions** (with OpenAI)
- ✅ **Header-based semantic chunking**

```bash
# Basic:
result = process_pdf_with_markitdown_advanced("document.pdf")

# With OpenAI:

result = process_pdf_with_markitdown_advanced("document.pdf", openai_api_key="your-key")
```

## Framework Selection Guide

### When to Use Each Framework:

**🔬 Use DOCLING when:**
- Complex PDFs with sophisticated layouts
- Scientific papers, financial reports
- Table structure preservation is critical
- Highest quality extraction needed
- You can handle higher resource requirements

**⚙️ Use UNSTRUCTURED.IO when:**
- Building production RAG systems
- Need element classification and rich metadata
- Want built-in chunking strategies
- Processing mixed document types
- Require enterprise-grade reliability

**⚡ Use MARKITDOWN when:**
- Quick prototyping and simple conversion
- Working with straightforward PDFs
- Want Markdown output specifically
- Have limited computational resources
- Need LLM-enhanced image descriptions

## Key Differences Summary:

| Content Type | Docling | Unstructured.io | MarkItDown |
|-------------|---------|-----------------|-------------|
| **TEXT** | Excellent quality with layout understanding | Good with element classification | Fast with clean Markdown output |
| **TABLES** | Superior - preserves complex structures, exports to DataFrame | Good - HTML structure with CSV conversion | Basic - simple Markdown tables only |
| **IMAGES** | Excellent - extracts all types with metadata | Good - Base64 extraction with coordinates | Limited - references only, but great LLM descriptions |

## Usage Recommendations:

1. **Start with complexity analysis** - Use MarkItDown for quick assessment
2. **Choose based on content type importance** - Docling for tables, Unstructured for production
3. **Consider processing speed vs quality trade-offs**
4. **Implement fallback strategies** for robust processing
5. **Use appropriate chunking strategies** based on document structure

All frameworks support:
- ✅ Vector database integration (ChromaDB shown)
- ✅ Knowledge graph preparation
- ✅ Comprehensive metadata preservation
- ✅ Multiple output formats
- ✅ Semantic chunking strategies
- ✅ Production-ready error handling

Choose the framework that best matches your specific requirements for text quality, table complexity, and image processing needs.