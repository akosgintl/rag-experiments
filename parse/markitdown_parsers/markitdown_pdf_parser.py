# ============================================================
# MARKITDOWN - COMPREHENSIVE PDF TEXT, TABLE, AND IMAGE PARSING
# ============================================================

# Installation with full capabilities
# pip install markitdown sentence-transformers chromadb openai python-dotenv
# pip install pandas beautifulsoup4 lxml
#
# Setup:
# 1. Copy env.sample to .env
# 2. Add your OpenAI API key to the .env file (optional, for enhanced image processing)

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

class MarkItDownPDFProcessor:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", openai_client: Optional[OpenAI] = None, 
                 output_dir="./markitdown_pdf_output"):
        # Initialize with or without OpenAI for enhanced image processing
        if openai_client:
            self.md = MarkItDown(llm_client=openai_client, llm_model="gpt-4o")
            self.has_llm = True
            print("✓ Initialized with OpenAI LLM support for image descriptions")
        else:
            self.md = MarkItDown()
            self.has_llm = False
            print("✓ Initialized without LLM support (basic mode)")
        
        self.embedder = SentenceTransformer(embedding_model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
        # Text splitters for different approaches
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3"),
                ("####", "Header 4"),
            ],
            strip_headers=False
        )
    
    def parse_comprehensive_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse TEXT, TABLES, and IMAGES from PDF using MarkItDown
        Note: MarkItDown provides good text extraction and markdown conversion
        """
        try:
            print(f"Processing {pdf_path} with MarkItDown...")
            
            # Convert PDF to Markdown
            result = self.md.convert(pdf_path)
            
            parsed_content = {
                "text_content": {"full_markdown": result.markdown},
                "tables": [],
                "images": [],
                "metadata": {
                    "source": pdf_path,
                    "title": result.title or Path(pdf_path).stem,
                    "conversion_successful": True,
                    "has_llm_support": self.has_llm
                }
            }
            
            # Save original markdown
            md_path = self.output_dir / "text" / f"{Path(pdf_path).stem}_markdown.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(result.markdown)
            
            # Save plain text version
            txt_path = self.output_dir / "text" / f"{Path(pdf_path).stem}_plain_text.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result.text_content)

            # Analyze and parse different content types
            self._parse_text_content(result.markdown, parsed_content, pdf_path)
            self._parse_tables(result.markdown, parsed_content, pdf_path)
            self._parse_image_references(result.markdown, parsed_content, pdf_path)
            
            return parsed_content
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return {
                "text_content": {"full_markdown": ""},
                "tables": [],
                "images": [],
                "metadata": {
                    "source": pdf_path,
                    "conversion_successful": False,
                    "error": str(e)
                }
            }
    
    def _parse_text_content(self, markdown_content: str, parsed_content: Dict, pdf_path: str):
        """Parse and analyze text content structure"""
        
        # Basic text analysis
        analysis = {
            "character_count": len(markdown_content),
            "word_count": len(markdown_content.split()),
            "line_count": len(markdown_content.split('\n')),
            "paragraph_count": len([p for p in markdown_content.split('\n\n') if p.strip()]),
            "headers": {"h1": 0, "h2": 0, "h3": 0, "h4": 0, "h5": 0, "h6": 0}
        }
        
        # Count headers
        for level in range(1, 7):
            pattern = f"^{'#' * level} "
            analysis["headers"][f"h{level}"] = len(re.findall(pattern, markdown_content, re.MULTILINE))
        
        # Extract structured elements
        elements = []
        lines = markdown_content.split('\n')
        
        current_section = {"level": 0, "title": ""}
        element_counter = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            element_counter += 1
            
            # Detect headers
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = {"level": level, "title": title}
                
                elements.append({
                    "id": f"header_{element_counter}",
                    "type": f"Header{level}",
                    "text": title,
                    "line_number": i + 1,
                    "section": current_section.copy()
                })
            
            # Detect list items
            elif line.startswith('- ') or line.startswith('* ') or re.match(r'^\d+\.\s+', line):
                elements.append({
                    "id": f"list_item_{element_counter}",
                    "type": "ListItem",
                    "text": re.sub(r'^[-*]\s+|^\d+\.\s+', '', line),
                    "line_number": i + 1,
                    "section": current_section.copy()
                })
            
            # Detect page markers (if present)
            elif re.match(r'^\[Page\s+\d+\]', line) or re.match(r'^Page\s+\d+', line):
                elements.append({
                    "id": f"page_marker_{element_counter}",
                    "type": "PageMarker",
                    "text": line,
                    "line_number": i + 1,
                    "section": current_section.copy()
                })
            
            elif line and not line.startswith('|') and not line.startswith('!['):
                # Regular paragraph text
                elements.append({
                    "id": f"text_{element_counter}",
                    "type": "Text",
                    "text": line,
                    "line_number": i + 1,
                    "section": current_section.copy()
                })
        
        parsed_content["text_content"].update({
            "analysis": analysis,
            "elements": elements
        })
        
        print(f"✓ Parsed text: {analysis['word_count']} words, {len(elements)} elements")
    
    def _parse_tables(self, markdown_content: str, parsed_content: Dict, pdf_path: str):
        """Parse tables from Markdown content"""
        tables = []
        table_counter = 0
        
        # Find markdown tables using regex
        table_pattern = r'(\|[^\n]+\|[\n\r]+\|[-\s\|:]+\|[\n\r]+(\|[^\n]*\|[\n\r]*)*)'
        table_matches = re.findall(table_pattern, markdown_content, re.MULTILINE)
        
        for match in table_matches:
            table_counter += 1
            table_markdown = match[0] if isinstance(match, tuple) else match
            
            try:
                # Parse markdown table
                lines = [line.strip() for line in table_markdown.strip().split('\n') if line.strip()]
                
                if len(lines) < 2:
                    continue
                
                # Extract headers and data
                header_line = lines[0]
                separator_line = lines[1] if len(lines) > 1 else ""
                data_lines = lines[2:] if len(lines) > 2 else []
                
                # Parse headers
                headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
                
                # Parse data rows
                rows = []
                for line in data_lines:
                    if '|' in line:
                        row = [cell.strip() for cell in line.split('|')[1:-1]]
                        if len(row) == len(headers):
                            rows.append(row)
                
                if headers and rows:
                    # Create DataFrame
                    df = pd.DataFrame(rows, columns=headers)
                    
                    # Save as CSV
                    csv_path = self.output_dir / "tables" / f"{Path(pdf_path).stem}_table_{table_counter}.csv"
                    df.to_csv(csv_path, index=False)
                    
                    # Save original markdown table
                    md_path = self.output_dir / "tables" / f"{Path(pdf_path).stem}_table_{table_counter}.md"
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(table_markdown)
                    
                    # Calculate basic statistics
                    stats = self._calculate_table_statistics(df)
                    
                    table_info = {
                        "id": f"table_{table_counter}",
                        "markdown": table_markdown,
                        "headers": headers,
                        "data": rows,
                        "row_count": len(rows),
                        "col_count": len(headers),
                        "csv_file": str(csv_path),
                        "markdown_file": str(md_path),
                        "statistics": stats
                    }
                    
                    tables.append(table_info)
                    
            except Exception as e:
                print(f"Error processing table {table_counter}: {e}")
                tables.append({
                    "id": f"table_{table_counter}",
                    "markdown": table_markdown,
                    "error": str(e)
                })
        
        parsed_content["tables"] = tables
        print(f"✓ Parsed {len(tables)} tables")
    
    def _calculate_table_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for a table"""
        stats = {
            "total_cells": df.size,
            "empty_cells": df.isna().sum().sum(),
            "numeric_columns": [],
            "text_columns": []
        }
        
        for col in df.columns:
            try:
                # Try to convert to numeric
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if numeric_col.notna().sum() > len(df) * 0.5:  # More than 50% are numbers
                    stats["numeric_columns"].append({
                        "name": col,
                        "type": "numeric",
                        "non_null_count": numeric_col.notna().sum()
                    })
                else:
                    stats["text_columns"].append({
                        "name": col,
                        "type": "text",
                        "non_null_count": df[col].notna().sum()
                    })
            except:
                stats["text_columns"].append({
                    "name": col,
                    "type": "text",
                    "non_null_count": df[col].notna().sum()
                })
        
        return stats
    
    def _parse_image_references(self, markdown_content: str, parsed_content: Dict, pdf_path: str):
        """Parse image references and descriptions from Markdown"""
        images = []
        
        # Find markdown image syntax: ![alt text](image_path)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        image_matches = re.findall(image_pattern, markdown_content)
        
        for i, (alt_text, image_path) in enumerate(image_matches):
            image_info = {
                "id": f"image_{i+1}",
                "alt_text": alt_text,
                "path_reference": image_path,
                "type": "markdown_reference"
            }
            
            # Enhanced description with LLM if available
            if self.has_llm and alt_text:
                try:
                    # The LLM integration in MarkItDown should have already processed images
                    # We can use the alt_text which might contain LLM-generated descriptions
                    image_info["llm_description"] = alt_text
                    image_info["enhanced"] = True
                except Exception as e:
                    print(f"Could not get enhanced description for image {i+1}: {e}")
                    image_info["enhanced"] = False
            
            images.append(image_info)
        
        # Also look for any image-related text that might be descriptions
        # This is useful when images are converted to text descriptions by MarkItDown+LLM
        image_description_patterns = [
            r'\[Image:\s*([^\]]+)\]',
            r'\[Figure\s*\d*:?\s*([^\]]+)\]',
            r'\[Chart:\s*([^\]]+)\]',
            r'\[Diagram:\s*([^\]]+)\]',
            r'\[Graph:\s*([^\]]+)\]',
            r'\[Photo:\s*([^\]]+)\]'
        ]
        
        for pattern in image_description_patterns:
            matches = re.findall(pattern, markdown_content, re.IGNORECASE)
            for j, description in enumerate(matches):
                image_info = {
                    "id": f"description_{len(images) + j + 1}",
                    "description": description.strip(),
                    "type": "text_description",
                    "enhanced": self.has_llm
                }
                images.append(image_info)
        
        parsed_content["images"] = images
        print(f"✓ Parsed {len(images)} image references/descriptions")
    
    def create_comprehensive_chunks(self, parsed_content: Dict, pdf_path: str) -> List[Dict[str, Any]]:
        """Create chunks using header-based semantic chunking when possible"""
        chunks = []
        
        markdown_content = parsed_content.get("text_content", {}).get("full_markdown", "")
        if not markdown_content:
            return chunks
        
        # Determine chunking strategy based on structure
        analysis = parsed_content.get("text_content", {}).get("analysis", {})
        has_headers = sum(analysis.get("headers", {}).values()) > 0
        
        if has_headers:
            print("Using header-based semantic chunking...")
            chunks = self._chunk_by_headers(markdown_content, pdf_path)
        else:
            print("Using character-based chunking...")
            chunks = self._chunk_by_characters(markdown_content, pdf_path)
        
        # Add table chunks
        table_chunks = self._create_table_chunks(parsed_content, pdf_path)
        chunks.extend(table_chunks)
        
        # Add image description chunks
        image_chunks = self._create_image_chunks(parsed_content, pdf_path)
        chunks.extend(image_chunks)
        
        print(f"✓ Created {len(chunks)} comprehensive chunks")
        return chunks
    
    def _chunk_by_headers(self, markdown_content: str, pdf_path: str) -> List[Dict[str, Any]]:
        """Chunk content by Markdown headers"""
        try:
            header_chunks = self.header_splitter.split_text(markdown_content)
            chunks = []
            
            for i, chunk in enumerate(header_chunks):
                if len(chunk.page_content) > 1500:
                    # Split large chunks further
                    sub_chunks = self.char_splitter.split_text(chunk.page_content)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_obj = {
                            "id": f"{Path(pdf_path).stem}_header_chunk_{i}_{j}",
                            "type": "text",
                            "content": sub_chunk,
                            "embedding": self.embedder.encode(sub_chunk).tolist(),
                            "metadata": {
                                **chunk.metadata,
                                "chunk_type": "header_subchunk",
                                "parent_chunk": i,
                                "sub_chunk": j,
                                "chunk_size": len(sub_chunk),
                                "source": pdf_path
                            }
                        }
                        chunks.append(chunk_obj)
                else:
                    chunk_obj = {
                        "id": f"{Path(pdf_path).stem}_header_chunk_{i}",
                        "type": "text", 
                        "content": chunk.page_content,
                        "embedding": self.embedder.encode(chunk.page_content).tolist(),
                        "metadata": {
                            **chunk.metadata,
                            "chunk_type": "header_chunk",
                            "chunk_index": i,
                            "chunk_size": len(chunk.page_content),
                            "source": pdf_path
                        }
                    }
                    chunks.append(chunk_obj)
            
            return chunks
            
        except Exception as e:
            print(f"Header chunking failed: {e}")
            return self._chunk_by_characters(markdown_content, pdf_path)
    
    def _chunk_by_characters(self, markdown_content: str, pdf_path: str) -> List[Dict[str, Any]]:
        """Fallback character-based chunking"""
        text_chunks = self.char_splitter.split_text(markdown_content)
        chunks = []
        
        for i, chunk in enumerate(text_chunks):
            chunk_obj = {
                "id": f"{Path(pdf_path).stem}_char_chunk_{i}",
                "type": "text",
                "content": chunk,
                "embedding": self.embedder.encode(chunk).tolist(),
                "metadata": {
                    "chunk_type": "character_chunk",
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "source": pdf_path
                }
            }
            chunks.append(chunk_obj)
        
        return chunks
    
    def _create_table_chunks(self, parsed_content: Dict, pdf_path: str) -> List[Dict[str, Any]]:
        """Create chunks for tables"""
        chunks = []
        
        for table in parsed_content.get("tables", []):
            if table.get("markdown"):
                table_text = f"Table {table['id']}\n{table['markdown']}"
                
                chunk_obj = {
                    "id": f"{Path(pdf_path).stem}_table_chunk_{table['id']}",
                    "type": "table",
                    "content": table_text,
                    "embedding": self.embedder.encode(table_text).tolist(),
                    "metadata": {
                        "content_type": "table",
                        "table_id": table["id"],
                        "row_count": table.get("row_count", 0),
                        "col_count": table.get("col_count", 0),
                        "csv_file": table.get("csv_file"),
                        "markdown_file": table.get("markdown_file"),
                        "source": pdf_path
                    }
                }
                chunks.append(chunk_obj)
        
        return chunks
    
    def _create_image_chunks(self, parsed_content: Dict, pdf_path: str) -> List[Dict[str, Any]]:
        """Create chunks for image descriptions"""
        chunks = []
        
        for image in parsed_content.get("images", []):
            description = image.get("llm_description") or image.get("description") or image.get("alt_text")
            if description and description.strip():
                image_text = f"Image {image['id']}: {description}"
                
                chunk_obj = {
                    "id": f"{Path(pdf_path).stem}_image_chunk_{image['id']}",
                    "type": "image",
                    "content": image_text,
                    "embedding": self.embedder.encode(image_text).tolist(),
                    "metadata": {
                        "content_type": "image", 
                        "image_id": image["id"],
                        "image_type": image.get("type", "unknown"),
                        "enhanced": image.get("enhanced", False),
                        "source": pdf_path
                    }
                }
                chunks.append(chunk_obj)
        
        return chunks
    
    def save_parsing_summary(self, parsed_content: Dict, pdf_path: str):
        """Save parsing summary"""
        analysis = parsed_content.get("text_content", {}).get("analysis", {})
        
        summary = {
            "document": Path(pdf_path).name,
            "parsing_summary": {
                "text": {
                    "word_count": analysis.get("word_count", 0),
                    "character_count": analysis.get("character_count", 0),
                    "elements_count": len(parsed_content.get("text_content", {}).get("elements", [])),
                    "headers": analysis.get("headers", {}),
                    "structure_detected": sum(analysis.get("headers", {}).values()) > 0
                },
                "tables": {
                    "count": len(parsed_content.get("tables", [])),
                    "with_data": len([t for t in parsed_content.get("tables", []) if t.get("data")]),
                    "total_rows": sum(t.get("row_count", 0) for t in parsed_content.get("tables", [])),
                    "total_columns": sum(t.get("col_count", 0) for t in parsed_content.get("tables", [])),
                    "files": [t.get("csv_file") for t in parsed_content.get("tables", []) if t.get("csv_file")]
                },
                "images": {
                    "count": len(parsed_content.get("images", [])),
                    "with_descriptions": len([img for img in parsed_content.get("images", []) if img.get("description") or img.get("llm_description")]),
                    "enhanced_with_llm": len([img for img in parsed_content.get("images", []) if img.get("enhanced")])
                }
            },
            "metadata": parsed_content.get("metadata", {})
        }
        
        summary_path = self.output_dir / f"{Path(pdf_path).stem}_markitdown_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved parsing summary to {summary_path}")
        return summary

# Usage function
def process_pdf_with_markitdown(pdf_path: str, openai_api_key: Optional[str] = None, 
                                output_dir: str = "./markitdown_pdf_output"):
    """
    Complete workflow for comprehensive PDF processing with MarkItDown
    
    Args:
        pdf_path: Path to the PDF file to process
        openai_api_key: Optional OpenAI API key. If not provided, will try to load from OPENAI_API_KEY environment variable
        output_dir: Directory to save the output files
    """
    # Get API key from parameter or environment variable
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    
    # Initialize with or without OpenAI
    openai_client = OpenAI(api_key=api_key) if api_key else None
    processor = MarkItDownPDFProcessor(openai_client=openai_client, output_dir=output_dir)
    
    # 1. Parse all content types
    parsed_content = processor.parse_comprehensive_content(pdf_path)
    if not parsed_content:
        return None
    
    # 2. Create comprehensive chunks
    # chunks = processor.create_comprehensive_chunks(parsed_content, pdf_path)
    
    # 3. Save summary
    summary = processor.save_parsing_summary(parsed_content, pdf_path)
    
    # 4. Store in vector database (optional)
#    client = chromadb.PersistentClient(path="./chroma_db")
#    collection_name = f"markitdown_pdf_{Path(pdf_path).stem}"
#    collection = client.get_or_create_collection(name=collection_name)
    
#    valid_chunks = [c for c in chunks if c.get("embedding")]
#    if valid_chunks:
#        collection.upsert(
#            documents=[c["content"] for c in valid_chunks],
#            embeddings=[c["embedding"] for c in valid_chunks],
#            ids=[c["id"] for c in valid_chunks],
#            metadatas=[c["metadata"] for c in valid_chunks]
#        )
#        print(f"✓ Stored {len(valid_chunks)} chunks in ChromaDB")
    
    return {
        "parsed_content": parsed_content,
        # "chunks": chunks,
        "summary": summary,
        "output_directory": output_dir
    }

# Usage examples:
# Basic usage (no LLM for images)
result = process_pdf_with_markitdown("docs\\DO_NOT_KovSpec.pdf", openai_api_key=None, output_dir="./markitdown_pdf_output")

# With OpenAI for enhanced image processing (API key from .env file)
# result = process_pdf_with_markitdown("docs\\DO_NOT_KovSpec.pdf")

# With OpenAI for enhanced image processing (API key passed directly)
# result = process_pdf_with_markitdown("docs\\DO_NOT_KovSpec.pdf", openai_api_key="sk-proj-1234567890")


print("\nMARKITDOWN PDF Parser Created")
print("=" * 70)
print("Features:")
print("✓ Comprehensive PDF TEXT parsing with structure analysis")
print("✓ TABLE parsing from Markdown tables with CSV conversion")
print("✓ IMAGE reference parsing and optional LLM-enhanced descriptions")
print("✓ Header-based semantic chunking with fallback")
print("✓ Markdown structure preservation")
print("✓ Optional OpenAI integration for better image understanding")
print("✓ Efficient processing of PDF documents")
print("✓ Plain text and Markdown output for flexibility")
print("\nNote: MarkItDown provides clean text extraction from PDFs")
print("and is excellent for converting PDF content to Markdown format.")
print("For complex PDF features (forms, annotations), consider specialized")
print("libraries like PyMuPDF or pdfplumber.")

