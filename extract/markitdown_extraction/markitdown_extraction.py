# ============================================================
# MARKITDOWN - COMPREHENSIVE TEXT, TABLE, AND IMAGE EXTRACTION
# ============================================================

# Installation with full capabilities
# pip install markitdown sentence-transformers chromadb openai
# pip install pandas beautifulsoup4 lxml

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from bs4 import BeautifulSoup

from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

class MarkItDownAdvancedProcessor:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", openai_client: Optional[OpenAI] = None, 
                 output_dir="./extracted_content"):
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
        
        # Text splitters for different approaches
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\\n\\n", "\\n", " ", ""]
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
    
    def extract_comprehensive_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract TEXT, TABLES, and IMAGES from PDF using MarkItDown
        Note: MarkItDown has limitations for images and tables compared to other frameworks
        """
        try:
            print(f"Processing {pdf_path} with MarkItDown...")
            
            # Convert PDF to Markdown
            result = self.md.convert(pdf_path)
            
            if not result or not result.text_content:
                print("Conversion failed or returned empty content")
                return None
            
            markdown_content = result.text_content
            
            extracted_content = {
                "text_content": {"full_markdown": markdown_content},
                "tables": [],
                "images": [],
                "metadata": {
                    "source": pdf_path,
                    "title": result.title or Path(pdf_path).stem,
                    "conversion_successful": True,
                    "has_llm_support": self.has_llm
                }
            }
            
            # Analyze and extract different content types
            self._extract_text_content(markdown_content, extracted_content)
            self._extract_tables(markdown_content, extracted_content, pdf_path)
            self._extract_image_references(markdown_content, extracted_content, pdf_path)
            
            return extracted_content
            
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
    
    def _extract_text_content(self, markdown_content: str, extracted_content: Dict):
        """Extract and analyze text content structure"""
        
        # Basic text analysis
        analysis = {
            "character_count": len(markdown_content),
            "word_count": len(markdown_content.split()),
            "line_count": len(markdown_content.split('\\n')),
            "paragraph_count": len([p for p in markdown_content.split('\\n\\n') if p.strip()]),
            "headers": {"h1": 0, "h2": 0, "h3": 0, "h4": 0, "h5": 0, "h6": 0}
        }
        
        # Count headers
        for level in range(1, 7):
            pattern = f"^{'#' * level} "
            analysis["headers"][f"h{level}"] = len(re.findall(pattern, markdown_content, re.MULTILINE))
        
        # Extract structured elements
        elements = []
        lines = markdown_content.split('\\n')
        
        current_section = {"level": 0, "title": ""}
        element_counter = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            element_counter += 1
            
            # Detect headers
            header_match = re.match(r'^(#{1,6})\\s+(.+)', line)
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
            
            # Detect other content
            elif line.startswith('- ') or line.startswith('* ') or re.match(r'^\\d+\\.\\s+', line):
                elements.append({
                    "id": f"list_item_{element_counter}",
                    "type": "ListItem",
                    "text": re.sub(r'^[-*]\\s+|^\\d+\\.\\s+', '', line),
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
        
        extracted_content["text_content"].update({
            "analysis": analysis,
            "elements": elements
        })
        
        print(f"✓ Analyzed text: {analysis['word_count']} words, {len(elements)} elements")
    
    def _extract_tables(self, markdown_content: str, extracted_content: Dict, pdf_path: str):
        """Extract tables from Markdown content"""
        tables = []
        table_counter = 0
        
        # Find markdown tables using regex
        table_pattern = r'(\\|[^\\n]+\\|[\\n\\r]+\\|[-\\s\\|:]+\\|[\\n\\r]+(\\|[^\\n]*\\|[\\n\\r]*)*)'
        table_matches = re.findall(table_pattern, markdown_content, re.MULTILINE)
        
        for match in table_matches:
            table_counter += 1
            table_markdown = match[0] if isinstance(match, tuple) else match
            
            try:
                # Parse markdown table
                lines = [line.strip() for line in table_markdown.strip().split('\\n') if line.strip()]
                
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
                    
                    table_info = {
                        "id": f"table_{table_counter}",
                        "markdown": table_markdown,
                        "headers": headers,
                        "data": rows,
                        "row_count": len(rows),
                        "col_count": len(headers),
                        "csv_file": str(csv_path),
                        "markdown_file": str(md_path)
                    }
                    
                    tables.append(table_info)
                    
            except Exception as e:
                print(f"Error processing table {table_counter}: {e}")
                tables.append({
                    "id": f"table_{table_counter}",
                    "markdown": table_markdown,
                    "error": str(e)
                })
        
        extracted_content["tables"] = tables
        print(f"✓ Extracted {len(tables)} tables")
    
    def _extract_image_references(self, markdown_content: str, extracted_content: Dict, pdf_path: str):
        """Extract image references and descriptions from Markdown"""
        images = []
        
        # Find markdown image syntax: ![alt text](image_path)
        image_pattern = r'!\\[([^\\]]*)\\]\\(([^)]+)\\)'
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
            r'\\[Image:\\s*([^\\]]+)\\]',
            r'\\[Figure\\s*\\d*:?\\s*([^\\]]+)\\]',
            r'\\[Chart:\\s*([^\\]]+)\\]',
            r'\\[Diagram:\\s*([^\\]]+)\\]'
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
        
        extracted_content["images"] = images
        print(f"✓ Found {len(images)} image references/descriptions")
    
    def create_comprehensive_chunks(self, extracted_content: Dict, pdf_path: str) -> List[Dict[str, Any]]:
        """Create chunks using header-based semantic chunking when possible"""
        chunks = []
        
        markdown_content = extracted_content.get("text_content", {}).get("full_markdown", "")
        if not markdown_content:
            return chunks
        
        # Determine chunking strategy based on structure
        analysis = extracted_content.get("text_content", {}).get("analysis", {})
        has_headers = sum(analysis.get("headers", {}).values()) > 0
        
        if has_headers:
            print("Using header-based semantic chunking...")
            chunks = self._chunk_by_headers(markdown_content, pdf_path)
        else:
            print("Using character-based chunking...")
            chunks = self._chunk_by_characters(markdown_content, pdf_path)
        
        # Add table chunks
        table_chunks = self._create_table_chunks(extracted_content, pdf_path)
        chunks.extend(table_chunks)
        
        # Add image description chunks
        image_chunks = self._create_image_chunks(extracted_content, pdf_path)
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
    
    def _create_table_chunks(self, extracted_content: Dict, pdf_path: str) -> List[Dict[str, Any]]:
        """Create chunks for tables"""
        chunks = []
        
        for table in extracted_content.get("tables", []):
            if table.get("markdown"):
                table_text = f"Table {table['id']}\\n{table['markdown']}"
                
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
    
    def _create_image_chunks(self, extracted_content: Dict, pdf_path: str) -> List[Dict[str, Any]]:
        """Create chunks for image descriptions"""
        chunks = []
        
        for image in extracted_content.get("images", []):
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
    
    def save_extraction_summary(self, extracted_content: Dict, pdf_path: str):
        """Save extraction summary"""
        analysis = extracted_content.get("text_content", {}).get("analysis", {})
        
        summary = {
            "document": Path(pdf_path).name,
            "extraction_summary": {
                "text": {
                    "word_count": analysis.get("word_count", 0),
                    "character_count": analysis.get("character_count", 0),
                    "elements_count": len(extracted_content.get("text_content", {}).get("elements", [])),
                    "headers": analysis.get("headers", {}),
                    "structure_detected": sum(analysis.get("headers", {}).values()) > 0
                },
                "tables": {
                    "count": len(extracted_content.get("tables", [])),
                    "with_data": len([t for t in extracted_content.get("tables", []) if t.get("data")]),
                    "files": [t.get("csv_file") for t in extracted_content.get("tables", []) if t.get("csv_file")]
                },
                "images": {
                    "count": len(extracted_content.get("images", [])),
                    "with_descriptions": len([img for img in extracted_content.get("images", []) if img.get("description") or img.get("llm_description")]),
                    "enhanced_with_llm": len([img for img in extracted_content.get("images", []) if img.get("enhanced")])
                }
            },
            "metadata": extracted_content.get("metadata", {})
        }
        
        summary_path = self.output_dir / f"{Path(pdf_path).stem}_markitdown_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved extraction summary to {summary_path}")
        return summary

# Usage function
def process_pdf_with_markitdown_advanced(pdf_path: str, openai_api_key: Optional[str] = None, 
                                        output_dir: str = "./markitdown_output"):
    """
    Complete workflow for comprehensive PDF processing with MarkItDown
    """
    # Initialize with or without OpenAI
    openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    processor = MarkItDownAdvancedProcessor(openai_client=openai_client, output_dir=output_dir)
    
    # 1. Extract all content types
    extracted_content = processor.extract_comprehensive_content(pdf_path)
    if not extracted_content:
        return None
    
    # 2. Create comprehensive chunks
    chunks = processor.create_comprehensive_chunks(extracted_content, pdf_path)
    
    # 3. Save summary
    summary = processor.save_extraction_summary(extracted_content, pdf_path)
    
    # 4. Store in vector database
#    client = chromadb.PersistentClient(path="./chroma_db")
#    collection_name = f"markitdown_comprehensive_{Path(pdf_path).stem}"
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
        "extracted_content": extracted_content,
        "chunks": chunks,
        "summary": summary,
        "output_directory": output_dir
    }

# Usage examples:
# Basic usage (no LLM for images)
# result = process_pdf_with_markitdown_advanced("document.pdf")

# With OpenAI for enhanced image processing
result = process_pdf_with_markitdown_advanced("document.pdf", openai_api_key="your-api-key")

# Custom output directory
# result = process_pdf_with_markitdown_advanced("document.pdf", output_dir="./my_extraction")


print("\nMARKITDOWN Enhanced Example Created")
print("=" * 40)
print("Features:")
print("✓ Comprehensive TEXT extraction with structure analysis")
print("✓ TABLE extraction from Markdown tables with CSV conversion")
print("✓ IMAGE reference extraction and optional LLM-enhanced descriptions")
print("✓ Header-based semantic chunking with fallback")
print("✓ Markdown structure preservation")
print("✓ Optional OpenAI integration for better image understanding")
print("✓ Lightweight processing suitable for simpler documents")
print("\nNote: MarkItDown has more limited table/image capabilities")
print("compared to Docling and Unstructured.io, but excels at clean")
print("text extraction and Markdown conversion.")
