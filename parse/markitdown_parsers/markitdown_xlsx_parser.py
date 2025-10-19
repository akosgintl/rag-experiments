# ============================================================
# MARKITDOWN - COMPREHENSIVE XLSX/EXCEL PARSING
# ============================================================

# Installation with full capabilities
# pip install markitdown sentence-transformers chromadb openai python-dotenv
# pip install pandas beautifulsoup4 lxml openpyxl
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

class MarkItDownXLSXProcessor:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", openai_client: Optional[OpenAI] = None, 
                 output_dir="./markitdown_xlsx_output"):
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
        (self.output_dir / "sheets").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        
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
    
    def parse_comprehensive_content(self, xlsx_path: str) -> Dict[str, Any]:
        """
        Parse SHEETS, TABLES, and IMAGES from XLSX using MarkItDown
        Note: MarkItDown converts Excel sheets to Markdown format
        """
        try:
            print(f"Processing {xlsx_path} with MarkItDown...")
            
            # Convert XLSX to Markdown
            result = self.md.convert(xlsx_path)
            
            parsed_content = {
                "text_content": {"full_markdown": result.markdown},
                "sheets": [],
                "tables": [],
                "images": [],
                "metadata": {
                    "source": xlsx_path,
                    "title": result.title or Path(xlsx_path).stem,
                    "conversion_successful": True,
                    "has_llm_support": self.has_llm
                }
            }
            
            # Save original markdown
            md_path = self.output_dir / "text" / f"{Path(xlsx_path).stem}_markdown.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(result.markdown)
            
            # Analyze and parse different content types
            self._parse_sheets_and_tables(result.markdown, parsed_content, xlsx_path)
            self._parse_image_references(result.markdown, parsed_content, xlsx_path)
            self._create_statistics(parsed_content)
            
            return parsed_content
            
        except Exception as e:
            print(f"Error processing {xlsx_path}: {e}")
            return {
                "text_content": {"full_markdown": ""},
                "sheets": [],
                "tables": [],
                "images": [],
                "metadata": {
                    "source": xlsx_path,
                    "conversion_successful": False,
                    "error": str(e)
                }
            }
    
    def _parse_sheets_and_tables(self, markdown_content: str, parsed_content: Dict, xlsx_path: str):
        """Parse sheets and tables from Excel file converted to Markdown"""
        
        # MarkItDown typically outputs Excel sheets with headers like "## Sheet Name"
        # Split by sheet headers
        sheet_pattern = r'^##\s+(.+?)$'
        sheet_matches = list(re.finditer(sheet_pattern, markdown_content, re.MULTILINE))
        
        sheets = []
        tables = []
        table_counter = 0
        
        if sheet_matches:
            # Process each sheet
            for i, match in enumerate(sheet_matches):
                sheet_name = match.group(1).strip()
                start_pos = match.end()
                end_pos = sheet_matches[i + 1].start() if i + 1 < len(sheet_matches) else len(markdown_content)
                sheet_content = markdown_content[start_pos:end_pos].strip()
                
                # Parse tables within this sheet
                sheet_tables = self._extract_tables_from_content(sheet_content, sheet_name, xlsx_path, table_counter)
                table_counter += len(sheet_tables)
                tables.extend(sheet_tables)
                
                # Analyze sheet content
                sheet_info = {
                    "id": f"sheet_{i+1}",
                    "name": sheet_name,
                    "content": sheet_content,
                    "table_count": len(sheet_tables),
                    "table_ids": [t["id"] for t in sheet_tables],
                    "row_count": sum(t.get("row_count", 0) for t in sheet_tables),
                    "has_data": len(sheet_tables) > 0
                }
                
                # Save individual sheet content
                sheet_path = self.output_dir / "sheets" / f"{Path(xlsx_path).stem}_sheet_{i+1}_{sheet_name.replace(' ', '_')}.md"
                with open(sheet_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {sheet_name}\n\n{sheet_content}")
                
                sheet_info["markdown_file"] = str(sheet_path)
                sheets.append(sheet_info)
            
            print(f"✓ Parsed {len(sheets)} sheets with {len(tables)} total tables")
        else:
            # No clear sheet headers, treat entire content as tables
            print("No sheet headers found, parsing as single data source...")
            tables = self._extract_tables_from_content(markdown_content, "Sheet1", xlsx_path, 0)
            
            if tables:
                sheets.append({
                    "id": "sheet_1",
                    "name": "Sheet1",
                    "content": markdown_content,
                    "table_count": len(tables),
                    "table_ids": [t["id"] for t in tables],
                    "row_count": sum(t.get("row_count", 0) for t in tables),
                    "has_data": True
                })
            
            print(f"✓ Parsed {len(tables)} tables from content")
        
        parsed_content["sheets"] = sheets
        parsed_content["tables"] = tables
    
    def _extract_tables_from_content(self, content: str, sheet_name: str, xlsx_path: str, 
                                     start_counter: int) -> List[Dict[str, Any]]:
        """Extract tables from markdown content"""
        tables = []
        
        # Find markdown tables using regex
        table_pattern = r'(\|[^\n]+\|[\n\r]+\|[-\s\|:]+\|[\n\r]+(\|[^\n]*\|[\n\r]*)*)'
        table_matches = re.findall(table_pattern, content, re.MULTILINE)
        
        for idx, match in enumerate(table_matches):
            table_counter = start_counter + idx + 1
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
                    csv_path = self.output_dir / "tables" / f"{Path(xlsx_path).stem}_{sheet_name.replace(' ', '_')}_table_{table_counter}.csv"
                    df.to_csv(csv_path, index=False)
                    
                    # Save original markdown table
                    md_path = self.output_dir / "tables" / f"{Path(xlsx_path).stem}_{sheet_name.replace(' ', '_')}_table_{table_counter}.md"
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(table_markdown)
                    
                    # Calculate basic statistics
                    stats = self._calculate_table_statistics(df)
                    
                    table_info = {
                        "id": f"table_{table_counter}",
                        "sheet_name": sheet_name,
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
                print(f"Error processing table {table_counter} in {sheet_name}: {e}")
                tables.append({
                    "id": f"table_{table_counter}",
                    "sheet_name": sheet_name,
                    "markdown": table_markdown,
                    "error": str(e)
                })
        
        return tables
    
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
    
    def _parse_image_references(self, markdown_content: str, parsed_content: Dict, xlsx_path: str):
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
                image_info["llm_description"] = alt_text
                image_info["enhanced"] = True
            
            images.append(image_info)
        
        # Look for chart/image descriptions that might be in text format
        image_description_patterns = [
            r'\[Image:\s*([^\]]+)\]',
            r'\[Chart:\s*([^\]]+)\]',
            r'\[Figure\s*\d*:?\s*([^\]]+)\]',
            r'\[Graph:\s*([^\]]+)\]'
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
    
    def _create_statistics(self, parsed_content: Dict):
        """Create overall statistics for the Excel file"""
        stats = {
            "total_sheets": len(parsed_content["sheets"]),
            "total_tables": len(parsed_content["tables"]),
            "total_images": len(parsed_content["images"]),
            "total_rows": sum(t.get("row_count", 0) for t in parsed_content["tables"]),
            "total_columns": sum(t.get("col_count", 0) for t in parsed_content["tables"]),
            "sheets_with_data": len([s for s in parsed_content["sheets"] if s.get("has_data")])
        }
        
        parsed_content["statistics"] = stats
        print(f"✓ Statistics: {stats['total_sheets']} sheets, {stats['total_tables']} tables, {stats['total_rows']} rows")
    
    def create_comprehensive_chunks(self, parsed_content: Dict, xlsx_path: str) -> List[Dict[str, Any]]:
        """Create chunks from parsed Excel content"""
        chunks = []
        
        # Create sheet-level chunks
        for sheet in parsed_content.get("sheets", []):
            if sheet.get("content"):
                # Create a chunk for the entire sheet
                sheet_text = f"Sheet: {sheet['name']}\n\n{sheet['content']}"
                
                # Split if too large
                if len(sheet_text) > 1500:
                    sub_chunks = self.char_splitter.split_text(sheet_text)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_obj = {
                            "id": f"{Path(xlsx_path).stem}_sheet_{sheet['id']}_chunk_{j}",
                            "type": "sheet",
                            "content": sub_chunk,
                            "embedding": self.embedder.encode(sub_chunk).tolist(),
                            "metadata": {
                                "content_type": "sheet",
                                "sheet_id": sheet["id"],
                                "sheet_name": sheet["name"],
                                "chunk_index": j,
                                "chunk_size": len(sub_chunk),
                                "source": xlsx_path
                            }
                        }
                        chunks.append(chunk_obj)
                else:
                    chunk_obj = {
                        "id": f"{Path(xlsx_path).stem}_sheet_{sheet['id']}",
                        "type": "sheet",
                        "content": sheet_text,
                        "embedding": self.embedder.encode(sheet_text).tolist(),
                        "metadata": {
                            "content_type": "sheet",
                            "sheet_id": sheet["id"],
                            "sheet_name": sheet["name"],
                            "chunk_size": len(sheet_text),
                            "source": xlsx_path
                        }
                    }
                    chunks.append(chunk_obj)
        
        # Add table chunks
        table_chunks = self._create_table_chunks(parsed_content, xlsx_path)
        chunks.extend(table_chunks)
        
        # Add image description chunks
        image_chunks = self._create_image_chunks(parsed_content, xlsx_path)
        chunks.extend(image_chunks)
        
        print(f"✓ Created {len(chunks)} comprehensive chunks")
        return chunks
    
    def _create_table_chunks(self, parsed_content: Dict, xlsx_path: str) -> List[Dict[str, Any]]:
        """Create chunks for tables"""
        chunks = []
        
        for table in parsed_content.get("tables", []):
            if table.get("markdown"):
                table_text = f"Table {table['id']} from {table.get('sheet_name', 'Unknown Sheet')}\n{table['markdown']}"
                
                chunk_obj = {
                    "id": f"{Path(xlsx_path).stem}_table_chunk_{table['id']}",
                    "type": "table",
                    "content": table_text,
                    "embedding": self.embedder.encode(table_text).tolist(),
                    "metadata": {
                        "content_type": "table",
                        "table_id": table["id"],
                        "sheet_name": table.get("sheet_name", "Unknown"),
                        "row_count": table.get("row_count", 0),
                        "col_count": table.get("col_count", 0),
                        "csv_file": table.get("csv_file"),
                        "markdown_file": table.get("markdown_file"),
                        "source": xlsx_path
                    }
                }
                chunks.append(chunk_obj)
        
        return chunks
    
    def _create_image_chunks(self, parsed_content: Dict, xlsx_path: str) -> List[Dict[str, Any]]:
        """Create chunks for image descriptions (charts, etc.)"""
        chunks = []
        
        for image in parsed_content.get("images", []):
            description = image.get("llm_description") or image.get("description") or image.get("alt_text")
            if description and description.strip():
                image_text = f"Image/Chart {image['id']}: {description}"
                
                chunk_obj = {
                    "id": f"{Path(xlsx_path).stem}_image_chunk_{image['id']}",
                    "type": "image",
                    "content": image_text,
                    "embedding": self.embedder.encode(image_text).tolist(),
                    "metadata": {
                        "content_type": "image", 
                        "image_id": image["id"],
                        "image_type": image.get("type", "unknown"),
                        "enhanced": image.get("enhanced", False),
                        "source": xlsx_path
                    }
                }
                chunks.append(chunk_obj)
        
        return chunks
    
    def save_parsing_summary(self, parsed_content: Dict, xlsx_path: str):
        """Save parsing summary"""
        stats = parsed_content.get("statistics", {})
        
        summary = {
            "document": Path(xlsx_path).name,
            "parsing_summary": {
                "sheets": {
                    "count": stats.get("total_sheets", 0),
                    "with_data": stats.get("sheets_with_data", 0),
                    "names": [s["name"] for s in parsed_content.get("sheets", [])]
                },
                "tables": {
                    "count": stats.get("total_tables", 0),
                    "total_rows": stats.get("total_rows", 0),
                    "total_columns": stats.get("total_columns", 0),
                    "files": [t.get("csv_file") for t in parsed_content.get("tables", []) if t.get("csv_file")]
                },
                "images": {
                    "count": stats.get("total_images", 0),
                    "with_descriptions": len([img for img in parsed_content.get("images", []) if img.get("description") or img.get("llm_description")]),
                    "enhanced_with_llm": len([img for img in parsed_content.get("images", []) if img.get("enhanced")])
                }
            },
            "metadata": parsed_content.get("metadata", {})
        }
        
        summary_path = self.output_dir / f"{Path(xlsx_path).stem}_markitdown_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved parsing summary to {summary_path}")
        return summary

# Usage function
def process_xlsx_with_markitdown(xlsx_path: str, openai_api_key: Optional[str] = None, 
                                 output_dir: str = "./markitdown_xlsx_output"):
    """
    Complete workflow for comprehensive XLSX processing with MarkItDown
    
    Args:
        xlsx_path: Path to the XLSX file to process
        openai_api_key: Optional OpenAI API key. If not provided, will try to load from OPENAI_API_KEY environment variable
        output_dir: Directory to save the output files
    """
    # Get API key from parameter or environment variable
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    
    # Initialize with or without OpenAI
    openai_client = OpenAI(api_key=api_key) if api_key else None
    processor = MarkItDownXLSXProcessor(openai_client=openai_client, output_dir=output_dir)
    
    # 1. Parse all content types
    parsed_content = processor.parse_comprehensive_content(xlsx_path)
    if not parsed_content:
        return None
    
    # 2. Create comprehensive chunks
    # chunks = processor.create_comprehensive_chunks(parsed_content, xlsx_path)
    
    # 3. Save summary
    summary = processor.save_parsing_summary(parsed_content, xlsx_path)
    
    # 4. Store in vector database (optional)
#    client = chromadb.PersistentClient(path="./chroma_db")
#    collection_name = f"markitdown_xlsx_{Path(xlsx_path).stem}"
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
# Basic usage (no LLM for images/charts)
result = process_xlsx_with_markitdown("docs\\DO_NOT_Adatkor.xlsx", openai_api_key=None, output_dir="./markitdown_xlsx_output")

# With OpenAI for enhanced image/chart processing (API key from .env file)
# result = process_xlsx_with_markitdown("docs\\DO_NOT_Adatkor.xlsx")

# With OpenAI for enhanced processing (API key passed directly)
# result = process_xlsx_with_markitdown("docs\\DO_NOT_Adatkor.xlsx", openai_api_key="sk-proj-1234567890")


print("\nMARKITDOWN XLSX Parser Created")
print("=" * 70)
print("Features:")
print("✓ Comprehensive XLSX/Excel parsing with sheet detection")
print("✓ TABLE parsing from each sheet with CSV conversion")
print("✓ Multiple sheets support with individual markdown files")
print("✓ IMAGE/CHART reference parsing with optional LLM descriptions")
print("✓ Statistical analysis of numeric and text columns")
print("✓ Markdown structure preservation")
print("✓ Optional OpenAI integration for better chart understanding")
print("✓ Efficient processing of large Excel files")
print("\nNote: MarkItDown converts Excel to Markdown format, which is")
print("excellent for text-based processing and searching. For complex")
print("Excel features (formulas, formatting), consider using specialized")
print("libraries like openpyxl or pandas directly.")

