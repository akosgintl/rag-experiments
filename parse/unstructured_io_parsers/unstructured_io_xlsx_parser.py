# =================================================================
# UNSTRUCTURED.IO - COMPREHENSIVE EXCEL (XLSX) PARSING
# =================================================================

# Installation with full capabilities
# pip install unstructured[xlsx] sentence-transformers chromadb
# pip install pillow pandas openpyxl xlrd msoffcrypto-tool

import os
import json
import base64
import shutil
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from PIL import Image
import io
import torch
import openpyxl
from openpyxl.drawing.image import Image as XLImage

from unstructured.partition.xlsx import partition_xlsx
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_to_json, dict_to_elements
from sentence_transformers import SentenceTransformer
import chromadb

def get_available_device(preferred_device: str = "cuda") -> str:
    """
    Check device availability and return the best available device.
    
    Args:
        preferred_device: Preferred device ("cuda", "cpu", or "auto")
    
    Returns:
        Available device string ("cuda" or "cpu")
    """
    if preferred_device == "cpu":
        return "cpu"
    
    if preferred_device in ["cuda", "auto"]:
        if torch.cuda.is_available():
            print(f"✓ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            print("⚠ CUDA requested but not available. Falling back to CPU.")
            print("  To enable CUDA, install PyTorch with CUDA support:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return "cpu"
    
    return "cpu"

class UnstructuredXlsxProcessor:
    def __init__(
        self, 
        embedding_model="all-MiniLM-L6-v2", 
        output_dir="./unstructured_output",
        device="auto"  # "cpu", "cuda", or "auto" (auto-detect best available)
    ):
        """
        Initialize the UnstructuredXlsxProcessor
        
        Args:
            embedding_model: Name of the SentenceTransformer model
            output_dir: Directory to save parsed content
            device: Device to use for processing ("cpu", "cuda", or "auto")
        """
        # Auto-detect or validate device availability
        self.device = get_available_device(device)
        
        # Initialize embedder with device support
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "sheets").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        
        print(f"✓ Initialized XLSX processor with device: {self.device}")
    
    def parse_comprehensive_content(self, xlsx_path: str) -> Dict[str, Any]:
        """
        Parse SHEETS, TABLES, and IMAGES comprehensively from XLSX
        """
        try:
            print(f"Processing {xlsx_path} with combined approach...")
            
            parsed_content = {
                "sheets": [],
                "tables": [],
                "images": [],
                "text_content": [],
                "metadata": {
                    "source": xlsx_path,
                    "title": Path(xlsx_path).stem
                }
            }
            
            # Try unstructured parsing first
            try:
                self._parse_with_unstructured(xlsx_path, parsed_content)
            except Exception as e:
                print(f"Note: Unstructured parsing encountered issues: {e}")
            
            # Always use pandas for comprehensive sheet parsing
            self._parse_with_pandas(xlsx_path, parsed_content)
            
            # Extract images using openpyxl
            self._extract_images_with_openpyxl(xlsx_path, parsed_content)
            
            return parsed_content
            
        except Exception as e:
            print(f"Error processing {xlsx_path}: {e}")
            return None
    
    def _parse_with_unstructured(self, xlsx_path: str, parsed_content: Dict):
        """Parse with Unstructured.io"""
        try:
            print(f"  Attempting Unstructured.io parsing...")
            
            # Parse with unstructured
            elements = partition_xlsx(
                filename=xlsx_path,
                include_metadata=True
            )
            
            if not elements:
                print("  No elements parsed by Unstructured")
                return
            
            parsed_content["metadata"]["unstructured_elements"] = len(elements)
            
            # Process elements
            element_types_found = set()
            table_counter = 0
            
            for i, element in enumerate(elements):
                element_type = type(element).__name__
                element_types_found.add(element_type)
                
                # Get metadata
                metadata = getattr(element, 'metadata', None)
                
                # TEXT PARSING
                if element_type in ["Title", "NarrativeText", "Header", "Footer", "ListItem", 
                        "UncategorizedText", "Text"]:
                    text_info = {
                        "text": element.text,
                        "type": element_type,
                        "element_id": getattr(element, 'id', f"text_{i}"),
                        "sheet_name": getattr(metadata, 'page_name', None) if metadata else None
                    }
                    parsed_content["text_content"].append(text_info)
                
                # TABLE PARSING
                elif element_type == "Table":
                    table_counter += 1
                    table_info = self._parse_table_element(element, table_counter, xlsx_path, metadata)
                    if table_info:
                        parsed_content["tables"].append(table_info)
            
            print(f"  DEBUG: Unstructured element types found: {sorted(element_types_found)}")
            print(f"  ✓ Parsed {len(parsed_content['text_content'])} text elements via Unstructured")
            print(f"  ✓ Parsed {table_counter} tables via Unstructured")
            
        except Exception as e:
            print(f"  Unstructured parsing failed: {e}")
    
    def _parse_table_element(self, table_element, table_counter: int, xlsx_path: str, metadata) -> Dict:
        """Parse table from unstructured element"""
        try:
            table_info = {
                "id": f"unstructured_table_{table_counter}",
                "text": table_element.text,
                "element_id": getattr(table_element, 'id', f"table_{table_counter}"),
                "sheet_name": getattr(metadata, 'page_name', None) if metadata else None,
                "source": "unstructured"
            }
            
            # Get HTML structure if available
            if metadata and hasattr(metadata, 'text_as_html'):
                html_content = metadata.text_as_html
                table_info["html"] = html_content
                
                # Save HTML table
                html_path = self.output_dir / "tables" / f"{Path(xlsx_path).stem}_unstructured_table_{table_counter}.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(f"<html><body>{html_content}</body></html>")
                table_info["html_file"] = str(html_path)
                
                # Convert HTML to DataFrame and CSV if possible
                try:
                    df = pd.read_html(io.StringIO(html_content))[0]
                    csv_content = df.to_csv(index=False)
                    
                    # Save as CSV
                    csv_path = self.output_dir / "tables" / f"{Path(xlsx_path).stem}_unstructured_table_{table_counter}.csv"
                    df.to_csv(csv_path, index=False)
                    
                    table_info.update({
                        "csv": csv_content,
                        "csv_file": str(csv_path),
                        "row_count": len(df),
                        "col_count": len(df.columns),
                        "columns": df.columns.tolist()
                    })
                except Exception as e:
                    print(f"  Could not convert table {table_counter} to DataFrame: {e}")
            
            return table_info
            
        except Exception as e:
            print(f"  Error processing table {table_counter}: {e}")
            return {
                "id": f"unstructured_table_{table_counter}",
                "text": getattr(table_element, 'text', ''),
                "error": str(e)
            }
    
    def _parse_with_pandas(self, xlsx_path: str, parsed_content: Dict):
        """Parse Excel sheets with pandas for comprehensive data extraction"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(xlsx_path)
            sheet_names = excel_file.sheet_names
            
            print(f"✓ Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")
            
            for sheet_name in sheet_names:
                try:
                    # Read sheet
                    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
                    
                    # Skip empty sheets
                    if df.empty:
                        print(f"  - Skipping empty sheet: {sheet_name}")
                        continue
                    
                    sheet_info = {
                        "sheet_name": sheet_name,
                        "row_count": len(df),
                        "col_count": len(df.columns),
                        "columns": df.columns.tolist(),
                        "data": df.to_dict(),
                        "csv": df.to_csv(index=False)
                    }
                    
                    # Save sheet as CSV
                    csv_path = self.output_dir / "sheets" / f"{Path(xlsx_path).stem}_sheet_{sheet_name}.csv"
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    sheet_info["csv_file"] = str(csv_path)
                    
                    # Save sheet as markdown
                    md_path = self.output_dir / "sheets" / f"{Path(xlsx_path).stem}_sheet_{sheet_name}.md"
                    sheet_markdown = f"# Sheet: {sheet_name}\n\n"
                    sheet_markdown += f"**Source:** {Path(xlsx_path).name}\n"
                    sheet_markdown += f"**Dimensions:** {len(df)} rows × {len(df.columns)} columns\n\n"
                    sheet_markdown += "## Column Names\n\n"
                    sheet_markdown += "- " + "\n- ".join(df.columns.astype(str).tolist()) + "\n\n"
                    sheet_markdown += "## Data\n\n"
                    
                    try:
                        sheet_markdown += df.to_markdown(index=False)
                    except Exception as e:
                        sheet_markdown += f"(Table too large or complex for markdown: {e})\n\n"
                        sheet_markdown += f"See CSV file: {csv_path.name}"
                    
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(sheet_markdown)
                    sheet_info["markdown_file"] = str(md_path)
                    
                    # Create text summary
                    txt_path = self.output_dir / "sheets" / f"{Path(xlsx_path).stem}_sheet_{sheet_name}.txt"
                    text_summary = f"Sheet: {sheet_name}\n"
                    text_summary += f"{'=' * 60}\n\n"
                    text_summary += f"Source: {Path(xlsx_path).name}\n"
                    text_summary += f"Dimensions: {len(df)} rows × {len(df.columns)} columns\n\n"
                    text_summary += f"Columns:\n"
                    for col in df.columns:
                        text_summary += f"  - {col}\n"
                    text_summary += f"\nData preview (first 10 rows):\n"
                    text_summary += df.head(10).to_string(index=False)
                    
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text_summary)
                    sheet_info["text_file"] = str(txt_path)
                    
                    parsed_content["sheets"].append(sheet_info)
                    print(f"  ✓ Parsed sheet '{sheet_name}': {len(df)} rows × {len(df.columns)} columns")
                    
                except Exception as e:
                    print(f"  ✗ Error parsing sheet '{sheet_name}': {e}")
            
        except Exception as e:
            print(f"Error in pandas parsing: {e}")
    
    def _extract_images_with_openpyxl(self, xlsx_path: str, parsed_content: Dict):
        """Extract embedded images using openpyxl"""
        try:
            workbook = openpyxl.load_workbook(xlsx_path)
            image_counter = 0
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                
                # Check for images in the sheet
                if hasattr(sheet, '_images') and sheet._images:
                    for img in sheet._images:
                        image_counter += 1
                        
                        try:
                            # Get image data
                            image_data = img._data()
                            
                            # Save image
                            image_path = self.output_dir / "images" / f"{Path(xlsx_path).stem}_image_{image_counter}.png"
                            
                            # Convert to PIL Image and save
                            pil_image = Image.open(io.BytesIO(image_data))
                            pil_image.save(image_path, "PNG")
                            
                            # Get anchor info (position in sheet)
                            anchor_info = ""
                            if hasattr(img, 'anchor'):
                                if hasattr(img.anchor, '_from'):
                                    anchor_info = f"Cell: {img.anchor._from.col}, {img.anchor._from.row}"
                            
                            image_info = {
                                "type": "embedded_image",
                                "id": f"image_{image_counter}",
                                "sheet": sheet_name,
                                "file_path": str(image_path),
                                "size": pil_image.size,
                                "position": anchor_info
                            }
                            
                            # Save image metadata
                            txt_path = self.output_dir / "images" / f"{Path(xlsx_path).stem}_image_{image_counter}.txt"
                            image_text = f"Image {image_counter}\n"
                            image_text += f"{'=' * 60}\n\n"
                            image_text += f"Source: {Path(xlsx_path).name}\n"
                            image_text += f"Sheet: {sheet_name}\n"
                            image_text += f"Size: {pil_image.size[0]}x{pil_image.size[1]} pixels\n"
                            if anchor_info:
                                image_text += f"Position: {anchor_info}\n"
                            
                            with open(txt_path, 'w', encoding='utf-8') as f:
                                f.write(image_text)
                            image_info["text_file"] = str(txt_path)
                            
                            parsed_content["images"].append(image_info)
                            
                        except Exception as e:
                            print(f"  Could not extract image {image_counter}: {e}")
            
            if image_counter > 0:
                print(f"✓ Extracted {image_counter} embedded images")
            
        except Exception as e:
            print(f"Note: Could not extract images with openpyxl: {e}")
    
    def create_comprehensive_chunks(self, parsed_content: Dict, xlsx_path: str) -> List[Dict[str, Any]]:
        """Create chunks incorporating sheets, tables, and images"""
        chunks = []
        
        # Sheet chunks
        for sheet in parsed_content.get("sheets", []):
            if sheet.get("csv"):
                # Create chunk for sheet
                sheet_text = f"Sheet: {sheet['sheet_name']}\n"
                sheet_text += f"Dimensions: {sheet['row_count']} rows × {sheet['col_count']} columns\n\n"
                sheet_text += f"Columns: {', '.join(sheet['columns'])}\n\n"
                
                # Include a preview of the CSV data
                csv_preview = sheet['csv'][:2000] if len(sheet['csv']) > 2000 else sheet['csv']
                sheet_text += f"Data (CSV format):\n{csv_preview}"
                if len(sheet['csv']) > 2000:
                    sheet_text += "\n... (truncated)"
                
                chunk_obj = {
                    "id": f"{Path(xlsx_path).stem}_sheet_{sheet['sheet_name']}",
                    "type": "sheet",
                    "content": sheet_text,
                    "embedding": self.embedder.encode(sheet_text).tolist(),
                    "metadata": {
                        "content_type": "sheet",
                        "sheet_name": sheet["sheet_name"],
                        "row_count": sheet["row_count"],
                        "col_count": sheet["col_count"],
                        "csv_file": sheet.get("csv_file"),
                        "markdown_file": sheet.get("markdown_file"),
                        "source": xlsx_path
                    }
                }
                chunks.append(chunk_obj)
        
        # Table chunks (from Unstructured if available)
        for table in parsed_content.get("tables", []):
            if table.get("text") or table.get("csv"):
                table_text = f"Table {table['id']}"
                if table.get("sheet_name"):
                    table_text += f" from sheet '{table['sheet_name']}'"
                table_text += "\n"
                
                if table.get("text"):
                    table_text += f"\n{table['text']}"
                    
                if table.get("csv"):
                    csv_preview = table['csv'][:1000] if len(table['csv']) > 1000 else table['csv']
                    table_text += f"\n\nTable Data (CSV format):\n{csv_preview}"
                    if len(table['csv']) > 1000:
                        table_text += "\n... (truncated)"
                
                chunk_obj = {
                    "id": f"{Path(xlsx_path).stem}_table_{table['id']}",
                    "type": "table",
                    "content": table_text,
                    "embedding": self.embedder.encode(table_text).tolist(),
                    "metadata": {
                        "content_type": "table",
                        "table_id": table["id"],
                        "sheet_name": table.get("sheet_name"),
                        "row_count": table.get("row_count", 0),
                        "col_count": table.get("col_count", 0),
                        "csv_file": table.get("csv_file"),
                        "source": xlsx_path
                    }
                }
                chunks.append(chunk_obj)
        
        # Image chunks
        for image in parsed_content.get("images", []):
            if image.get("text_file"):
                image_text = f"Image {image['id']} from sheet '{image.get('sheet', 'unknown')}'"
                if image.get("position"):
                    image_text += f" at {image['position']}"
                
                chunk_obj = {
                    "id": f"{Path(xlsx_path).stem}_image_{image['id']}",
                    "type": "image",
                    "content": image_text,
                    "embedding": self.embedder.encode(image_text).tolist(),
                    "metadata": {
                        "content_type": "image",
                        "image_id": image["id"],
                        "sheet": image.get("sheet"),
                        "image_file": image["file_path"],
                        "text_file": image.get("text_file"),
                        "image_size": image.get("size"),
                        "source": xlsx_path
                    }
                }
                chunks.append(chunk_obj)
        
        print(f"✓ Created {len(chunks)} comprehensive chunks")
        return chunks
    
    def save_parsing_summary(self, parsed_content: Dict, xlsx_path: str):
        """Save detailed parsing summary"""
        summary = {
            "document": Path(xlsx_path).name,
            "parsing_summary": {
                "text": {
                    "elements_count": len(parsed_content.get("text_content", [])),
                    "types": list(set([elem.get("type", "Unknown") for elem in parsed_content.get("text_content", [])]))
                },
                "sheets": {
                    "count": len(parsed_content.get("sheets", [])),
                    "names": [s["sheet_name"] for s in parsed_content.get("sheets", [])],
                    "csv_files": [s.get("csv_file") for s in parsed_content.get("sheets", []) if s.get("csv_file")],
                    "markdown_files": [s.get("markdown_file") for s in parsed_content.get("sheets", []) if s.get("markdown_file")]
                },
                "tables": {
                    "count": len(parsed_content.get("tables", [])),
                    "with_html": len([t for t in parsed_content.get("tables", []) if t.get("html")]),
                    "with_csv": len([t for t in parsed_content.get("tables", []) if t.get("csv_file")]),
                    "files": {
                        "csv": [t.get("csv_file") for t in parsed_content.get("tables", []) if t.get("csv_file")],
                        "html": [t.get("html_file") for t in parsed_content.get("tables", []) if t.get("html_file")]
                    }
                },
                "images": {
                    "count": len(parsed_content.get("images", [])),
                    "with_files": len([img for img in parsed_content.get("images", []) if img.get("file_path")]),
                    "files": [img.get("file_path") for img in parsed_content.get("images", []) if img.get("file_path")]
                }
            },
            "metadata": parsed_content.get("metadata", {})
        }
        
        summary_path = self.output_dir / f"{Path(xlsx_path).stem}_unstructured_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved parsing summary to {summary_path}")
        return summary

# Usage function
def process_xlsx_with_unstructured_advanced(
    xlsx_path: str, 
    output_dir: str = "./unstructured_output",
    device: str = "auto"  # "auto" (recommended), "cpu", or "cuda"
):
    """
    Complete workflow for comprehensive XLSX processing with Unstructured.io
    
    Args:
        xlsx_path: Path to the XLSX file
        output_dir: Directory to save parsed content
        device: "auto" (auto-detect), "cpu", or "cuda" for GPU acceleration
    """
    processor = UnstructuredXlsxProcessor(
        output_dir=output_dir, 
        device=device
    )
    
    # 1. Parse all content types
    parsed_content = processor.parse_comprehensive_content(xlsx_path)
    if not parsed_content:
        return None
    
    # 2. Create comprehensive chunks
    # chunks = processor.create_comprehensive_chunks(parsed_content, xlsx_path)
    
    # 3. Save summary
    summary = processor.save_parsing_summary(parsed_content, xlsx_path)
    
    # 4. Store in vector database (optional)
    # client = chromadb.PersistentClient(path="./chroma_db")
    # collection_name = f"unstructured_comprehensive_{Path(xlsx_path).stem}"
    # collection = client.get_or_create_collection(name=collection_name)
    
    # valid_chunks = [c for c in chunks if c.get("embedding")]
    # if valid_chunks:
    #     collection.upsert(
    #         documents=[c["content"] for c in valid_chunks],
    #         embeddings=[c["embedding"] for c in valid_chunks],
    #         ids=[c["id"] for c in valid_chunks],
    #         metadatas=[c["metadata"] for c in valid_chunks]
    #     )
    #     print(f"✓ Stored {len(valid_chunks)} chunks in ChromaDB")
    
    return {
        "parsed_content": parsed_content,
        # "chunks": chunks,
        "summary": summary,
        "output_directory": output_dir
    }

# Usage:
if __name__ == "__main__":
    # Example: Process XLSX file with auto device detection (recommended)
    result = process_xlsx_with_unstructured_advanced(
        xlsx_path="./docs/DO_NOT_KovSpec.xlsx",
        device="auto"  # Auto-detect best device (CUDA if available, else CPU)
    )
    
    # Example 2: Force CPU usage
    # result = process_xlsx_with_unstructured_advanced(
    #     xlsx_path="./docs/DO_NOT_KovSpec.xlsx",
    #     device="cpu"   # Force CPU usage
    # )
    
    print("\n" + "=" * 70)
    print("UNSTRUCTURED.IO XLSX Parser - Processing Complete!")
    print("=" * 70)
    print("Features:")
    print("✓ Comprehensive SHEET parsing:")
    print("  - All sheets extracted with pandas")
    print("  - CSV export for each sheet in 'sheets/' folder")
    print("  - Markdown files with formatted sheet content")
    print("  - Text summaries with data previews")
    print("✓ TABLE parsing (via Unstructured.io if supported):")
    print("  - HTML structure and CSV conversion")
    print("✓ IMAGE extraction:")
    print("  - Embedded images extracted with openpyxl")
    print("  - Image metadata and position information")
    print("✓ Structured chunking for all content types")
    print("✓ Organized output directory structure")
    print("✓ Detailed parsing summaries")
    print(f"✓ Device support: CPU and CUDA")
    print("=" * 70)
    print("\nOutput Structure:")
    print("  unstructured_output/")
    print("  ├── sheets/    (CSV, markdown, and text for each sheet)")
    print("  ├── tables/    (parsed tables if available)")
    print("  ├── images/    (embedded images)")
    print("  └── text/      (text elements if available)")
    print("=" * 70)

