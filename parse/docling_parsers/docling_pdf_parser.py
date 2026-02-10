# =========================================================
# DOCLING - COMPREHENSIVE TEXT, TABLE, AND IMAGE PARSING
# =========================================================
#
# This script uses the latest Docling API (updated 2025):
# - TableItem.export_to_dataframe(document) - requires document parameter
# - TableItem.export_to_markdown(document) - for parsing table text
# - No warnings or errors with current Docling version
#
# Installation with full capabilities:
# pip install docling[vlm] docling-core sentence-transformers chromadb
# pip install pillow pandas openpyxl tabulate

import os
import json
import base64
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from PIL import Image
import io
import torch

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractOcrOptions, RapidOcrOptions, OcrMacOptions, TesseractCliOcrOptions, EasyOcrOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling_core.types.doc.document import DoclingDocument
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
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

class DoclingAdvancedProcessor:
    def __init__(
        self, 
        embedding_model="all-MiniLM-L6-v2", 
        output_dir="./docling_pdf_output",
        device="auto",  # "cpu", "cuda", or "auto" (auto-detect best available)
        do_ocr=False  # Set to False for text-based PDFs to avoid OCR overhead
    ):
        """
        Initialize the DoclingAdvancedProcessor
        
        Args:
            embedding_model: Name of the SentenceTransformer model
            output_dir: Directory to save parsed content
            device: Device to use for processing ("cpu", "cuda", or "auto")
            do_ocr: Whether to perform OCR (set False for text-based PDFs)
        """
        # Auto-detect or validate device availability
        self.device = get_available_device(device)
        
        # Configure pipeline for comprehensive parsing
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.images_scale = 2  # Normal resolution (was 1.0 - too slow)
        self.pipeline_options.generate_page_images = True
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.generate_table_images = True  # Usually not needed, very slow
        self.pipeline_options.do_ocr = do_ocr  # Control OCR
        # Any of the OCR options can be used: EasyOcrOptions, TesseractOcrOptions, TesseractCliOcrOptions, OcrMacOptions (macOS only), RapidOcrOptions
        # ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=["hu"])
        # ocr_options = TesseractOcrOptions(force_full_page_ocr=True, lang=["hu"])
        # ocr_options = OcrMacOptions(force_full_page_ocr=True, lang=["hu"])
        # ocr_options = RapidOcrOptions(force_full_page_ocr=True, lang=["hu"])
        # ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=["hu"])
        # self.pipeline_options.ocr_options = ocr_options
        self.pipeline_options.do_table_structure = True  # Control table structure parsing
        self.pipeline_options.table_structure_options.do_cell_matching = True  # Control cell matching
        self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.pipeline_options.accelerator_options = AcceleratorOptions(num_threads=8, device=device)  # Increased from 8 to 16 for better performance
        
        # Initialize converter with pipeline options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        
        # Initialize embedder with device support
        # self.embedder = SentenceTransformer(embedding_model, device=self.device)
        # self.text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=10000000,
        #     chunk_overlap=100000
        # )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for organized output
        self.tables_dir = self.output_dir / "tables"
        self.pages_dir = self.output_dir / "pages"
        self.figures_dir = self.output_dir / "figures"
        self.text_dir = self.output_dir / "text"
        
        self.tables_dir.mkdir(exist_ok=True, parents=True)
        self.pages_dir.mkdir(exist_ok=True, parents=True)
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        self.text_dir.mkdir(exist_ok=True, parents=True)

        print(f"✓ Initialized with device: {self.device}, OCR: {do_ocr}")
    
    def parse_comprehensive_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse TEXT, TABLES, and IMAGES comprehensively from PDF
        """
        try:
            print(f"Processing {pdf_path} with Docling...")
            result = self.converter.convert(pdf_path)
            document = result.document
                       
            # Export and save JSON
            json_path = self.output_dir / f"{Path(pdf_path).stem}_full.json"
            document.save_as_json(json_path)
            html_path = self.output_dir / f"{Path(pdf_path).stem}_full.html"
            # Suppress harmless "Could not parse formula with MathML" warnings from docling's HTML serializer
            _html_logger = logging.getLogger("docling_core.transforms.serializer.html")
            _prev_level = _html_logger.level
            _html_logger.setLevel(logging.ERROR)
            try:
                document.save_as_html(html_path)
            finally:
                _html_logger.setLevel(_prev_level)
            markdown_path = self.output_dir / f"{Path(pdf_path).stem}_full.md"
            document.save_as_markdown(markdown_path)
            
            print(f"✓ Saved full JSON to {json_path}")

            parsed_content = {
                "text_content": [],
                "tables": [],
                "images": [],
                "metadata": {
                    "source": pdf_path,
                    "total_pages": len(document.pages) if hasattr(document, 'pages') else 0,
                    "title": getattr(document, 'title', Path(pdf_path).stem)
                }
            }
            
            # Parse different content types
            self._parse_text_content(document, parsed_content, pdf_path)
            self._parse_tables(document, parsed_content, pdf_path)
            self._parse_images(document, parsed_content, pdf_path)
            
            return parsed_content
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return None
    
    def _parse_text_content(self, document: DoclingDocument, parsed_content: Dict, pdf_path: str):
        """Parse and process text content"""
        # Get full text as markdown
        full_markdown = document.export_to_markdown()
        # Suppress harmless "Could not parse formula with MathML" warnings from docling's HTML serializer
        _html_logger = logging.getLogger("docling_core.transforms.serializer.html")
        _prev_level = _html_logger.level
        _html_logger.setLevel(logging.ERROR)
        try:
            full_html = document.export_to_html()
        finally:
            _html_logger.setLevel(_prev_level)
        full_text = document.export_to_text()
        parsed_content["text_content"] = {
            "full_markdown": full_markdown,
            "full_html": full_html,
            "full_text": full_text,
            "word_count": len(full_markdown.split()),
            "char_count": len(full_markdown)
        }
        
        # Parse text by elements for detailed analysis
        text_elements = []
        for element, level in document.iterate_items():
            if hasattr(element, 'text') and element.text.strip():
                # Get page number from provenance
                page_no = 'Unknown'
                if hasattr(element, 'prov') and element.prov:
                    prov_item = element.prov[0]
                    page_no = getattr(prov_item, 'page_no', 'Unknown')
                
                element_info = {
                    "text": element.text,
                    "type": type(element).__name__,
                    "level": level,
                    "page": page_no
                }
                text_elements.append(element_info)

        parsed_content["text_content"]["elements"] = text_elements

        # Save full markdown
        if full_markdown:
            md_path = self.text_dir / f"{Path(pdf_path).stem}_full_markdown.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(full_markdown)
            parsed_content["text_content"]["full_markdown_file"] = str(md_path)

        # Save full html
        if full_html:
            html_path = self.text_dir / f"{Path(pdf_path).stem}_full_html.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(full_html)
            parsed_content["text_content"]["full_html_file"] = str(html_path)

        # Save full text
        if full_text:
            txt_path = self.text_dir / f"{Path(pdf_path).stem}_full_text.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            parsed_content["text_content"]["full_text_file"] = str(txt_path)

        print(f"✓ Parsed {len(text_elements)} text elements")
    
    def _parse_tables(self, document: DoclingDocument, parsed_content: Dict, pdf_path: str):
        """Parse tables with structure preservation"""
        tables = []
        table_counter = 0
        
        for element, level in document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                
                # Parse table data
                try:
                    # Get DataFrame if possible (using document parameter for latest API)
                    df = None
                    table_data = None
                    table_csv = None
                    table_html = None
                    table_markdown = None
                    table_text = None
                    
                    # Export DataFrame (for CSV and dict data)
                    if hasattr(element, 'export_to_dataframe'):
                        try:
                            df = element.export_to_dataframe(document)
                            # Deduplicate column names to avoid data loss in to_dict()
                            if df.columns.duplicated().any():
                                cols = list(df.columns)
                                seen: Dict[str, int] = {}
                                for i, col in enumerate(cols):
                                    if col in seen:
                                        seen[col] += 1
                                        cols[i] = f"{col}_{seen[col]}"
                                    else:
                                        seen[col] = 0
                                df.columns = cols
                            table_data = df.to_dict()
                            table_csv = df.to_csv(index=False)
                        except Exception as e:
                            print(f"Warning: Could not export table {table_counter} to dataframe: {e}")
                    
                    # Use TableItem's own export methods for HTML/markdown/text
                    # (NOT pandas DataFrame methods, which expect a file buf as first arg)
                    if hasattr(element, 'export_to_html'):
                        try:
                            table_html = element.export_to_html(document)
                        except Exception as e:
                            print(f"Warning: Could not export table {table_counter} to HTML: {e}")
                    
                    if hasattr(element, 'export_to_markdown'):
                        try:
                            table_markdown = element.export_to_markdown(document)
                            # TableItem has no export_to_text; use markdown as text fallback
                            table_text = table_markdown
                        except Exception as e:
                            print(f"Warning: Could not export table {table_counter} to markdown: {e}")
                        
                    # Get table metadata from provenance
                    page_no = 'Unknown'
                    if hasattr(element, 'prov') and element.prov:
                        prov_item = element.prov[0]
                        page_no = getattr(prov_item, 'page_no', 'Unknown')
                    
                    # Get table text content (TableItem uses export_to_markdown for text)
                    table_info = {
                        "id": f"table_{table_counter}",
                        "text": table_text,
                        "data": table_data,
                        "csv": table_csv,
                        "html": table_html,
                        "markdown": table_markdown,
                        "page_number": page_no,
                        "bbox": getattr(element, 'bbox', None),
                        "row_count": len(df) if df is not None else 0,
                        "col_count": len(df.columns) if df is not None else 0
                    }
                    
                    # Save table as CSV
                    if table_csv:
                        csv_path = self.tables_dir / f"{Path(pdf_path).stem}_table_{table_counter}.csv"
                        with open(csv_path, 'w', encoding='utf-8') as f:
                            f.write(table_csv)
                        table_info["csv_file"] = str(csv_path)

                    # Save table as HTML
                    if table_html:
                        html_path = self.tables_dir / f"{Path(pdf_path).stem}_table_{table_counter}.html"
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(table_html)
                        table_info["html_file"] = str(html_path)

                    if table_text:
                        txt_path = self.tables_dir / f"{Path(pdf_path).stem}_table_{table_counter}.txt"
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(table_text)
                        table_info["text_file"] = str(txt_path)

                    if table_markdown:
                        md_path = self.tables_dir / f"{Path(pdf_path).stem}_table_{table_counter}.md"
                        with open(md_path, 'w', encoding='utf-8') as f:
                            f.write(table_markdown)
                        table_info["markdown_file"] = str(md_path)
                    
                    # Save table image if available
                    if hasattr(element, 'get_image'):
                        try:
                            table_image = element.get_image(document)
                            if table_image:
                                image_path = self.tables_dir / f"{Path(pdf_path).stem}_table_{table_counter}.png"
                                table_image.save(image_path, "PNG")
                                table_info["image_file"] = str(image_path)
                        except Exception as e:
                            print(f"Could not save table image: {e}")
                    
                    tables.append(table_info)
                    
                except Exception as e:
                    print(f"Error processing table {table_counter}: {e}")
                    # Still add basic info
                    tables.append({
                        "id": f"table_{table_counter}",
                        "text": table_text,
                        "data": table_data,
                        "csv": table_csv,
                        "html": table_html,
                        "markdown": table_markdown,
                        "page_number": page_no,
                        "bbox": getattr(element, 'bbox', None),
                        "row_count": len(df) if df is not None else 0,
                        "col_count": len(df.columns) if df is not None else 0
                    })
        
        parsed_content["tables"] = tables
        print(f"✓ Parsed {len(tables)} tables")
    
    def _parse_images(self, document: DoclingDocument, parsed_content: Dict, pdf_path: str):
        """Parse images and figures"""
        images = []
        image_counter = 0
        
        # Parse page images
        if hasattr(document, 'pages'):
            for page_no, page in document.pages.items():
                if hasattr(page, 'image') and page.image:
                    try:
                        page_image_path = self.pages_dir / f"{Path(pdf_path).stem}_page_{page_no}.png"
                        page.image.pil_image.save(page_image_path, "PNG")
                        
                        images.append({
                            "type": "page",
                            "id": f"page_{page_no}",
                            "page_number": page_no,
                            "bbox": getattr(page, 'bbox', None),
                            "file_path": str(page_image_path),
                            "size": page.image.pil_image.size if hasattr(page.image, 'pil_image') else None
                        })
                    except Exception as e:
                        print(f"Could not save page {page_no} image: {e}")
        
        # Parse figure/picture images
        for element, level in document.iterate_items():
            if isinstance(element, PictureItem):
                image_counter += 1
                
                try:
                    # Get image
                    image = element.get_image(document)
                    if image:
                        image_path = self.figures_dir / f"{Path(pdf_path).stem}_figure_{image_counter}.png"
                        image.save(image_path, "PNG")
                        
                        # Get metadata from provenance
                        page_no = 'Unknown'
                        if hasattr(element, 'prov') and element.prov:
                            prov_item = element.prov[0]
                            page_no = getattr(prov_item, 'page_no', 'Unknown')
                        
                        caption = element.caption_text(document)
                        
                        image_info = {
                            "type": "figure",
                            "id": f"figure_{image_counter}",
                            "page_number": page_no,
                            "file_path": str(image_path),
                            "bbox": getattr(element, 'bbox', None),
                            "size": image.size,
                            "caption": caption
                        }
                        
                        # Always save a text file for each figure (with or without caption)
                        txt_path = self.figures_dir / f"{Path(pdf_path).stem}_figure_{image_counter}.txt"
                        caption_text = f"Figure {image_counter}\n"
                        caption_text += f"{'=' * 60}\n\n"
                        caption_text += f"Source: {Path(pdf_path).name}\n"
                        caption_text += f"Page: {page_no}\n"
                        caption_text += f"Size: {image.size[0]}x{image.size[1]} pixels\n\n"
                        
                        if caption:
                            caption_text += f"Caption/Description:\n{'-' * 60}\n{caption}"
                        else:
                            caption_text += f"Caption/Description:\n{'-' * 60}\n(No caption detected)"
                        
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(caption_text)
                        image_info["text_file"] = str(txt_path)
                        
                        images.append(image_info)
                        
                except Exception as e:
                    print(f"Error processing image {image_counter}: {e}")
        
        parsed_content["images"] = images
        text_count = sum(1 for img in images if img.get("text_file"))
        figure_count = sum(1 for img in images if img.get("type") == "figure")
        print(f"✓ Parsed {len(images)} images ({figure_count} figures with text files, {text_count} total text files)")
    
    # def create_comprehensive_chunks(self, parsed_content: Dict, pdf_path: str) -> List[Dict[str, Any]]:
    #     """Create chunks incorporating text, tables, and images"""
    #     chunks = []
    #     chunk_counter = 0
        
    #     # Text chunks
    #     if parsed_content.get("text_content", {}).get("full_markdown"):
    #         text_chunks = self.text_splitter.split_text(parsed_content["text_content"]["full_markdown"])
            
    #         for i, chunk in enumerate(text_chunks):
    #             chunk_obj = {
    #                 "id": f"{Path(pdf_path).stem}_text_chunk_{i}",
    #                 "type": "text",
    #                 "content": chunk,
    #                 "embedding": self.embedder.encode(chunk).tolist(),
    #                 "metadata": {
    #                     "chunk_index": i,
    #                     "content_type": "text",
    #                     "source": pdf_path,
    #                     "chunk_size": len(chunk)
    #                 }
    #             }
    #             chunks.append(chunk_obj)
        
    #     # Table chunks
    #     for table in parsed_content.get("tables", []):
    #         # Create chunk for table if it has text or CSV data
    #         if table.get("text") or table.get("csv"):
    #             # Create chunk for table text content
    #             table_text = f"Table {table['id']}"
    #             if table.get("page_number"):
    #                 table_text += f" (Page {table['page_number']})"
    #             table_text += "\\n"
                
    #             if table.get("text"):
    #                 table_text += f"\\n{table['text']}"
                    
    #             if table.get("csv"):
    #                 # Include a preview of the CSV data
    #                 csv_preview = table['csv'][:1000] if len(table['csv']) > 1000 else table['csv']
    #                 table_text += f"\\n\\nTable Data (CSV format):\\n{csv_preview}"
    #                 if len(table['csv']) > 1000:
    #                     table_text += "\\n... (truncated)"
                
    #             chunk_obj = {
    #                 "id": f"{Path(pdf_path).stem}_table_chunk_{table['id']}",
    #                 "type": "table",
    #                 "content": table_text,
    #                 "embedding": self.embedder.encode(table_text).tolist(),
    #                 "metadata": {
    #                     "content_type": "table",
    #                     "table_id": table["id"],
    #                     "page_number": table.get("page_number"),
    #                     "row_count": table.get("row_count", 0),
    #                     "col_count": table.get("col_count", 0),
    #                     "csv_file": table.get("csv_file"),
    #                     "markdown_file": table.get("markdown_file"),
    #                     "image_file": table.get("image_file"),
    #                     "source": pdf_path
    #                 }
    #             }
    #             chunks.append(chunk_obj)
        
    #     # Image chunks (for figures with text files)
    #     for image in parsed_content.get("images", []):
    #         # Only create chunks for figures (not pages) that have text files
    #         if image.get("type") == "figure" and image.get("text_file"):
    #             image_text = f"Image {image['id']}"
    #             if image.get("page_number"):
    #                 image_text += f" (Page {image['page_number']})"
                
    #             # Add caption if available
    #             if image.get("caption"):
    #                 image_text += f": {image['caption']}"
    #             else:
    #                 image_text += " (Figure from document - no caption detected)"
                
    #             chunk_obj = {
    #                 "id": f"{Path(pdf_path).stem}_image_chunk_{image['id']}",
    #                 "type": "image",
    #                 "content": image_text,
    #                 "embedding": self.embedder.encode(image_text).tolist(),
    #                 "metadata": {
    #                     "content_type": "image",
    #                     "image_id": image["id"],
    #                     "page_number": image.get("page_number"),
    #                     "image_file": image["file_path"],
    #                     "text_file": image.get("text_file"),
    #                     "image_size": image.get("size"),
    #                     "source": pdf_path
    #                 }
    #             }
    #             chunks.append(chunk_obj)
        
    #     print(f"✓ Created {len(chunks)} comprehensive chunks")
    #     return chunks
        
    def save_parsing_summary(self, parsed_content: Dict, pdf_path: str):
        """Save a summary of parsed content"""
        summary = {
            "document": Path(pdf_path).name,
            "parsing_summary": {
                "text": {
                    "elements_count": len(parsed_content.get("text_content", {}).get("elements", [])),
                    "word_count": parsed_content.get("text_content", {}).get("word_count", 0),
                    "char_count": parsed_content.get("text_content", {}).get("char_count", 0)
                },
                "tables": {
                    "count": len(parsed_content.get("tables", [])),
                    "csv_files": [t.get("csv_file") for t in parsed_content.get("tables", []) if t.get("csv_file")],
                    "markdown_files": [t.get("markdown_file") for t in parsed_content.get("tables", []) if t.get("markdown_file")]
                },
                "images": {
                    "count": len(parsed_content.get("images", [])),
                    "image_files": [img["file_path"] for img in parsed_content.get("images", [])],
                    "text_files": [img.get("text_file") for img in parsed_content.get("images", []) if img.get("text_file")]
                }
            }
        }
        
        summary_path = self.output_dir / f"{Path(pdf_path).stem}_parsing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved parsing summary to {summary_path}")
        return summary

# Usage example
def process_pdf_with_docling_advanced(
    pdf_path: str, 
    output_dir: str = "./docling_pdf_output",
    device: str = "auto",  # "auto" (recommended), "cpu", or "cuda"
    do_ocr: bool = False  # Set to True for scanned PDFs
):
    """
    Complete workflow for comprehensive PDF processing with Docling
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save parsed content
        device: "auto" (auto-detect), "cpu", or "cuda" for GPU acceleration
        do_ocr: Whether to perform OCR (False for text-based PDFs, True for scanned PDFs)
    """
    processor = DoclingAdvancedProcessor(output_dir=output_dir, device=device, do_ocr=do_ocr)
    
    # 1. Parse all content types
    parsed_content = processor.parse_comprehensive_content(pdf_path)
    if not parsed_content:
        return None
    
    # 2. Create comprehensive chunks
    # chunks = processor.create_comprehensive_chunks(parsed_content, pdf_path)
    
    # 3. Save summary
    summary = processor.save_parsing_summary(parsed_content, pdf_path)
    
    # 4. Store in vector database
#    client = chromadb.PersistentClient(path="./chroma_db")
#    collection_name = f"docling_comprehensive_{Path(pdf_path).stem}"
#    collection = client.get_or_create_collection(name=collection_name)
    
 #   valid_chunks = [c for c in chunks if c.get("embedding")]
 #   if valid_chunks:
 #       collection.upsert(
 #           documents=[c["content"] for c in valid_chunks],
 #           embeddings=[c["embedding"] for c in valid_chunks],
 #           ids=[c["id"] for c in valid_chunks],
 #           metadatas=[c["metadata"] for c in valid_chunks]
 #       )
 #       print(f"✓ Stored {len(valid_chunks)} chunks in ChromaDB")
    
    return {
        "parsed_content": parsed_content,
        # "chunks": chunks,
        "summary": summary,
        "output_directory": output_dir
    }

# Usage:
if __name__ == "__main__":
    # Example 1: Text-based PDF with auto device detection (recommended)
    result = process_pdf_with_docling_advanced(
        pdf_path="./docs/DO_NOT_KovSpec.pdf",
        device="auto",  # Auto-detect best device (CUDA if available, else CPU)
        do_ocr=False    # False for text-based PDFs
    )
    
    # Example 2: Force CPU usage
    # result = process_pdf_with_docling_advanced(
    #     pdf_path="./docs/DO_NOT_KovSpec.pdf",
    #     device="cpu",   # Force CPU usage
    #     do_ocr=False
    # )
    
    # Example 3: Scanned PDF with OCR (slower but needed for scanned docs)
    # result = process_pdf_with_docling_advanced(
    #     pdf_path="./docs/scanned_document.pdf",
    #     device="auto",  # Use best available device
    #     do_ocr=True     # Enable OCR for scanned PDFs
    # )
    
    print("\n" + "=" * 70)
    print("DOCLING Enhanced Example - Processing Complete!")
    print("=" * 70)
    print("Features:")
    print("✓ Comprehensive TEXT parsing with element analysis")
    print("✓ Advanced TABLE parsing:")
    print("  - DataFrame export and CSV saving in 'tables/' folder")
    print("  - Markdown files with formatted table content")
    print("  - Table images in 'tables/' folder")
    print("✓ Complete IMAGE parsing:")
    print("  - Page screenshots in 'pages/' folder")
    print("  - Figure images in 'figures/' folder")
    print("  - Text files for ALL figures (with captions/metadata)")
    print("✓ Structured chunking for all content types")
    print("✓ Individual chunk text files in 'chunks/' subdirectory")
    print("✓ Organized output directory structure")
    print("✓ Detailed parsing summaries")
    print(f"✓ Device support: CPU and CUDA")
    print(f"✓ OCR control: Enabled/Disabled based on document type")
    print("=" * 70)
    print("\nOutput Structure:")
    print("  docling_output/")
    print("  ├── tables/        (CSV, markdown, and table images)")
    print("  ├── pages/         (page screenshots)")
    print("  ├── figures/       (figures and captions)")
    print("  └── chunks/        (individual chunk text files)")
    print("=" * 70)
