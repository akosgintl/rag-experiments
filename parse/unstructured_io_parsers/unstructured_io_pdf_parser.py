# =================================================================
# UNSTRUCTURED.IO - COMPREHENSIVE TEXT, TABLE, AND IMAGE PARSING
# =================================================================

# Installation with full capabilities
# pip install unstructured[pdf,paddleocr] sentence-transformers chromadb
# pip install pillow pandas openpyxl

# IMAGE PARSING NOTES:
# -----------------------
# Unstructured.io saves parsed images to disk by default.
# - The 'image_output_dir_path' parameter controls where images are saved
# - By default, images go to a 'figures/' directory if not specified
# - This script configures it to save to the output directory
# - Images are then organized into 'images/' and 'tables/' subdirectories
# - Image metadata contains file paths (image_path, image_file, or image_filepath)
# - This script handles both disk-based parsing (preferred) and base64 (fallback)

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

from unstructured.partition.pdf import partition_pdf
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

class UnstructuredAdvancedProcessor:
    def __init__(
        self, 
        embedding_model="all-MiniLM-L6-v2", 
        output_dir="./unstructured_pdf_output",
        device="auto",  # "cpu", "cuda", or "auto" (auto-detect best available)
        do_ocr=True,  # Set to False for text-based PDFs to avoid OCR overhead
        ocr_languages=["hun"]  # OCR languages, e.g., ["eng"], ["hun"], ["eng", "hun"]
    ):
        """
        Initialize the UnstructuredAdvancedProcessor
        
        Args:
            embedding_model: Name of the SentenceTransformer model
            output_dir: Directory to save parsed content
            device: Device to use for processing ("cpu", "cuda", or "auto")
            do_ocr: Whether to perform OCR (False for text-based PDFs, True for scanned PDFs)
            ocr_languages: List of OCR language codes (e.g., ["eng"], ["hun"])
        """
        # Auto-detect or validate device availability
        self.device = get_available_device(device)
        self.do_ocr = do_ocr
        self.ocr_languages = ocr_languages if ocr_languages else ["eng"]
        
        # Initialize embedder with device support
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        
        print(f"✓ Initialized with device: {self.device}, OCR: {self.do_ocr}")
    
    def parse_comprehensive_content(self, pdf_path: str, strategy: str = "hi_res") -> Dict[str, Any]:
        """
        Parse TEXT, TABLES, and IMAGES comprehensively from PDF using Unstructured
        """
        try:
            print(f"Processing {pdf_path} with Unstructured.io using '{strategy}' strategy...")
            
            # Build partition_pdf parameters based on OCR setting
            partition_params = {
                "filename": pdf_path,
                "strategy": strategy,  # "hi_res" for best quality, "auto" for speed
                "infer_table_structure": True,  # Critical for table parsing
                "include_page_breaks": True,
                "extract_images_in_pdf": True,
                "include_metadata": True,
                
                # IMAGE AND TABLE PARSING - KEY PARAMETERS
                "extract_image_block_types": ["Image", "Table"],  # Parse images from Image and Table blocks
                "image_output_dir_path": str(self.output_dir),  # Save images to output directory
                
                # Additional quality settings for hi_res
                "hi_res_model_name": "yolox" if strategy == "hi_res" else None,
                "pdf_infer_table_structure": True,
                
                # Coordinate settings for precise location
                "include_orig_elements": True
            }
            
            # Add OCR settings only if OCR is enabled
            if self.do_ocr:
                partition_params["languages"] = self.ocr_languages
                print(f"  OCR enabled with languages: {self.ocr_languages}")
            else:
                print(f"  OCR disabled (text-based PDF mode)")
            
            # Parse all elements with comprehensive settings
            elements = partition_pdf(**partition_params)
            
            if not elements:
                print("No elements parsed!")
                return None
            
            parsed_content = {
                "text_content": [],
                "tables": [],
                "images": [],
                "metadata": {
                    "source": pdf_path,
                    "strategy": strategy,
                    "total_elements": len(elements)
                }
            }
            
            # Process all elements
            self._process_elements(elements, parsed_content, pdf_path)
            
            # Clean up any leftover 'figures' directory created by unstructured
            # (images should already be organized into images/ and tables/ directories)
            figures_dir = self.output_dir / "figures"
            if figures_dir.exists():
                try:
                    # Check if there are any remaining images that weren't processed
                    remaining_images = list(figures_dir.glob("*"))
                    if remaining_images:
                        print(f"  Note: Found {len(remaining_images)} unprocessed images in figures directory")
                        print(f"  These may be images not associated with Table or Image elements")
                    # Optionally remove the figures directory
                    # shutil.rmtree(figures_dir)
                    # print(f"  Cleaned up figures directory")
                except Exception as e:
                    print(f"  Warning: Could not clean up figures directory: {e}")
            
            return parsed_content
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return None
    
    def _process_elements(self, elements: List, parsed_content: Dict, pdf_path: str):
        """Process all elements and categorize them"""
        text_elements = []
        tables = []
        images = []
        
        table_counter = 0
        image_counter = 0
        
        # Debug: Collect all element types found
        element_types_found = set()
        
        for i, element in enumerate(elements):
            element_type = type(element).__name__
            element_types_found.add(element_type)
            
            # Get common metadata
            metadata = getattr(element, 'metadata', None)
            page_number = getattr(metadata, 'page_number', 'Unknown') if metadata else 'Unknown'
            coordinates = getattr(metadata, 'coordinates', None) if metadata else None
            
            # TEXT PARSING
            # if element_type in ["Title", "NarrativeText", "Header", "Footer", "ListItem"]:

            if element_type in ["Title", "NarrativeText", "Header", "Footer", "ListItem", 
                    "FigureCaption", "PageNumber", "UncategorizedText", 
                    "Address", "EmailAddress", "CodeSnippet", "Formula"]:

                text_info = {
                    "text": element.text,
                    "type": element_type,
                    "page_number": page_number,
                    "coordinates": coordinates,
                    "element_id": getattr(element, 'id', f"text_{i}")
                }
                text_elements.append(text_info)

            # TABLE PARSING
            elif element_type == "Table":
                table_counter += 1
                print(f"  Processing Table element #{table_counter} on page {page_number}")
                table_info = self._parse_table_content(element, table_counter, pdf_path, page_number, coordinates)
                if table_info:
                    tables.append(table_info)
            
            # IMAGE PARSING
            elif element_type == "Image":
                image_counter += 1
                print(f"  Processing Image element #{image_counter} on page {page_number}")
                image_info = self._parse_image_content(element, image_counter, pdf_path, page_number, coordinates)
                if image_info:
                    images.append(image_info)
        
        # Store processed content
        parsed_content["text_content"] = text_elements
        parsed_content["tables"] = tables
        parsed_content["images"] = images
        
        # Save text content to file
        self._save_text_content(text_elements, pdf_path)
        
        # Debug output
        print(f"\n  DEBUG: Element types found in document: {sorted(element_types_found)}")
        print(f"✓ Processed {len(text_elements)} text elements")
        print(f"✓ Processed {len(tables)} tables")
        print(f"✓ Processed {len(images)} images")
    
    def _save_text_content(self, text_elements: List[Dict], pdf_path: str):
        """Save text content to a markdown file in the text folder"""
        try:
            if not text_elements:
                print("No text content to save")
                return
            
            # Create markdown content organized by pages
            markdown_content = []
            markdown_content.append(f"# Text Parsing from {Path(pdf_path).name}\n\n")
            
            # Group text elements by page
            pages_dict = {}
            for elem in text_elements:
                page_num = elem.get('page_number', 'Unknown')
                if page_num not in pages_dict:
                    pages_dict[page_num] = []
                pages_dict[page_num].append(elem)
            
            # Write content organized by page
            for page_num in sorted(pages_dict.keys(), key=lambda x: (x == 'Unknown', x)):
                markdown_content.append(f"## Page {page_num}\n\n")
                
                for elem in pages_dict[page_num]:
                    elem_type = elem.get('type', 'Text')
                    text = elem.get('text', '')
                    
                    # Format based on element type
                    if elem_type == "Title":
                        markdown_content.append(f"### {text}\n\n")
                    elif elem_type == "Header":
                        markdown_content.append(f"**{text}**\n\n")
                    elif elem_type == "ListItem":
                        markdown_content.append(f"- {text}\n")
                    else:
                        markdown_content.append(f"{text}\n\n")
            
            # Save to file
            text_file_path = self.output_dir / "text" / f"{Path(pdf_path).stem}_full_text.md"
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(''.join(markdown_content))
            
            print(f"✓ Saved text content to {text_file_path}")
            
            # Also save a plain text version
            plain_text_content = '\n'.join([elem.get('text', '') for elem in text_elements])
            plain_text_path = self.output_dir / "text" / f"{Path(pdf_path).stem}_plain_text.txt"
            with open(plain_text_path, 'w', encoding='utf-8') as f:
                f.write(plain_text_content)
            
            print(f"✓ Saved plain text to {plain_text_path}")
            
        except Exception as e:
            print(f"Error saving text content: {e}")
    
    def _parse_table_content(self, table_element, table_counter: int, pdf_path: str, page_number, coordinates) -> Dict:
        """Parse comprehensive table information"""
        try:
            table_info = {
                "id": f"table_{table_counter}",
                "text": table_element.text,
                "page_number": page_number,
                "coordinates": coordinates,
                "element_id": getattr(table_element, 'id', f"table_{table_counter}")
            }
            
            # Get HTML structure if available
            metadata = getattr(table_element, 'metadata', None)
            if metadata and hasattr(metadata, 'text_as_html'):
                html_content = metadata.text_as_html
                table_info["html"] = html_content
                
                # Save HTML table
                html_path = self.output_dir / "tables" / f"{Path(pdf_path).stem}_table_{table_counter}.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(f"<html><body>{html_content}</body></html>")
                table_info["html_file"] = str(html_path)
                
                # Convert HTML to DataFrame and CSV if possible
                try:
                    df = pd.read_html(io.StringIO(html_content))[0]
                    csv_content = df.to_csv(index=False)
                    
                    # Save as CSV
                    csv_path = self.output_dir / "tables" / f"{Path(pdf_path).stem}_table_{table_counter}.csv"
                    df.to_csv(csv_path, index=False)
                    
                    table_info.update({
                        "csv": csv_content,
                        "csv_file": str(csv_path),
                        "row_count": len(df),
                        "col_count": len(df.columns),
                        "columns": df.columns.tolist()
                    })
                except Exception as e:
                    print(f"Could not convert table {table_counter} to DataFrame: {e}")
            
            # Parse image if available (from disk or base64)
            # Debug: Print available metadata attributes
            if metadata:
                available_attrs = [attr for attr in dir(metadata) if not attr.startswith('_')]
                if table_counter == 1:  # Only print for first table to avoid spam
                    print(f"  DEBUG: Available metadata attributes for Table: {available_attrs}")
                
                # Method 1: Check if image was saved to disk (newer unstructured versions)
                image_saved = False
                for attr_name in ['image_path', 'image_file', 'image_filepath']:
                    if hasattr(metadata, attr_name):
                        source_image_path = getattr(metadata, attr_name)
                        if source_image_path and Path(source_image_path).exists():
                            try:
                                # Copy/move the image from figures to tables directory
                                source_path = Path(source_image_path)
                                target_path = self.output_dir / "tables" / f"{Path(pdf_path).stem}_table_{table_counter}{source_path.suffix}"
                                
                                # Open and save to ensure it's in our preferred format
                                image = Image.open(source_path)
                                image.save(target_path, "PNG")
                                
                                table_info.update({
                                    "image_file": str(target_path),
                                    "image_size": image.size,
                                    "original_path": str(source_path)
                                })
                                print(f"  ✓ Saved table {table_counter} image from {source_path.name}")
                                image_saved = True
                                break
                            except Exception as e:
                                print(f"  Could not copy table {table_counter} image: {e}")
                
                # Method 2: Try base64 image data (fallback for older versions or different configs)
                if not image_saved:
                    image_base64 = None
                    if hasattr(metadata, 'image_base64'):
                        image_base64 = metadata.image_base64
                    elif hasattr(metadata, 'orig_elements'):
                        # Try to get image from original elements
                        orig_elements = metadata.orig_elements
                        if orig_elements and len(orig_elements) > 0:
                            for orig_elem in orig_elements:
                                if hasattr(orig_elem, 'image_base64'):
                                    image_base64 = orig_elem.image_base64
                                    break
                    
                    if image_base64 and isinstance(image_base64, (str, bytes)) and len(str(image_base64).strip()) > 0:
                        try:
                            image_data = base64.b64decode(image_base64)
                            if len(image_data) > 0:  # Ensure we have actual data
                                image = Image.open(io.BytesIO(image_data))
                                
                                # Save table image
                                image_path = self.output_dir / "tables" / f"{Path(pdf_path).stem}_table_{table_counter}_image.png"
                                image.save(image_path, "PNG")
                                
                                table_info.update({
                                    "image_file": str(image_path),
                                    "image_size": image.size
                                })
                                print(f"  ✓ Saved table {table_counter} image from base64")
                        except Exception as e:
                            print(f"  Could not save table {table_counter} image from base64: {e}")
            
            return table_info
            
        except Exception as e:
            print(f"Error processing table {table_counter}: {e}")
            return {
                "id": f"table_{table_counter}",
                "text": getattr(table_element, 'text', ''),
                "error": str(e)
            }
    
    def _parse_image_content(self, image_element, image_counter: int, pdf_path: str, page_number, coordinates) -> Dict:
        """Parse comprehensive image information"""
        try:
            image_info = {
                "id": f"image_{image_counter}",
                "text": getattr(image_element, 'text', ''),  # Image caption/alt text
                "page_number": page_number,
                "coordinates": coordinates,
                "element_id": getattr(image_element, 'id', f"image_{image_counter}")
            }
            
            # Parse image if available (from disk or base64)
            metadata = getattr(image_element, 'metadata', None)
            if metadata:
                available_attrs = [attr for attr in dir(metadata) if not attr.startswith('_')]
                if image_counter == 1:  # Only print for first image to avoid spam
                    print(f"  DEBUG: Available metadata attributes for Image: {available_attrs}")
                
                # Method 1: Check if image was saved to disk (newer unstructured versions)
                image_saved = False
                for attr_name in ['image_path', 'image_file', 'image_filepath']:
                    if hasattr(metadata, attr_name):
                        source_image_path = getattr(metadata, attr_name)
                        if source_image_path and Path(source_image_path).exists():
                            try:
                                # Copy/move the image from figures to images directory
                                source_path = Path(source_image_path)
                                target_path = self.output_dir / "images" / f"{Path(pdf_path).stem}_image_{image_counter}{source_path.suffix}"
                                
                                # Open and save to ensure it's in our preferred format
                                image = Image.open(source_path)
                                image.save(target_path, "PNG")
                                
                                image_info.update({
                                    "image_file": str(target_path),
                                    "image_size": image.size,
                                    "format": image.format,
                                    "original_path": str(source_path)
                                })
                                print(f"  ✓ Saved image {image_counter} from {source_path.name}")
                                image_saved = True
                                break
                            except Exception as e:
                                print(f"  Could not copy image {image_counter}: {e}")
                
                # Method 2: Try base64 image data (fallback for older versions or different configs)
                if not image_saved:
                    image_base64 = None
                    if hasattr(metadata, 'image_base64'):
                        image_base64 = metadata.image_base64
                    elif hasattr(metadata, 'orig_elements'):
                        # Try to get image from original elements
                        orig_elements = metadata.orig_elements
                        if orig_elements and len(orig_elements) > 0:
                            for orig_elem in orig_elements:
                                if hasattr(orig_elem, 'image_base64'):
                                    image_base64 = orig_elem.image_base64
                                    break
                    
                    if image_base64 and isinstance(image_base64, (str, bytes)) and len(str(image_base64).strip()) > 0:
                        try:
                            image_data = base64.b64decode(image_base64)
                            if len(image_data) > 0:  # Ensure we have actual data
                                image = Image.open(io.BytesIO(image_data))
                                
                                # Save image
                                image_path = self.output_dir / "images" / f"{Path(pdf_path).stem}_image_{image_counter}.png"
                                image.save(image_path, "PNG")
                                
                                image_info.update({
                                    "image_file": str(image_path),
                                    "image_size": image.size,
                                    "format": image.format
                                })
                                print(f"  ✓ Saved image {image_counter} from base64")
                        except Exception as e:
                            print(f"  Could not save image {image_counter} from base64: {e}")
            
            # Get additional metadata
            if metadata:
                image_info.update({
                    "filename": getattr(metadata, 'filename', None),
                    "filetype": getattr(metadata, 'filetype', None),
                    "languages": getattr(metadata, 'languages', [])
                })
            
            return image_info
            
        except Exception as e:
            print(f"Error processing image {image_counter}: {e}")
            return {
                "id": f"image_{image_counter}",
                "error": str(e)
            }
    
    def create_comprehensive_chunks(self, parsed_content: Dict, pdf_path: str, max_characters: int = 1000) -> List[Dict[str, Any]]:
        """Create semantic chunks incorporating all content types"""
        chunks = []
        
        # Create elements list for chunking
        all_elements = []
        
        # Add text elements back to element-like objects for chunking
        for text_elem in parsed_content.get("text_content", []):
            # Create mock element for chunking
            mock_element = type('MockElement', (), {
                'text': text_elem['text'],
                'metadata': type('MockMetadata', (), {
                    'page_number': text_elem['page_number'],
                    'element_type': text_elem['type'],
                    'coordinates': text_elem.get('coordinates')
                })()
            })()
            all_elements.append(mock_element)
        
        # Use Unstructured's semantic chunking
        if all_elements:
            try:
                chunked_elements = chunk_by_title(
                    all_elements,
                    max_characters=max_characters,
                    combine_text_under_n_chars=100,
                    new_after_n_chars=int(max_characters * 0.8)
                )
                
                # Convert to standardized chunks
                for i, chunk in enumerate(chunked_elements):
                    chunk_text = chunk.text
                    embedding = self.embedder.encode(chunk_text).tolist()
                    
                    chunk_obj = {
                        "id": f"{Path(pdf_path).stem}_text_chunk_{i}",
                        "type": "text",
                        "content": chunk_text,
                        "embedding": embedding,
                        "metadata": {
                            "chunk_index": i,
                            "content_type": "text",
                            "source": pdf_path,
                            "chunk_size": len(chunk_text),
                            "page_number": getattr(chunk.metadata, 'page_number', None) if hasattr(chunk, 'metadata') else None
                        }
                    }
                    chunks.append(chunk_obj)
                    
            except Exception as e:
                print(f"Semantic chunking failed, using basic chunking: {e}")
                # Fallback: basic text chunking
                full_text = " ".join([elem['text'] for elem in parsed_content.get("text_content", [])])
                text_chunks = [full_text[i:i+max_characters] for i in range(0, len(full_text), max_characters)]
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk_obj = {
                        "id": f"{Path(pdf_path).stem}_text_chunk_{i}",
                        "type": "text",
                        "content": chunk_text,
                        "embedding": self.embedder.encode(chunk_text).tolist(),
                        "metadata": {
                            "chunk_index": i,
                            "content_type": "text",
                            "source": pdf_path,
                            "chunk_size": len(chunk_text)
                        }
                    }
                    chunks.append(chunk_obj)
        
        # Add table chunks
        for table in parsed_content.get("tables", []):
            if table.get("text"):
                table_text = f"Table {table['id']} (Page {table.get('page_number', 'Unknown')})\\n"
                table_text += table["text"]
                if table.get("html"):
                    table_text += f"\\n\\nStructured Data: {table['html'][:300]}..."
                
                chunk_obj = {
                    "id": f"{Path(pdf_path).stem}_table_chunk_{table['id']}",
                    "type": "table",
                    "content": table_text,
                    "embedding": self.embedder.encode(table_text).tolist(),
                    "metadata": {
                        "content_type": "table",
                        "table_id": table["id"],
                        "page_number": table.get("page_number"),
                        "html_file": table.get("html_file"),
                        "csv_file": table.get("csv_file"),
                        "image_file": table.get("image_file"),
                        "row_count": table.get("row_count", 0),
                        "col_count": table.get("col_count", 0),
                        "source": pdf_path
                    }
                }
                chunks.append(chunk_obj)
        
        # Add image chunks
        for image in parsed_content.get("images", []):
            if image.get("text"):  # Only if there's caption/alt text
                image_text = f"Image {image['id']} (Page {image.get('page_number', 'Unknown')})\\n"
                image_text += image["text"]
                
                chunk_obj = {
                    "id": f"{Path(pdf_path).stem}_image_chunk_{image['id']}",
                    "type": "image",
                    "content": image_text,
                    "embedding": self.embedder.encode(image_text).tolist(),
                    "metadata": {
                        "content_type": "image",
                        "image_id": image["id"],
                        "page_number": image.get("page_number"),
                        "image_file": image.get("image_file"),
                        "image_size": image.get("image_size"),
                        "source": pdf_path
                    }
                }
                chunks.append(chunk_obj)
        
        print(f"✓ Created {len(chunks)} comprehensive chunks")
        return chunks
    
    def save_parsing_summary(self, parsed_content: Dict, pdf_path: str):
        """Save detailed parsing summary"""
        summary = {
            "document": Path(pdf_path).name,
            "parsing_summary": {
                "text": {
                    "elements_count": len(parsed_content.get("text_content", [])),
                    "types": list(set([elem.get("type", "Unknown") for elem in parsed_content.get("text_content", [])]))
                },
                "tables": {
                    "count": len(parsed_content.get("tables", [])),
                    "with_html": len([t for t in parsed_content.get("tables", []) if t.get("html")]),
                    "with_csv": len([t for t in parsed_content.get("tables", []) if t.get("csv_file")]),
                    "with_images": len([t for t in parsed_content.get("tables", []) if t.get("image_file")]),
                    "files": {
                        "csv": [t.get("csv_file") for t in parsed_content.get("tables", []) if t.get("csv_file")],
                        "html": [t.get("html_file") for t in parsed_content.get("tables", []) if t.get("html_file")],
                        "images": [t.get("image_file") for t in parsed_content.get("tables", []) if t.get("image_file")]
                    }
                },
                "images": {
                    "count": len(parsed_content.get("images", [])),
                    "with_files": len([img for img in parsed_content.get("images", []) if img.get("image_file")]),
                    "files": [img.get("image_file") for img in parsed_content.get("images", []) if img.get("image_file")]
                }
            },
            "metadata": parsed_content.get("metadata", {})
        }
        
        summary_path = self.output_dir / f"{Path(pdf_path).stem}_unstructured_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved parsing summary to {summary_path}")
        return summary

# Usage function
def process_pdf_with_unstructured_advanced(
    pdf_path: str, 
    strategy: str = "hi_res", 
    output_dir: str = "./unstructured_pdf_output",
    device: str = "auto",  # "auto" (recommended), "cpu", or "cuda"
    do_ocr: bool = True,  # Set to True for scanned PDFs
    ocr_languages: list = ["hun"]  # OCR languages, e.g., ["eng"], ["hun"], ["eng", "hun"]
):
    """
    Complete workflow for comprehensive PDF processing with Unstructured.io
    
    Args:
        pdf_path: Path to the PDF file
        strategy: Parsing strategy ("fast", "auto", or "hi_res")
        output_dir: Directory to save parsed content
        device: "auto" (auto-detect), "cpu", or "cuda" for GPU acceleration
        do_ocr: Whether to perform OCR (False for text-based PDFs, True for scanned PDFs)
        ocr_languages: List of OCR language codes (e.g., ["eng"], ["hun"])
    """
    processor = UnstructuredAdvancedProcessor(
        output_dir=output_dir, 
        device=device, 
        do_ocr=do_ocr,
        ocr_languages=ocr_languages
    )
    
    # 1. Parse all content types
    parsed_content = processor.parse_comprehensive_content(pdf_path, strategy)
    if not parsed_content:
        return None
    
    # 2. Create comprehensive chunks
    # chunks = processor.create_comprehensive_chunks(parsed_content, pdf_path)
    
    # 3. Save summary
    summary = processor.save_parsing_summary(parsed_content, pdf_path)
    
    # 4. Store in vector database
#    client = chromadb.PersistentClient(path="./chroma_db")
#    collection_name = f"unstructured_comprehensive_{Path(pdf_path).stem}"
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
# Example 1: Text-based PDF with auto device detection (recommended)
result = process_pdf_with_unstructured_advanced(
    pdf_path="./docs/DO_NOT_KovSpec.pdf",
    strategy="hi_res",  # Best quality
    device="auto",      # Auto-detect best device (CUDA if available, else CPU)
    do_ocr=True        # False for text-based PDFs (faster, no tesseract needed)
)

# Example 2: Scanned PDF with OCR (requires tesseract installation)
# result = process_pdf_with_unstructured_advanced(
#     pdf_path="./docs/DO_NOT_KovSpec.pdf",
#     strategy="hi_res",
#     device="auto",
#     do_ocr=True,              # Enable OCR for scanned PDFs
#     ocr_languages=["eng"]     # Language(s) for OCR
# )

# Example 3: Hungarian document with OCR
# result = process_pdf_with_unstructured_advanced(
#     pdf_path="./docs/DO_NOT_KovSpec.pdf",
#     strategy="hi_res",
#     device="auto",
#     do_ocr=True,
#     ocr_languages=["hun"]     # Hungarian OCR
# )

# Example 4: Multi-language OCR
# result = process_pdf_with_unstructured_advanced(
#     pdf_path="./docs/DO_NOT_KovSpec.pdf",
#     strategy="hi_res",
#     device="auto",
#     do_ocr=True,
#     ocr_languages=["eng", "hun"]  # English + Hungarian
# )

# Example 5: Fast processing with CPU only
# result = process_pdf_with_unstructured_advanced(
#     pdf_path="./docs/DO_NOT_KovSpec.pdf",
#     strategy="fast",
#     device="cpu",      # Force CPU usage
#     do_ocr=False
# )

print("\nUNSTRUCTURED.IO Enhanced Example Created")
print("=" * 70)
print("Features:")
print("✓ Comprehensive TEXT parsing with element classification")
print("✓ Advanced TABLE parsing with HTML structure and CSV conversion")
print("✓ Complete IMAGE parsing with base64 decoding and file saving")
print("✓ Semantic chunking using Unstructured's built-in chunking")
print("✓ Organized file output with subdirectories")
print("✓ Multiple processing strategies (fast/auto/hi_res)")
print("✓ Device support: CPU and CUDA (GPU acceleration)")
print("✓ OCR control: Enable/Disable based on document type")
print("✓ Multi-language OCR support (requires tesseract installation)")
print("✓ Rich metadata preservation and detailed summaries")
print("=" * 70)
