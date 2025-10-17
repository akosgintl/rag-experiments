# =================================================================
# UNSTRUCTURED.IO - COMPREHENSIVE TEXT, TABLE, AND IMAGE EXTRACTION
# =================================================================

# Installation with full capabilities
# pip install unstructured[pdf,paddleocr] sentence-transformers chromadb
# pip install pillow pandas openpyxl

import os
import json
import base64
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
        output_dir="./unstructured_output",
        device="auto",  # "cpu", "cuda", or "auto" (auto-detect best available)
        do_ocr=False,  # Set to False for text-based PDFs to avoid OCR overhead
        ocr_languages=None  # OCR languages, e.g., ["eng"], ["hun"], ["eng", "hun"]
    ):
        """
        Initialize the UnstructuredAdvancedProcessor
        
        Args:
            embedding_model: Name of the SentenceTransformer model
            output_dir: Directory to save extracted content
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
    
    def extract_comprehensive_content(self, pdf_path: str, strategy: str = "hi_res") -> Dict[str, Any]:
        """
        Extract TEXT, TABLES, and IMAGES comprehensively from PDF using Unstructured
        """
        try:
            print(f"Processing {pdf_path} with Unstructured.io using '{strategy}' strategy...")
            
            # Build partition_pdf parameters based on OCR setting
            partition_params = {
                "filename": pdf_path,
                "strategy": strategy,  # "hi_res" for best quality, "auto" for speed
                "infer_table_structure": True,  # Critical for table extraction
                "include_page_breaks": True,
                "extract_images_in_pdf": True,
                "include_metadata": True,
                
                # IMAGE AND TABLE EXTRACTION - KEY PARAMETERS
                "extract_image_block_types": ["Image", "Table"],  # Extract base64 images
                
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
            
            # Extract all elements with comprehensive settings
            elements = partition_pdf(**partition_params)
            
            if not elements:
                print("No elements extracted!")
                return None
            
            extracted_content = {
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
            self._process_elements(elements, extracted_content, pdf_path)
            
            return extracted_content
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return None
    
    def _process_elements(self, elements: List, extracted_content: Dict, pdf_path: str):
        """Process all elements and categorize them"""
        text_elements = []
        tables = []
        images = []
        
        table_counter = 0
        image_counter = 0
        
        for i, element in enumerate(elements):
            element_type = type(element).__name__
            
            # Get common metadata
            metadata = getattr(element, 'metadata', None)
            page_number = getattr(metadata, 'page_number', 'Unknown') if metadata else 'Unknown'
            coordinates = getattr(metadata, 'coordinates', None) if metadata else None
            
            # TEXT EXTRACTION
            if element_type in ["Title", "NarrativeText", "Header", "Footer", "ListItem"]:
                text_info = {
                    "text": element.text,
                    "type": element_type,
                    "page_number": page_number,
                    "coordinates": coordinates,
                    "element_id": getattr(element, 'id', f"text_{i}")
                }
                text_elements.append(text_info)
            
            # TABLE EXTRACTION
            elif element_type == "Table":
                table_counter += 1
                table_info = self._extract_table_content(element, table_counter, pdf_path, page_number, coordinates)
                if table_info:
                    tables.append(table_info)
            
            # IMAGE EXTRACTION
            elif element_type == "Image":
                image_counter += 1
                image_info = self._extract_image_content(element, image_counter, pdf_path, page_number, coordinates)
                if image_info:
                    images.append(image_info)
        
        # Store processed content
        extracted_content["text_content"] = text_elements
        extracted_content["tables"] = tables
        extracted_content["images"] = images
        
        print(f"✓ Processed {len(text_elements)} text elements")
        print(f"✓ Processed {len(tables)} tables")
        print(f"✓ Processed {len(images)} images")
    
    def _extract_table_content(self, table_element, table_counter: int, pdf_path: str, page_number, coordinates) -> Dict:
        """Extract comprehensive table information"""
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
            
            # Extract base64 image if available
            if metadata and hasattr(metadata, 'image_base64'):
                try:
                    image_data = base64.b64decode(metadata.image_base64)
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Save table image
                    image_path = self.output_dir / "tables" / f"{Path(pdf_path).stem}_table_{table_counter}_image.png"
                    image.save(image_path, "PNG")
                    
                    table_info.update({
                        "image_file": str(image_path),
                        "image_size": image.size
                    })
                except Exception as e:
                    print(f"Could not save table {table_counter} image: {e}")
            
            return table_info
            
        except Exception as e:
            print(f"Error processing table {table_counter}: {e}")
            return {
                "id": f"table_{table_counter}",
                "text": getattr(table_element, 'text', ''),
                "error": str(e)
            }
    
    def _extract_image_content(self, image_element, image_counter: int, pdf_path: str, page_number, coordinates) -> Dict:
        """Extract comprehensive image information"""
        try:
            image_info = {
                "id": f"image_{image_counter}",
                "text": getattr(image_element, 'text', ''),  # Image caption/alt text
                "page_number": page_number,
                "coordinates": coordinates,
                "element_id": getattr(image_element, 'id', f"image_{image_counter}")
            }
            
            # Extract base64 image if available
            metadata = getattr(image_element, 'metadata', None)
            if metadata and hasattr(metadata, 'image_base64'):
                try:
                    image_data = base64.b64decode(metadata.image_base64)
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Save image
                    image_path = self.output_dir / "images" / f"{Path(pdf_path).stem}_image_{image_counter}.png"
                    image.save(image_path, "PNG")
                    
                    image_info.update({
                        "image_file": str(image_path),
                        "image_size": image.size,
                        "format": image.format
                    })
                    
                except Exception as e:
                    print(f"Could not save image {image_counter}: {e}")
            
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
    
    def create_comprehensive_chunks(self, extracted_content: Dict, pdf_path: str, max_characters: int = 1000) -> List[Dict[str, Any]]:
        """Create semantic chunks incorporating all content types"""
        chunks = []
        
        # Create elements list for chunking
        all_elements = []
        
        # Add text elements back to element-like objects for chunking
        for text_elem in extracted_content.get("text_content", []):
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
                full_text = " ".join([elem['text'] for elem in extracted_content.get("text_content", [])])
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
        for table in extracted_content.get("tables", []):
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
        for image in extracted_content.get("images", []):
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
    
    def save_extraction_summary(self, extracted_content: Dict, pdf_path: str):
        """Save detailed extraction summary"""
        summary = {
            "document": Path(pdf_path).name,
            "extraction_summary": {
                "text": {
                    "elements_count": len(extracted_content.get("text_content", [])),
                    "types": list(set([elem.get("type", "Unknown") for elem in extracted_content.get("text_content", [])]))
                },
                "tables": {
                    "count": len(extracted_content.get("tables", [])),
                    "with_html": len([t for t in extracted_content.get("tables", []) if t.get("html")]),
                    "with_csv": len([t for t in extracted_content.get("tables", []) if t.get("csv_file")]),
                    "with_images": len([t for t in extracted_content.get("tables", []) if t.get("image_file")]),
                    "files": {
                        "csv": [t.get("csv_file") for t in extracted_content.get("tables", []) if t.get("csv_file")],
                        "html": [t.get("html_file") for t in extracted_content.get("tables", []) if t.get("html_file")],
                        "images": [t.get("image_file") for t in extracted_content.get("tables", []) if t.get("image_file")]
                    }
                },
                "images": {
                    "count": len(extracted_content.get("images", [])),
                    "with_files": len([img for img in extracted_content.get("images", []) if img.get("image_file")]),
                    "files": [img.get("image_file") for img in extracted_content.get("images", []) if img.get("image_file")]
                }
            },
            "metadata": extracted_content.get("metadata", {})
        }
        
        summary_path = self.output_dir / f"{Path(pdf_path).stem}_unstructured_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved extraction summary to {summary_path}")
        return summary

# Usage function
def process_pdf_with_unstructured_advanced(
    pdf_path: str, 
    strategy: str = "hi_res", 
    output_dir: str = "./unstructured_output",
    device: str = "auto",  # "auto" (recommended), "cpu", or "cuda"
    do_ocr: bool = False,  # Set to True for scanned PDFs
    ocr_languages: list = None  # OCR languages, e.g., ["eng"], ["hun"], ["eng", "hun"]
):
    """
    Complete workflow for comprehensive PDF processing with Unstructured.io
    
    Args:
        pdf_path: Path to the PDF file
        strategy: Extraction strategy ("fast", "auto", or "hi_res")
        output_dir: Directory to save extracted content
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
    
    # 1. Extract all content types
    extracted_content = processor.extract_comprehensive_content(pdf_path, strategy)
    if not extracted_content:
        return None
    
    # 2. Create comprehensive chunks
    # chunks = processor.create_comprehensive_chunks(extracted_content, pdf_path)
    
    # 3. Save summary
    summary = processor.save_extraction_summary(extracted_content, pdf_path)
    
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
        "extracted_content": extracted_content,
        # "chunks": chunks,
        "summary": summary,
        "output_directory": output_dir
    }

# Usage examples:
# Example 1: Text-based PDF with auto device detection (recommended)
result = process_pdf_with_unstructured_advanced(
    pdf_path="docs\\DO_NOT_KovSpec.pdf",
    strategy="hi_res",  # Best quality
    device="auto",      # Auto-detect best device (CUDA if available, else CPU)
    do_ocr=False        # False for text-based PDFs (faster, no tesseract needed)
)

# Example 2: Scanned PDF with OCR (requires tesseract installation)
# result = process_pdf_with_unstructured_advanced(
#     pdf_path="docs\\scanned_document.pdf",
#     strategy="hi_res",
#     device="auto",
#     do_ocr=True,              # Enable OCR for scanned PDFs
#     ocr_languages=["eng"]     # Language(s) for OCR
# )

# Example 3: Hungarian document with OCR
# result = process_pdf_with_unstructured_advanced(
#     pdf_path="docs\\hungarian_scanned.pdf",
#     strategy="hi_res",
#     device="auto",
#     do_ocr=True,
#     ocr_languages=["hun"]     # Hungarian OCR
# )

# Example 4: Multi-language OCR
# result = process_pdf_with_unstructured_advanced(
#     pdf_path="docs\\multilang_document.pdf",
#     strategy="hi_res",
#     device="auto",
#     do_ocr=True,
#     ocr_languages=["eng", "hun"]  # English + Hungarian
# )

# Example 5: Fast processing with CPU only
# result = process_pdf_with_unstructured_advanced(
#     pdf_path="docs\\DO_NOT_KovSpec.pdf",
#     strategy="fast",
#     device="cpu",      # Force CPU usage
#     do_ocr=False
# )

print("\nUNSTRUCTURED.IO Enhanced Example Created")
print("=" * 70)
print("Features:")
print("✓ Comprehensive TEXT extraction with element classification")
print("✓ Advanced TABLE extraction with HTML structure and CSV conversion")
print("✓ Complete IMAGE extraction with base64 decoding and file saving")
print("✓ Semantic chunking using Unstructured's built-in chunking")
print("✓ Organized file output with subdirectories")
print("✓ Multiple processing strategies (fast/auto/hi_res)")
print("✓ Device support: CPU and CUDA (GPU acceleration)")
print("✓ OCR control: Enable/Disable based on document type")
print("✓ Multi-language OCR support (requires tesseract installation)")
print("✓ Rich metadata preservation and detailed summaries")
print("=" * 70)
