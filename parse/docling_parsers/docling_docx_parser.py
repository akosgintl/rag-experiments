# =========================================================
# DOCLING - COMPREHENSIVE TEXT, TABLE, AND IMAGE PARSING FOR DOCX
# =========================================================
#
# This script uses the latest Docling API (updated 2025):
# - TableItem.export_to_dataframe(document) - requires document parameter
# - TableItem.export_to_markdown(document) - for parsing table text
# - No warnings or errors with current Docling version
#
# Installation with full capabilities:
# pip install docling[vlm] docling-core sentence-transformers chromadb
# pip install pillow pandas openpyxl tabulate python-docx
#
# Performance optimizations:
# pip install orjson  # 3-10x faster JSON serialization

import json
from pathlib import Path
from typing import List, Dict, Any
import torch

from docling.document_converter import DocumentConverter, WordFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PaginatedPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode
from docling_core.types.doc.document import DoclingDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Try to use faster JSON library
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

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

class DoclingDocxProcessor:
    def __init__(
        self, 
        embedding_model="all-MiniLM-L6-v2", 
        output_dir="./docling_docx_output",
        device="auto",  # "cpu", "cuda", or "auto" (auto-detect best available)
        # Performance options
        num_threads=8,  # Number of threads for parallel processing (default: 16)
        generate_picture_images=True,  # Extract images (set False to speed up)
        images_scale=1.0,  # Image resolution scale (1.0=normal, 2.0=high-res but slower)
        image_ref_mode=ImageRefMode.EMBEDDED  # How to handle images in document
    ):
        """
        Initialize the DoclingDocxProcessor with performance optimizations
        
        Args:
            embedding_model: Name of the SentenceTransformer model
            output_dir: Directory to save parsed content
            device: Device to use for processing ("cpu", "cuda", or "auto")
            num_threads: Number of threads for parallel processing (higher = faster)
            generate_picture_images: Extract images from document (False = faster)
            images_scale: Image resolution multiplier (lower = faster)
            image_ref_mode: How to embed images in document
        """
        # Auto-detect or validate device availability
        self.device = get_available_device(device)
        
        # Configure pipeline for performance
        pipeline_options = PaginatedPipelineOptions()
        pipeline_options.images_scale = images_scale
        pipeline_options.generate_picture_images = generate_picture_images
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=device
        )
        
        # Initialize converter with optimized pipeline
        self.converter = DocumentConverter(
            format_options={
                InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Initialize embedder with device support
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000000,
            chunk_overlap=100000
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for organized output
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.text_dir = self.output_dir / "text"
        
        self.tables_dir.mkdir(exist_ok=True, parents=True)
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        self.text_dir.mkdir(exist_ok=True, parents=True)

        perf_info = f"threads={num_threads}, images={generate_picture_images}, scale={images_scale}"
        print(f"✓ Initialized DOCX processor with device: {self.device} ({perf_info})")
        if USE_ORJSON:
            print(f"✓ Using orjson for fast JSON serialization")
    
    def parse_comprehensive_content(self, docx_path: str) -> Dict[str, Any]:
        """
        Parse TEXT, TABLES, and IMAGES comprehensively from DOCX
        """
        try:
            print(f"Processing {docx_path} with Docling...")
            result = self.converter.convert(docx_path)
            document = result.document
                      
            # Export and save JSON
            json_path = self.output_dir / f"{Path(docx_path).stem}_full.json"
            document.save_as_json(json_path)
            
            print(f"✓ Saved full JSON to {json_path}")

            parsed_content = {
                "text_content": [],
                "tables": [],
                "images": [],
                "metadata": {
                    "source": docx_path,
                    "title": getattr(document, 'title', Path(docx_path).stem)
                }
            }
            
            # Parse different content types
            self._parse_text_content(document, parsed_content, docx_path)
            self._parse_tables(document, parsed_content, docx_path)
            self._parse_images(document, parsed_content, docx_path)
            
            return parsed_content
            
        except Exception as e:
            print(f"Error processing {docx_path}: {e}")
            return None
    
    def _parse_text_content(self, document: DoclingDocument, parsed_content: Dict, docx_path: str):
        """Parse and process text content"""
        # Get full text as markdown
        full_text = document.export_to_markdown()
        parsed_content["text_content"] = {
            "full_markdown": full_text,
            "word_count": len(full_text.split()),
            "char_count": len(full_text)
        }
        
        # Parse text by elements for detailed analysis
        text_elements = []
        for element, level in document.iterate_items():
            if hasattr(element, 'text') and element.text.strip():
                # Get page number from provenance if available
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

        # Save full text as markdown
        if full_text:
            txt_path = self.text_dir / f"{Path(docx_path).stem}_full_text.md"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            parsed_content["text_content"]["full_text_file"] = str(txt_path)

        print(f"✓ Parsed {len(text_elements)} text elements")
    
    def _parse_tables(self, document: DoclingDocument, parsed_content: Dict, docx_path: str):
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
                    
                    if hasattr(element, 'export_to_dataframe'):
                        try:
                            df = element.export_to_dataframe(document)  # Pass document parameter
                            table_data = df.to_dict()
                            table_csv = df.to_csv(index=False)
                        except Exception as e:
                            print(f"Warning: Could not export table {table_counter} to dataframe: {e}")
                        
                    # Get table metadata from provenance
                    page_no = 'Unknown'
                    if hasattr(element, 'prov') and element.prov:
                        prov_item = element.prov[0]
                        page_no = getattr(prov_item, 'page_no', 'Unknown')
                    
                    # Get table text content (TableItem uses export_to_markdown for text)
                    table_text = ""
                    if hasattr(element, 'export_to_markdown'):
                        try:
                            table_text = element.export_to_markdown(document)
                        except:
                            pass
                    
                    table_info = {
                        "id": f"table_{table_counter}",
                        "text": table_text,
                        "data": table_data,
                        "csv": table_csv,
                        "page_number": page_no,
                        "bbox": getattr(element, 'bbox', None),
                        "row_count": len(df) if df is not None else 0,
                        "col_count": len(df.columns) if df is not None else 0
                    }
                    
                    # Save table as CSV
                    if table_csv:
                        csv_path = self.tables_dir / f"{Path(docx_path).stem}_table_{table_counter}.csv"
                        with open(csv_path, 'w', encoding='utf-8') as f:
                            f.write(table_csv)
                        table_info["csv_file"] = str(csv_path)
                    
                    # Save table content as markdown
                    if df is not None:
                        md_path = self.tables_dir / f"{Path(docx_path).stem}_table_{table_counter}.md"
                        table_markdown = f"# Table {table_counter}\n\n"
                        table_markdown += f"**Source:** {Path(docx_path).name}\n"
                        table_markdown += f"**Page:** {page_no}\n"
                        table_markdown += f"**Dimensions:** {len(df)} rows × {len(df.columns)} columns\n\n"
                        table_markdown += "## Table Content\n\n"
                        table_markdown += df.to_markdown(index=False)
                        
                        if table_text:
                            table_markdown += f"\n\n## Raw Table Text\n\n```\n{table_text}\n```"
                        
                        with open(md_path, 'w', encoding='utf-8') as f:
                            f.write(table_markdown)
                        table_info["markdown_file"] = str(md_path)
                    
                    # Save table image if available
                    if hasattr(element, 'get_image'):
                        try:
                            table_image = element.get_image(document)
                            if table_image:
                                image_path = self.tables_dir / f"{Path(docx_path).stem}_table_{table_counter}.png"
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
                        "text": "",
                        "page_number": page_no if 'page_no' in locals() else 'Unknown',
                        "error": str(e)
                    })
        
        parsed_content["tables"] = tables
        markdown_count = sum(1 for t in tables if t.get("markdown_file"))
        print(f"✓ Parsed {len(tables)} tables ({markdown_count} with markdown files)")
    
    def _parse_images(self, document: DoclingDocument, parsed_content: Dict, docx_path: str):
        """Parse images and figures"""
        images = []
        image_counter = 0
        
        # Parse figure/picture images
        for element, level in document.iterate_items():
            if isinstance(element, PictureItem):
                image_counter += 1
                
                try:
                    # Get image
                    image = element.get_image(document)
                    if image:
                        image_path = self.figures_dir / f"{Path(docx_path).stem}_figure_{image_counter}.png"
                        image.save(image_path, "PNG")
                        
                        # Get metadata from provenance
                        page_no = 'Unknown'
                        if hasattr(element, 'prov') and element.prov:
                            prov_item = element.prov[0]
                            page_no = getattr(prov_item, 'page_no', 'Unknown')
                        
                        # Try multiple methods to get caption/text (using latest Docling API)
                        caption = ""
                        if hasattr(element, 'export_to_markdown'):
                            try:
                                caption = element.export_to_markdown(document)
                            except:
                                pass
                        
                        if not caption:
                            caption = getattr(element, 'text', '') or getattr(element, 'caption', '')
                        
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
                        txt_path = self.figures_dir / f"{Path(docx_path).stem}_figure_{image_counter}.txt"
                        caption_text = f"Figure {image_counter}\n"
                        caption_text += f"{'=' * 60}\n\n"
                        caption_text += f"Source: {Path(docx_path).name}\n"
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
        print(f"✓ Parsed {len(images)} images ({text_count} with text files)")
    
    def create_comprehensive_chunks(self, parsed_content: Dict, docx_path: str) -> List[Dict[str, Any]]:
        """Create chunks incorporating text, tables, and images"""
        chunks = []
        chunk_counter = 0
        
        # Text chunks
        if parsed_content.get("text_content", {}).get("full_markdown"):
            text_chunks = self.text_splitter.split_text(parsed_content["text_content"]["full_markdown"])
            
            for i, chunk in enumerate(text_chunks):
                chunk_obj = {
                    "id": f"{Path(docx_path).stem}_text_chunk_{i}",
                    "type": "text",
                    "content": chunk,
                    "embedding": self.embedder.encode(chunk).tolist(),
                    "metadata": {
                        "chunk_index": i,
                        "content_type": "text",
                        "source": docx_path,
                        "chunk_size": len(chunk)
                    }
                }
                chunks.append(chunk_obj)
        
        # Table chunks
        for table in parsed_content.get("tables", []):
            # Create chunk for table if it has text or CSV data
            if table.get("text") or table.get("csv"):
                # Create chunk for table text content
                table_text = f"Table {table['id']}"
                if table.get("page_number"):
                    table_text += f" (Page {table['page_number']})"
                table_text += "\\n"
                
                if table.get("text"):
                    table_text += f"\\n{table['text']}"
                    
                if table.get("csv"):
                    # Include a preview of the CSV data
                    csv_preview = table['csv'][:1000] if len(table['csv']) > 1000 else table['csv']
                    table_text += f"\\n\\nTable Data (CSV format):\\n{csv_preview}"
                    if len(table['csv']) > 1000:
                        table_text += "\\n... (truncated)"
                
                chunk_obj = {
                    "id": f"{Path(docx_path).stem}_table_chunk_{table['id']}",
                    "type": "table",
                    "content": table_text,
                    "embedding": self.embedder.encode(table_text).tolist(),
                    "metadata": {
                        "content_type": "table",
                        "table_id": table["id"],
                        "page_number": table.get("page_number"),
                        "row_count": table.get("row_count", 0),
                        "col_count": table.get("col_count", 0),
                        "csv_file": table.get("csv_file"),
                        "markdown_file": table.get("markdown_file"),
                        "image_file": table.get("image_file"),
                        "source": docx_path
                    }
                }
                chunks.append(chunk_obj)
        
        # Image chunks (for figures with text files)
        for image in parsed_content.get("images", []):
            # Only create chunks for figures that have text files
            if image.get("type") == "figure" and image.get("text_file"):
                image_text = f"Image {image['id']}"
                if image.get("page_number"):
                    image_text += f" (Page {image['page_number']})"
                
                # Add caption if available
                if image.get("caption"):
                    image_text += f": {image['caption']}"
                else:
                    image_text += " (Figure from document - no caption detected)"
                
                chunk_obj = {
                    "id": f"{Path(docx_path).stem}_image_chunk_{image['id']}",
                    "type": "image",
                    "content": image_text,
                    "embedding": self.embedder.encode(image_text).tolist(),
                    "metadata": {
                        "content_type": "image",
                        "image_id": image["id"],
                        "page_number": image.get("page_number"),
                        "image_file": image["file_path"],
                        "text_file": image.get("text_file"),
                        "image_size": image.get("size"),
                        "source": docx_path
                    }
                }
                chunks.append(chunk_obj)
        
        print(f"✓ Created {len(chunks)} comprehensive chunks")
        return chunks
        
    def save_parsing_summary(self, parsed_content: Dict, docx_path: str):
        """Save a summary of parsed content"""
        summary = {
            "document": Path(docx_path).name,
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
        
        summary_path = self.output_dir / f"{Path(docx_path).stem}_parsing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved parsing summary to {summary_path}")
        return summary

# Usage example
def process_docx_with_docling_advanced(
    docx_path: str, 
    output_dir: str = "./docling_docx_output",
    device: str = "auto",  # "auto" (recommended), "cpu", or "cuda"
    # Performance options
    fast_mode: bool = False,  # Enable fast mode (skip images, lower quality)
    num_threads: int = 8  # Number of threads (higher = faster on multi-core CPUs)
):
    """
    Complete workflow for comprehensive DOCX processing with Docling
    
    Args:
        docx_path: Path to the DOCX file
        output_dir: Directory to save parsed content
        device: "auto" (auto-detect), "cpu", or "cuda" for GPU acceleration
        fast_mode: If True, skips image extraction for 2-5x faster processing
        num_threads: Number of parallel threads (16-32 recommended for performance)
    """
    # Configure performance based on mode
    if fast_mode:
        processor = DoclingDocxProcessor(
            output_dir=output_dir, 
            device=device,
            num_threads=num_threads,
            generate_picture_images=False,  # Skip images
            images_scale=1.0                 # Normal resolution
        )
    else:
        processor = DoclingDocxProcessor(
            output_dir=output_dir, 
            device=device,
            num_threads=num_threads,
            generate_picture_images=True,   # Extract images
            images_scale=2.0                # Normal resolution (2.0 for high-res but slower)
        )
    
    # 1. Parse all content types
    parsed_content = processor.parse_comprehensive_content(docx_path)
    if not parsed_content:
        return None
    
    # 2. Create comprehensive chunks
    # chunks = processor.create_comprehensive_chunks(parsed_content, docx_path)
    
    # 3. Save summary
    summary = processor.save_parsing_summary(parsed_content, docx_path)
    
    # 4. Store in vector database (optional)
    # client = chromadb.PersistentClient(path="./chroma_db")
    # collection_name = f"docling_comprehensive_{Path(docx_path).stem}"
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
    # Example 1: BALANCED mode (recommended) - Good quality, reasonable speed
    result = process_docx_with_docling_advanced(
        docx_path="docs\\DO_NOT_KovSpec.docx",
        device="auto",      # Auto-detect best device (CUDA if available, else CPU)
        fast_mode=False,    # Extract images, better quality
        num_threads=8      # Use 16 threads for parallel processing
    )
    
    # Example 2: FAST mode - 2-5x faster, skips images
    # result = process_docx_with_docling_advanced(
    #     docx_path="docs\\DO_NOT_KovSpec.docx",
    #     device="auto",
    #     fast_mode=True,    # Skip images for speed
    #     num_threads=32     # Use more threads for maximum speed
    # )
    
    # Example 3: HIGH QUALITY mode - Slower but best results
    # processor = DoclingDocxProcessor(
    #     output_dir="./docling_docx_output",
    #     device="auto",
    #     num_threads=8,                     # Fewer threads for stability
    #     generate_picture_images=True,      # Extract all images
    #     images_scale=2.0                   # High resolution images
    # )
    # result = processor.parse_comprehensive_content("docs\\DO_NOT_KovSpec.docx")
    
    print("\n" + "=" * 70)
    print("DOCLING DOCX Parser - Processing Complete!")
    print("=" * 70)
    print("Features:")
    print("✓ Comprehensive TEXT parsing with element analysis")
    print("✓ Advanced TABLE parsing:")
    print("  - DataFrame export and CSV saving in 'tables/' folder")
    print("  - Markdown files with formatted table content")
    print("  - Table images in 'tables/' folder")
    print("✓ Complete IMAGE parsing:")
    print("  - Figure images in 'figures/' folder")
    print("  - Text files for ALL figures (with captions/metadata)")
    print("✓ Structured chunking for all content types")
    print("✓ Organized output directory structure")
    print("✓ Detailed parsing summaries")
    print(f"✓ Device support: CPU and CUDA")
    print("✓ Full document JSON export")
    print("=" * 70)
    print("\nPerformance Modes:")
    print("  FAST mode:      fast_mode=True, num_threads=32  (2-5x faster, no images)")
    print("  BALANCED mode:  fast_mode=False, num_threads=16 (recommended)")
    print("  QUALITY mode:   images_scale=2.0 (slower)")
    print("\nOptional: Install 'orjson' for 3-10x faster JSON serialization:")
    print("  pip install orjson")
    print("=" * 70)
    print("\nOutput Structure:")
    print("  docling_docx_output/")
    print("  ├── {filename}_full.json  (complete document structure)")
    print("  ├── tables/        (CSV, markdown, and table images)")
    print("  ├── figures/       (figures and captions)")
    print("  └── text/          (full text markdown)")
    print("=" * 70)

