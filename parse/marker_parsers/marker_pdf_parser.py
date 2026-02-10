# =========================================================
# MARKER - COMPREHENSIVE TEXT, TABLE, AND IMAGE PARSING
# =========================================================
#
# This script uses the Marker library (https://github.com/datalab-to/marker)
# to parse PDF documents and extract:
# - Full text in Markdown, HTML, and plain text formats
# - Tables with CSV, HTML, and Markdown output
# - Figures and images
# - Page-level metadata
#
# Marker converts PDFs using deep learning models (surya, texify)
# and supports GPU acceleration with CUDA.
#
# Installation:
#   pip install marker-pdf
#   pip install marker-pdf[full]    # for non-PDF formats
#   pip install pandas tabulate markdown lxml  # for table/HTML conversion
#
# Architecture note:
#   This parser runs TWO conversion passes over the PDF:
#   1. Markdown pass  -> clean text + extracted images
#   2. JSON pass      -> structured block tree for table/figure extraction
#   Models are loaded ONCE and shared between passes (create_model_dict).
#   The pipeline (layout detection, OCR, etc.) runs twice, which is the
#   trade-off for getting both clean markdown AND structured block data.

import os
import json
import re
import base64
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from io import StringIO, BytesIO

import torch
import pandas as pd
from PIL import Image

# Enable logging so Marker conversion shows progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("marker").setLevel(logging.INFO)

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser


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
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            print("CUDA requested but not available. Falling back to CPU.")
            print("  To enable CUDA, install PyTorch with CUDA support:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return "cpu"

    return "cpu"


class MarkerPdfProcessor:
    """
    Comprehensive PDF parser using the Marker library.

    Extracts text, tables, figures, and page metadata from PDF files,
    mirroring the functionality of the DoclingAdvancedProcessor.
    """

    def __init__(
        self,
        output_dir: str = "./marker_pdf_output",
        device: str = "auto",
        force_ocr: bool = False,
        use_llm: bool = False,
        paginate_output: bool = False,
    ):
        """
        Initialize the MarkerPdfProcessor.

        Args:
            output_dir: Directory to save parsed content
            device: Device to use for inference ("cpu", "cuda", or "auto")
            force_ocr: Force OCR on all pages (useful for scanned PDFs or inline math)
            use_llm: Use an LLM to improve accuracy (requires Gemini/Ollama/etc. config)
            paginate_output: Whether to paginate the output with page markers
        """
        self.device = get_available_device(device)
        # Tell Marker/PyTorch which device to use
        os.environ["TORCH_DEVICE"] = self.device

        self.force_ocr = force_ocr
        self.use_llm = use_llm
        self.paginate_output = paginate_output

        # Load models once — this is the expensive step (downloads & loads weights)
        print("Loading Marker models (this may take a while on first run)...")
        model_start = time.time()
        self.artifact_dict = create_model_dict()
        model_elapsed = time.time() - model_start
        print(f"Models loaded in {model_elapsed:.1f}s")

        # Create organized output directory structure
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.tables_dir = self.output_dir / "tables"
        self.pages_dir = self.output_dir / "pages"
        self.figures_dir = self.output_dir / "figures"
        self.text_dir = self.output_dir / "text"

        self.tables_dir.mkdir(exist_ok=True, parents=True)
        self.pages_dir.mkdir(exist_ok=True, parents=True)
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        self.text_dir.mkdir(exist_ok=True, parents=True)

        print(
            f"Initialized MarkerPdfProcessor "
            f"(device={self.device}, force_ocr={force_ocr}, use_llm={use_llm})"
        )

    # ------------------------------------------------------------------
    # Converter factory
    # ------------------------------------------------------------------

    def _create_converter(self, output_format: str) -> PdfConverter:
        """
        Create a PdfConverter configured for the given output format.

        Args:
            output_format: One of "markdown", "json", "html", "chunks"

        Returns:
            Configured PdfConverter instance (shares pre-loaded models)
        """
        config: Dict[str, Any] = {
            "output_format": output_format,
        }
        if self.force_ocr:
            config["force_ocr"] = True
        if self.paginate_output:
            config["paginate_output"] = True

        config_parser = ConfigParser(config)

        converter_kwargs: Dict[str, Any] = {
            "config": config_parser.generate_config_dict(),
            "artifact_dict": self.artifact_dict,
            "processor_list": config_parser.get_processors(),
            "renderer": config_parser.get_renderer(),
        }

        # Attach LLM service when requested
        if self.use_llm:
            try:
                converter_kwargs["llm_service"] = config_parser.get_llm_service()
            except Exception as e:
                print(f"Warning: Could not initialize LLM service: {e}")

        return PdfConverter(**converter_kwargs)

    # ------------------------------------------------------------------
    # Main parsing entry point
    # ------------------------------------------------------------------

    def parse_comprehensive_content(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse TEXT, TABLES, and IMAGES comprehensively from a PDF.

        Runs two conversion passes:
        1. Markdown -> clean full text + extracted images
        2. JSON     -> structured block tree for table/figure extraction

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dict with keys: text_content, tables, images, metadata
            or None on failure.
        """
        try:
            print(f"\nProcessing {pdf_path} with Marker...")
            pdf_stem = Path(pdf_path).stem

            parsed_content: Dict[str, Any] = {
                "text_content": {},
                "tables": [],
                "images": [],
                "metadata": {
                    "source": pdf_path,
                    "title": pdf_stem,
                },
            }

            # ── Pass 1: Markdown output ────────────────────────────
            print("\n--- Pass 1: Markdown conversion ---")
            md_start = time.time()
            md_converter = self._create_converter("markdown")
            md_rendered = md_converter(pdf_path)
            md_text, md_metadata, md_images = text_from_rendered(md_rendered)
            md_elapsed = time.time() - md_start
            print(f"Markdown conversion done in {md_elapsed:.1f}s")

            # ── Pass 2: JSON output ────────────────────────────────
            print("\n--- Pass 2: JSON conversion ---")
            json_start = time.time()
            json_converter = self._create_converter("json")
            json_rendered = json_converter(pdf_path)
            json_text, json_metadata, json_images = text_from_rendered(json_rendered)
            json_elapsed = time.time() - json_start
            print(f"JSON conversion done in {json_elapsed:.1f}s")

            # ── Metadata ───────────────────────────────────────────
            parsed_content["metadata"]["conversion_time_markdown"] = md_elapsed
            parsed_content["metadata"]["conversion_time_json"] = json_elapsed

            if md_metadata:
                parsed_content["metadata"]["table_of_contents"] = md_metadata.get(
                    "table_of_contents", []
                )
                parsed_content["metadata"]["page_stats"] = md_metadata.get(
                    "page_stats", []
                )
                parsed_content["metadata"]["total_pages"] = len(
                    md_metadata.get("page_stats", [])
                )

            # ── Parse the JSON tree ────────────────────────────────
            json_data = None
            try:
                json_data = (
                    json.loads(json_text) if isinstance(json_text, str) else json_text
                )
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not parse JSON output: {e}")

            # Save full JSON to output directory
            json_path = self.output_dir / f"{pdf_stem}_full.json"
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    if isinstance(json_data, (dict, list)):
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    else:
                        f.write(str(json_text))
                print(f"Saved full JSON to {json_path}")
            except Exception as e:
                print(f"Warning: Could not save JSON: {e}")

            # ── Extract content types ─────────────────────────────
            self._parse_text_content(md_text, md_metadata, parsed_content, pdf_path)
            self._parse_tables(json_data, parsed_content, pdf_path)
            self._parse_images(
                md_images, json_images, json_data, parsed_content, pdf_path
            )

            return parsed_content

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Text content
    # ------------------------------------------------------------------

    def _parse_text_content(
        self,
        markdown_text: str,
        metadata: Any,
        parsed_content: Dict[str, Any],
        pdf_path: str,
    ):
        """Parse and save text content in multiple formats."""
        print("\nParsing text content...")
        pdf_stem = Path(pdf_path).stem

        full_markdown = markdown_text or ""
        full_text = self._markdown_to_plain_text(full_markdown)
        full_html = self._markdown_to_html(full_markdown)

        parsed_content["text_content"] = {
            "full_markdown": full_markdown,
            "full_html": full_html,
            "full_text": full_text,
            "word_count": len(full_markdown.split()),
            "char_count": len(full_markdown),
        }

        # ── Save to text/ subdirectory ────────────────────────────
        if full_markdown:
            md_path = self.text_dir / f"{pdf_stem}_full_markdown.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(full_markdown)
            parsed_content["text_content"]["full_markdown_file"] = str(md_path)
            print(f"  Saved markdown to {md_path}")

        if full_html:
            html_path = self.text_dir / f"{pdf_stem}_full_html.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(full_html)
            parsed_content["text_content"]["full_html_file"] = str(html_path)
            print(f"  Saved HTML to {html_path}")

        if full_text:
            txt_path = self.text_dir / f"{pdf_stem}_full_text.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            parsed_content["text_content"]["full_text_file"] = str(txt_path)
            print(f"  Saved plain text to {txt_path}")

        # ── Also save to output root (like docling does) ──────────
        root_md = self.output_dir / f"{pdf_stem}_full.md"
        with open(root_md, "w", encoding="utf-8") as f:
            f.write(full_markdown)

        root_html = self.output_dir / f"{pdf_stem}_full.html"
        with open(root_html, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(
            f"Parsed text: {parsed_content['text_content']['word_count']} words, "
            f"{parsed_content['text_content']['char_count']} chars"
        )

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _parse_tables(
        self,
        json_data: Any,
        parsed_content: Dict[str, Any],
        pdf_path: str,
    ):
        """
        Parse tables from the JSON block tree.

        Finds all blocks with block_type == "Table", extracts the HTML,
        and converts to CSV / Markdown / DataFrame via pandas.
        """
        print("\nParsing tables...")
        pdf_stem = Path(pdf_path).stem
        tables: List[Dict[str, Any]] = []

        if json_data is None:
            print("  No JSON data available — skipping table extraction.")
            parsed_content["tables"] = tables
            return

        table_blocks = self._find_blocks_by_type(json_data, ["Table"])

        for idx, block in enumerate(table_blocks, start=1):
            table_html = block.get("html", "")
            block_id = block.get("id", f"table_{idx}")
            polygon = block.get("polygon", None)
            page_no = self._extract_page_from_id(block_id)

            table_info: Dict[str, Any] = {
                "id": f"table_{idx}",
                "block_id": block_id,
                "page_number": page_no,
                "polygon": polygon,
                "html": table_html,
                "text": None,
                "csv": None,
                "markdown": None,
                "data": None,
                "row_count": 0,
                "col_count": 0,
            }

            # Try converting the HTML table to a DataFrame
            if table_html:
                try:
                    dfs = pd.read_html(StringIO(table_html))
                    if dfs:
                        df = dfs[0]
                        # De-duplicate column names to prevent data loss in to_dict()
                        if df.columns.duplicated().any():
                            cols = list(df.columns)
                            seen: Dict[str, int] = {}
                            for i, col in enumerate(cols):
                                col_str = str(col)
                                if col_str in seen:
                                    seen[col_str] += 1
                                    cols[i] = f"{col_str}_{seen[col_str]}"
                                else:
                                    seen[col_str] = 0
                            df.columns = cols

                        table_info["data"] = df.to_dict()
                        table_info["csv"] = df.to_csv(index=False)
                        try:
                            table_info["markdown"] = df.to_markdown(index=False)
                        except ImportError:
                            # tabulate not installed — fall back to CSV
                            table_info["markdown"] = df.to_csv(index=False, sep="|")
                        table_info["text"] = table_info["markdown"]
                        table_info["row_count"] = len(df)
                        table_info["col_count"] = len(df.columns)
                except Exception as e:
                    print(
                        f"  Warning: Could not convert table {idx} HTML to DataFrame: {e}"
                    )
                    # Fallback: strip HTML tags for plain text
                    table_info["text"] = re.sub(r"<[^>]+>", " ", table_html).strip()

            # ── Save table files ──────────────────────────────────
            if table_info["csv"]:
                csv_path = self.tables_dir / f"{pdf_stem}_table_{idx}.csv"
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(table_info["csv"])
                table_info["csv_file"] = str(csv_path)

            if table_info["html"]:
                html_path = self.tables_dir / f"{pdf_stem}_table_{idx}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(table_info["html"])
                table_info["html_file"] = str(html_path)

            if table_info["markdown"]:
                md_path = self.tables_dir / f"{pdf_stem}_table_{idx}.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(table_info["markdown"])
                table_info["markdown_file"] = str(md_path)

            if table_info["text"]:
                txt_path = self.tables_dir / f"{pdf_stem}_table_{idx}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(table_info["text"])
                table_info["text_file"] = str(txt_path)

            tables.append(table_info)

        parsed_content["tables"] = tables
        print(f"Parsed {len(tables)} tables")

    # ------------------------------------------------------------------
    # Images / Figures
    # ------------------------------------------------------------------

    def _parse_images(
        self,
        md_images: Optional[Dict],
        json_images: Optional[Dict],
        json_data: Any,
        parsed_content: Dict[str, Any],
        pdf_path: str,
    ):
        """
        Parse and save images / figures.

        Sources:
        1. Images extracted by Marker (returned by text_from_rendered)
        2. Base64-encoded images inside JSON Figure/Picture blocks
        """
        print("\nParsing images and figures...")
        pdf_stem = Path(pdf_path).stem
        images: List[Dict[str, Any]] = []
        figure_counter = 0
        saved_image_hashes: set = set()  # avoid duplicates across sources

        # ── Source 1: Extracted images from rendered output ────────
        all_rendered_images: Dict = {}
        if md_images:
            all_rendered_images.update(md_images)
        if json_images:
            all_rendered_images.update(json_images)

        for img_name, img in all_rendered_images.items():
            if not isinstance(img, Image.Image):
                continue

            # Simple dedup by image size + mode
            img_hash = (img.size, img.mode, img_name)
            if img_hash in saved_image_hashes:
                continue
            saved_image_hashes.add(img_hash)

            figure_counter += 1
            try:
                image_path = (
                    self.figures_dir / f"{pdf_stem}_figure_{figure_counter}.png"
                )
                img.save(str(image_path), "PNG")

                image_info: Dict[str, Any] = {
                    "type": "figure",
                    "id": f"figure_{figure_counter}",
                    "original_name": img_name,
                    "file_path": str(image_path),
                    "size": img.size,
                    "page_number": "Unknown",
                }

                # Save a text placeholder for each figure
                txt_path = (
                    self.figures_dir / f"{pdf_stem}_figure_{figure_counter}.txt"
                )
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"Figure {figure_counter} "
                        f"(extracted by Marker, original name: {img_name})"
                    )
                image_info["text_file"] = str(txt_path)

                images.append(image_info)
            except Exception as e:
                print(f"  Error saving image {img_name}: {e}")

        # ── Source 2: Base64 images inside JSON blocks ────────────
        if json_data is not None:
            figure_blocks = self._find_blocks_by_type(
                json_data, ["Figure", "Picture", "FigureGroup", "PictureGroup"]
            )
            for block in figure_blocks:
                block_id = block.get("id", "")
                page_no = self._extract_page_from_id(block_id)
                polygon = block.get("polygon", None)
                block_images = block.get("images", {})

                if not block_images:
                    continue

                for img_id, img_data in block_images.items():
                    if not isinstance(img_data, str) or not img_data:
                        continue

                    figure_counter += 1
                    try:
                        img_bytes = base64.b64decode(img_data)
                        pil_img = Image.open(BytesIO(img_bytes))

                        # Dedup check
                        img_hash = (pil_img.size, pil_img.mode, img_id)
                        if img_hash in saved_image_hashes:
                            figure_counter -= 1
                            continue
                        saved_image_hashes.add(img_hash)

                        image_path = (
                            self.figures_dir
                            / f"{pdf_stem}_figure_{figure_counter}.png"
                        )
                        pil_img.save(str(image_path), "PNG")

                        image_info = {
                            "type": "figure",
                            "id": f"figure_{figure_counter}",
                            "block_id": block_id,
                            "page_number": page_no,
                            "polygon": polygon,
                            "file_path": str(image_path),
                            "size": pil_img.size,
                        }

                        txt_path = (
                            self.figures_dir
                            / f"{pdf_stem}_figure_{figure_counter}.txt"
                        )
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(f"Figure {figure_counter} from page {page_no}")
                        image_info["text_file"] = str(txt_path)

                        images.append(image_info)
                    except Exception as e:
                        print(
                            f"  Error decoding base64 image from block {block_id}: {e}"
                        )
                        figure_counter -= 1

        parsed_content["images"] = images
        figure_count = sum(1 for img in images if img.get("type") == "figure")
        print(f"Saved {len(images)} images ({figure_count} figures)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_blocks_by_type(
        self, data: Any, block_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Recursively search the JSON block tree for blocks matching given types.

        The Marker JSON output is a tree:
          Document -> Pages -> Blocks -> (nested children)
        """
        results: List[Dict[str, Any]] = []

        if isinstance(data, dict):
            if data.get("block_type") in block_types:
                results.append(data)
            # Recurse into children
            children = data.get("children")
            if children and isinstance(children, list):
                for child in children:
                    results.extend(self._find_blocks_by_type(child, block_types))
        elif isinstance(data, list):
            for item in data:
                results.extend(self._find_blocks_by_type(item, block_types))

        return results

    @staticmethod
    def _extract_page_from_id(block_id: str) -> Any:
        """
        Extract page number from a Marker block ID.

        Block IDs follow the pattern ``/page/<N>/<Type>/<M>``.
        """
        match = re.search(r"/page/(\d+)/", str(block_id))
        if match:
            return int(match.group(1))
        return "Unknown"

    @staticmethod
    def _markdown_to_plain_text(markdown: str) -> str:
        """Strip Markdown formatting to produce plain text."""
        text = markdown
        # Images: ![alt](url) -> alt
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        # Links: [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Bold / italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        # Headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Fenced code blocks
        text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # LaTeX fences
        text = re.sub(r"\$\$([^$]+)\$\$", r"\1", text)
        text = re.sub(r"\$([^$]+)\$", r"\1", text)
        # Horizontal rules
        text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\*{3,}$", "", text, flags=re.MULTILINE)
        # Collapse excess blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _markdown_to_html(markdown: str) -> str:
        """
        Convert Markdown to HTML.

        Uses the ``markdown`` library if installed, otherwise falls back
        to basic regex-based conversion.
        """
        try:
            import markdown as md_lib

            return md_lib.markdown(
                markdown, extensions=["tables", "fenced_code", "toc"]
            )
        except ImportError:
            # Minimal fallback
            html = markdown
            for level in range(6, 0, -1):
                pattern = r"^" + "#" * level + r"\s+(.+)$"
                html = re.sub(
                    pattern, f"<h{level}>\\1</h{level}>", html, flags=re.MULTILINE
                )
            html = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html)
            html = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", html)
            paragraphs = re.split(r"\n{2,}", html)
            html = "".join(f"<p>{p.strip()}</p>\n" for p in paragraphs if p.strip())
            return f"<html><body>\n{html}</body></html>"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def save_parsing_summary(
        self, parsed_content: Dict[str, Any], pdf_path: str
    ) -> Dict[str, Any]:
        """Save a JSON summary of all parsed content."""
        print("\nSaving parsing summary...")
        summary: Dict[str, Any] = {
            "document": Path(pdf_path).name,
            "parsing_tool": "Marker (marker-pdf)",
            "parsing_summary": {
                "text": {
                    "word_count": parsed_content.get("text_content", {}).get(
                        "word_count", 0
                    ),
                    "char_count": parsed_content.get("text_content", {}).get(
                        "char_count", 0
                    ),
                },
                "tables": {
                    "count": len(parsed_content.get("tables", [])),
                    "csv_files": [
                        t.get("csv_file")
                        for t in parsed_content.get("tables", [])
                        if t.get("csv_file")
                    ],
                    "markdown_files": [
                        t.get("markdown_file")
                        for t in parsed_content.get("tables", [])
                        if t.get("markdown_file")
                    ],
                },
                "images": {
                    "count": len(parsed_content.get("images", [])),
                    "image_files": [
                        img["file_path"]
                        for img in parsed_content.get("images", [])
                        if img.get("file_path")
                    ],
                    "text_files": [
                        img.get("text_file")
                        for img in parsed_content.get("images", [])
                        if img.get("text_file")
                    ],
                },
                "metadata": parsed_content.get("metadata", {}),
            },
        }

        summary_path = self.output_dir / f"{Path(pdf_path).stem}_parsing_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        print(f"Saved parsing summary to {summary_path}")
        return summary


# ======================================================================
# Convenience wrapper
# ======================================================================


def process_pdf_with_marker(
    pdf_path: str,
    output_dir: str = "./marker_pdf_output",
    device: str = "auto",
    force_ocr: bool = False,
    use_llm: bool = False,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """
    Complete workflow for comprehensive PDF processing with Marker.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save parsed content
        device: "auto" (auto-detect best), "cpu", or "cuda"
        force_ocr: Force OCR on all pages (set True for scanned PDFs)
        use_llm: Use an LLM to boost accuracy (needs API key configuration)

    Returns:
        Tuple of (parsed_content, summary, output_dir) or None on failure
    """
    processor = MarkerPdfProcessor(
        output_dir=output_dir,
        device=device,
        force_ocr=force_ocr,
        use_llm=use_llm,
    )

    # 1. Parse all content types
    parsed_content = processor.parse_comprehensive_content(pdf_path)
    if not parsed_content:
        return None

    # 2. Save summary
    summary = processor.save_parsing_summary(parsed_content, pdf_path)

    return parsed_content, summary, output_dir


# ======================================================================
# Usage example
# ======================================================================

if __name__ == "__main__":
    # ── Example 1: Text-based PDF with auto device detection ──────
    pdf_path = "./docs/DO_NOT_KovSpec.pdf"
    result = process_pdf_with_marker(
        pdf_path=pdf_path,
        output_dir="./marker_pdf_output",
        device="auto",      # auto-detect CUDA or CPU
        force_ocr=False,     # False for text-based PDFs
        use_llm=False,       # Set True + configure API key for higher accuracy
    )

    if result is not None:
        parsed_content, summary, output_dir = result

        # Print concise statistics
        text_info = parsed_content.get("text_content", {})
        tables_info = parsed_content.get("tables", [])
        images_info = parsed_content.get("images", [])
        meta = parsed_content.get("metadata", {})
        figure_count = sum(1 for img in images_info if img.get("type") == "figure")

        print(f"\n{'=' * 50}")
        print(f"  Parsing Results for: {Path(pdf_path).stem}")
        print(f"{'=' * 50}")
        print(f"  Pages:    {meta.get('total_pages', 'N/A')}")
        print(
            f"  Text:     {text_info.get('word_count', 0)} words, "
            f"{text_info.get('char_count', 0)} chars"
        )
        print(f"  Tables:   {len(tables_info)}")
        print(f"  Figures:  {figure_count}")
        print(f"  Output:   {output_dir}")
        print(f"{'=' * 50}")
    else:
        print("Processing failed. Check error messages above.")

    # ── Example 2: Force CPU usage ────────────────────────────────
    # result = process_pdf_with_marker(
    #     pdf_path="./docs/DO_NOT_KovSpec.pdf",
    #     device="cpu",
    #     force_ocr=False,
    # )

    # ── Example 3: Scanned PDF with OCR ───────────────────────────
    # result = process_pdf_with_marker(
    #     pdf_path="./docs/scanned_document.pdf",
    #     device="auto",
    #     force_ocr=True,   # Enable OCR for scanned PDFs
    # )

    # ── Example 4: High accuracy with LLM ────────────────────────
    # Requires: GOOGLE_API_KEY env var for Gemini, or configure another LLM
    # result = process_pdf_with_marker(
    #     pdf_path="./docs/complex_document.pdf",
    #     device="auto",
    #     force_ocr=True,
    #     use_llm=True,     # Uses Gemini 2.0 Flash by default
    # )

    print("\n" + "=" * 70)
    print("MARKER PDF Parser - Processing Complete!")
    print("=" * 70)
    print("Features:")
    print("  Comprehensive TEXT parsing (Markdown, HTML, plain text)")
    print("  Advanced TABLE parsing:")
    print("    - HTML, CSV, Markdown exports in 'tables/' folder")
    print("    - DataFrame conversion via pandas")
    print("  Complete IMAGE parsing:")
    print("    - Figure images in 'figures/' folder")
    print("    - Text placeholders for each figure")
    print("  Full Markdown + HTML in 'text/' subdirectory")
    print("  Organized output directory structure")
    print("  Detailed parsing summaries")
    print("=" * 70)
