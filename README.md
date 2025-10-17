# rag-experiments
RAG experiments with langchain, llama_index, docling, unstructured

## Installation

### 1. Install System Dependencies (Required)

For maximum compatibility with document processing libraries, you need to install system-level dependencies first.

**Quick Install (Linux/Mac):**
```bash
chmod +x install_system_deps.sh
./install_system_deps.sh
```

**Quick Install (Windows):**
```bash
install_system_deps.bat
```

**Manual Installation:**

The following system packages are required:
- **libmagic-dev** (Linux) / **libmagic** (Mac) - for filetype detection
- **poppler-utils** - for PDF processing
- **tesseract-ocr** - for OCR support on images and PDFs
- **tesseract-lang** packages - for additional language support
- **libreoffice** - for Microsoft Office document processing
- **pandoc** (version 2.14.2+) - for .epub, .odt, and .rtf files

<details>
<summary>Linux (Ubuntu/Debian)</summary>

```bash
sudo apt-get update
sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr \
  tesseract-ocr-eng tesseract-ocr-hun libreoffice

# Install Pandoc 3.x (required version 2.14.2+)
wget https://github.com/jgm/pandoc/releases/download/3.1.11/pandoc-3.1.11-1-amd64.deb
sudo dpkg -i pandoc-3.1.11-1-amd64.deb
rm pandoc-3.1.11-1-amd64.deb
```
</details>

<details>
<summary>macOS</summary>

```bash
brew install libmagic poppler tesseract tesseract-lang pandoc
brew install --cask libreoffice
```
</details>

<details>
<summary>Windows</summary>

Install via Chocolatey:
```bash
choco install -y tesseract poppler libreoffice-fresh pandoc
```

Or download installers manually:
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases/
- LibreOffice: https://www.libreoffice.org/download/download/
- Pandoc: https://pandoc.org/installing.html
</details>

### 2. Create and activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install PyTorch with CUDA support (for GPU acceleration)

**Quick Install (Windows):**
```bash
install_cuda.bat
```

**Quick Install (Linux/Mac):**
```bash
chmod +x install_cuda.sh
./install_cuda.sh
```

**Manual Installation:**

For CUDA 12.1 (Recommended for RTX 30/40 series):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8 (Older GPUs):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch torchvision torchaudio
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The `unstructured` library requires the system dependencies listed in step 1 to work properly. Without them, certain document types may fail to process.

## Features

- **Docling**: Advanced PDF extraction with text, tables, and images
  - GPU/CPU support with automatic device detection
  - OCR support for scanned documents
  - High-resolution image extraction
  
- **Unstructured.io**: Multi-format document processing

- **MarkItDown**: Markdown conversion utilities

## Usage

### Docling Extraction
```bash
cd extract/docling_extraction
python docling_extraction.py
```

The script will automatically detect and use CUDA if available, otherwise fall back to CPU.

## Project Structure
```
rag-experiments/
├── docs/                          # Sample documents
├── extract/                       # Extraction tools
│   ├── docling_extraction/       # Docling-based extraction
│   ├── unstructured_io_extraction/  # Unstructured.io extraction
│   └── markitdown/               # MarkItDown utilities
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```
