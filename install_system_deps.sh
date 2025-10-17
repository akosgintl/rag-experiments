#!/bin/bash
# System dependencies installer for Linux/Mac
# For maximum compatibility with document processing libraries

echo "Installing system dependencies for document processing..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    
    # Update package list
    sudo apt-get update
    
    # Install libmagic for filetype detection
    echo "Installing libmagic-dev for filetype detection..."
    sudo apt-get install -y libmagic-dev
    
    # Install poppler-utils for PDF processing
    echo "Installing poppler-utils for PDF processing..."
    sudo apt-get install -y poppler-utils
    
    # Install Tesseract OCR
    echo "Installing tesseract-ocr for OCR support..."
    sudo apt-get install -y tesseract-ocr
    
    # Install Tesseract language packs
    echo "Installing Tesseract language packs (English and Hungarian)..."
    sudo apt-get install -y tesseract-ocr-eng tesseract-ocr-hun
    
    # Install LibreOffice for Microsoft Office document processing
    echo "Installing LibreOffice for Office document support..."
    sudo apt-get install -y libreoffice
    
    # Install Pandoc (version 2.14.2 or newer required for .rtf files)
    echo "Installing Pandoc for .epub, .odt, and .rtf file support..."
    PANDOC_VERSION="3.1.11"
    PANDOC_DEB="pandoc-${PANDOC_VERSION}-1-amd64.deb"
    wget https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/${PANDOC_DEB}
    sudo dpkg -i ${PANDOC_DEB}
    rm ${PANDOC_DEB}
    
    echo "✓ All system dependencies installed successfully!"
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is not installed. Please install it first:"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    
    # Install libmagic for filetype detection
    echo "Installing libmagic for filetype detection..."
    brew install libmagic
    
    # Install poppler for PDF processing
    echo "Installing poppler for PDF processing..."
    brew install poppler
    
    # Install Tesseract OCR
    echo "Installing tesseract for OCR support..."
    brew install tesseract
    
    # Install Tesseract language packs
    echo "Installing Tesseract language packs..."
    brew install tesseract-lang
    
    # Install LibreOffice
    echo "Installing LibreOffice for Office document support..."
    brew install --cask libreoffice
    
    # Install Pandoc
    echo "Installing Pandoc for .epub, .odt, and .rtf file support..."
    brew install pandoc
    
    echo "✓ All system dependencies installed successfully!"
    
else
    echo "Error: Unsupported operating system: $OSTYPE"
    echo "This script supports Linux and macOS only."
    echo "For Windows, please see install_system_deps.bat"
    exit 1
fi

# Verify installations
echo ""
echo "Verifying installations..."
echo "Tesseract version: $(tesseract --version | head -n 1)"
echo "Pandoc version: $(pandoc --version | head -n 1)"
echo "Poppler (pdfinfo) version: $(pdfinfo -v 2>&1 | head -n 1)"

echo ""
echo "Installation complete! You can now install Python dependencies with:"
echo "pip install -r requirements.txt"

