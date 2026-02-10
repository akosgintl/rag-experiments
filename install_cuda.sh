#!/bin/bash
# Install PyTorch with CUDA support for GPU acceleration

echo "========================================"
echo "Installing PyTorch with CUDA 12.1"
echo "========================================"
echo ""

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "========================================"
echo "Installation complete!"
echo "Testing CUDA availability..."
echo "========================================"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"

echo ""
echo "========================================"

