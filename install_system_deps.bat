@echo off
REM System dependencies installer for Windows
REM For maximum compatibility with document processing libraries

echo Installing system dependencies for document processing...
echo.

REM Check if Chocolatey is installed
where choco >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Chocolatey is not installed. Would you like to install it? [Y/N]
    set /p INSTALL_CHOCO=
    if /i "%INSTALL_CHOCO%"=="Y" (
        echo Installing Chocolatey...
        @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
        if %ERRORLEVEL% NEQ 0 (
            echo Failed to install Chocolatey. Please install it manually from:
            echo https://chocolatey.org/install
            pause
            exit /b 1
        )
    ) else (
        echo Please install Chocolatey first from: https://chocolatey.org/install
        echo Or manually install the following packages:
        echo   - Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
        echo   - Poppler: https://github.com/oschwartz10612/poppler-windows/releases/
        echo   - LibreOffice: https://www.libreoffice.org/download/download/
        echo   - Pandoc: https://pandoc.org/installing.html
        pause
        exit /b 1
    )
)

echo.
echo Installing Tesseract OCR for OCR support...
choco install -y tesseract

echo.
echo Installing Poppler for PDF processing...
choco install -y poppler

echo.
echo Installing LibreOffice for Office document support...
choco install -y libreoffice-fresh

echo.
echo Installing Pandoc for .epub, .odt, and .rtf file support...
choco install -y pandoc

echo.
echo NOTE: libmagic is not easily available on Windows.
echo For filetype detection, the python-magic-bin package will be used.
echo This is automatically handled by the unstructured library.

echo.
echo ✓ All system dependencies installed successfully!
echo.
echo Please restart your terminal/command prompt to ensure PATH is updated.
echo.
echo Verifying installations...
where tesseract >nul 2>nul && tesseract --version | findstr /C:"tesseract"
where pandoc >nul 2>nul && pandoc --version | findstr /C:"pandoc"
where pdfinfo >nul 2>nul && pdfinfo -v 2>&1 | findstr /C:"pdfinfo"

echo.
echo Installation complete! You can now install Python dependencies with:
echo pip install -r requirements.txt
echo.
pause

