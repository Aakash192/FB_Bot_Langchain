# PowerShell installation script for Python 3.8 compatibility

Write-Host "Installing Python 3.8 compatible versions..." -ForegroundColor Green

# Install typing-extensions first
pip install typing-extensions>=4.0.0

# Install compatible ChromaDB version
pip install 'chromadb==0.4.18'

# Install other dependencies
pip install -r requirements.txt

Write-Host "Installation complete!" -ForegroundColor Green

