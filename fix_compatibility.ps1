# Quick fix script for Python 3.8 compatibility issues
# Run this if you encounter "TypeError: 'type' object is not subscriptable"

Write-Host "Fixing Python 3.8 compatibility issues..." -ForegroundColor Yellow

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Detected: $pythonVersion" -ForegroundColor Cyan

# Uninstall problematic packages
Write-Host "`nUninstalling problematic packages..." -ForegroundColor Yellow
pip uninstall -y chromadb posthog

# Install compatible versions
Write-Host "`nInstalling Python 3.8 compatible packages..." -ForegroundColor Yellow
pip install typing-extensions>=4.0.0
pip install 'chromadb==0.4.18'
pip install 'posthog<3.0.0'

# Reinstall other dependencies
Write-Host "`nReinstalling other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --upgrade

Write-Host "`n✓ Fix complete! Try running your script again." -ForegroundColor Green
Write-Host "`nIf issues persist, consider upgrading to Python 3.9+" -ForegroundColor Yellow

