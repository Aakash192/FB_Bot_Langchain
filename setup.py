"""
Setup script for DOCX RAG Pipeline

This script helps set up the environment and verify installation.
"""

import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages from requirements.txt."""
    print("\n📦 Installing dependencies...")
    
    # Use Python 3.8 compatible requirements if needed
    requirements_file = "requirements.txt"
    if sys.version_info < (3, 9):
        print("   Python 3.8 detected - using compatible package versions...")
        requirements_file = "requirements_python38.txt"
        # Also install typing-extensions explicitly
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "typing-extensions>=4.0.0"])
        except:
            pass
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        if sys.version_info < (3, 9):
            print("\n   For Python 3.8, you may need to run:")
            print("   pip install 'chromadb==0.4.18' typing-extensions>=4.0.0")
        return False

def check_env_file():
    """Check if .env file exists."""
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file found")
        return True
    else:
        print("⚠️  .env file not found")
        print("   Please create a .env file with your OPENAI_API_KEY")
        print("   You can copy env_template.txt to .env and add your key")
        return False

def create_directories():
    """Create necessary directories."""
    dirs = ["data", "chroma_db"]
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"✓ Created directory: {dir_name}")
        else:
            print(f"✓ Directory exists: {dir_name}")

def verify_imports():
    """Verify that all required modules can be imported."""
    print("\n🔍 Verifying imports...")
    try:
        import docx
        import openai
        import chromadb
        import tiktoken
        from dotenv import load_dotenv
        print("✓ All required modules imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please run: pip install -r requirements.txt")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("DOCX RAG Pipeline - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check .env file
    print("\n🔐 Checking configuration...")
    check_env_file()
    
    # Install requirements
    install_requirements()
    
    # Verify imports
    if verify_imports():
        print("\n" + "=" * 60)
        print("✅ Setup complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Create a .env file with your OPENAI_API_KEY")
        print("2. Add .docx files to the data/ folder")
        print("3. Run: python example_usage.py")
    else:
        print("\n" + "=" * 60)
        print("❌ Setup incomplete. Please fix the errors above.")
        print("=" * 60)

if __name__ == "__main__":
    main()

