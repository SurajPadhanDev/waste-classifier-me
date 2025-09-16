#!/usr/bin/env python3
"""
Smart Waste Classifier Setup Script
Automatically installs all required dependencies
"""

import subprocess
import sys
import os
import platform

def run_command(command):
    """Run a command and return True if successful"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {command}")
            return True
        else:
            print(f"❌ {command}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def check_python():
    """Check if Python 3.8+ is installed"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False

def install_requirements():
    """Install required packages"""
    print("\n🔄 Installing required packages...")
    
    packages = [
        "streamlit>=1.28.0",
        "tensorflow>=2.13.0", 
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        if not run_command(f"{sys.executable} -m pip install {package}"):
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    os.makedirs(".streamlit", exist_ok=True)
    os.makedirs("utils", exist_ok=True)
    print("✅ Directories created")

def check_model_file():
    """Check if model file exists"""
    if os.path.exists("best_mobilenetv2_model.keras"):
        print("✅ Model file found")
        return True
    else:
        print("❌ Model file 'best_mobilenetv2_model.keras' not found!")
        print("Please place your trained model file in the same directory as this script.")
        return False

def main():
    print("🚀 BOLTINNOVATOR Smart Waste Classifier Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Failed to install some packages. Please check the error messages above.")
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        print("\n⚠️  Setup completed but model file is missing.")
        print("The app will not work without the model file.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n▶️  To run the app:")
    print("   python run_app.py")
    print("   or")  
    print("   streamlit run app.py --server.port 5000")
    print("\n🌐 The app will be available at: http://localhost:5000")

if __name__ == "__main__":
    main()