#!/usr/bin/env python3
"""
Smart Waste Classifier Runner
Easy launcher for the waste classification app
"""

import subprocess
import sys
import os
import webbrowser
import time
from threading import Timer

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = ['streamlit', 'tensorflow', 'cv2', 'numpy', 'PIL']
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def open_browser():
    """Open browser after a delay"""
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open('http://localhost:5000')
    except:
        pass

def main():
    print("🚀 BOLTINNOVATOR Smart Waste Classifier")
    print("=" * 45)
    
    # Check if model file exists
    if not os.path.exists("best_mobilenetv2_model.keras"):
        print("❌ Model file 'best_mobilenetv2_model.keras' not found!")
        print("Please make sure the model file is in the same directory as this script.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Please run 'python setup.py' first to install dependencies.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print("✅ All requirements satisfied!")
    print("🔄 Starting Smart Waste Classifier...")
    print("\n🌐 Opening http://localhost:5000 in your browser...")
    print("\n📱 Use Ctrl+C to stop the application")
    print("-" * 45)
    
    # Open browser automatically
    timer = Timer(3.0, open_browser)
    timer.daemon = True
    timer.start()
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port=5000',
            '--server.headless=true',
            '--browser.gatherUsageStats=false'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Smart Waste Classifier stopped. Thank you for using BOLTINNOVATOR!")
    except Exception as e:
        print(f"\n❌ Error running the app: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()