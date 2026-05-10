#!/usr/bin/env python3
"""
Coronary AI Detection System - Launch Script
Run this file to start the application
"""
import os
import sys
import subprocess

def check_dependencies():
    """Check and install required packages"""
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'werkzeug'
    ]
    
    print("🔍 Checking dependencies...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'static/css',
        'static/uploads/ecg',
        'static/uploads/doctor_uploads',
        'templates'
    ]
    
    print("\n📁 Setting up directories...")
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Created: {dir_path}")
        else:
            print(f"✅ Already exists: {dir_path}")

def create_minimal_app():
    """Create a minimal working app.py if it doesn't exist"""
    if not os.path.exists('app.py'):
        print("\n📝 Creating minimal app.py...")
        with open('app.py', 'w') as f:
            f.write('''
from flask import Flask, render_template, request, session, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = 'coronary-ai-system-secret-key-2024'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient/login')
def patient_login():
    return render_template('patient_login.html')

@app.route('/patient/register')
def patient_register():
    return render_template('patient_register.html')

@app.route('/doctor/login')
def doctor_login():
    return render_template('doctor_login.html')

@app.route('/doctor/upload')
def doctor_upload():
    return render_template('doctor_upload.html')

if __name__ == '__main__':
    print("\\n🚀 Starting Coronary AI System...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, port=5000)
''')
        print("✅ app.py created")

def main():
    print("="*60)
    print("🫀 CORONARY AI DETECTION SYSTEM - SETUP")
    print("="*60)
    
    # Check and install dependencies
    check_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Create minimal app if needed
    create_minimal_app()
    
    print("\n" + "="*60)
    print("🎯 SETUP COMPLETE!")
    print("="*60)
    print("\nTo start the application:")
    print("1. Run: python app.py")
    print("2. Open browser and go to: http://localhost:5000")
    print("3. Don't open HTML files directly!")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()