# 🚀 BOLTINNOVATOR Smart Waste Classifier

## Quick Setup Guide for PC & Mac

### Prerequisites
- **Python 3.8 or higher** ([Download here](https://www.python.org/downloads/))
- **Webcam** (for live classification)

### Easy Installation (4 Steps)

#### Step 1: Download Files
Download and extract all files to a folder on your computer.

#### Step 2: Add Your Model
Place your trained model file `best_mobilenetv2_model.keras` in the same folder.

#### Step 3: Create Configuration
Create a `.streamlit` folder and add a `config.toml` file with:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

#### Step 4: Run Setup
Open terminal/command prompt in the folder and run:
```bash
python setup.py
```

### Running the App

**Option 1 (Easiest):**
```bash
python run_app.py
```
*This will automatically open your browser to http://localhost:5000*

**Option 2 (Manual):**
```bash
streamlit run app.py --server.port 5000
```

### Features
- **Live Camera Classification** - Real-time waste detection using your webcam
- **Image Upload** - Classify waste from photos
- **Modern UI** - Beautiful animations and effects
- **Three Categories** - Organic Waste 🌱, Hazardous Waste ☠️, Inorganic Waste ♻️

### Troubleshooting

**"Model file not found"**
- Make sure `best_mobilenetv2_model.keras` is in the same folder as the app files

**"Package not found" errors**
- Run `python setup.py` again
- Try `pip install -r requirements_standalone.txt`

**Camera not working**
- Check if another app is using your camera
- Try restarting the app

**Port 5000 already in use**
- Change port in run_app.py to 5001 or another number

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Support
For issues or questions, contact the BOLTINNOVATOR team.

---
*Built with ❤️ using Streamlit & TensorFlow*