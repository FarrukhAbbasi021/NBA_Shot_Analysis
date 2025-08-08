
# 🏀 NBA Shot Analyzer - Prototype

Welcome to your personal basketball shooting coach! This tool analyzes your shooting form using AI,
giving you detailed metrics like elbow angle, wrist flick, jump height, and even whether your shot 
was a MAKE or MISS.

This guide will help you set up and run the project from a ZIP file—no prior experience required.

---

## 📁 What’s Inside This Project

- `nba_shot_analysis.py` — The main app file to run.
- `requirements.txt` — All the Python packages you need.
- `README.md` — This file with step-by-step guidance.

---

## 🐍 Step 1: Install Python

If you don’t have Python installed:

👉 [Download Python 3.10+ here](https://www.python.org/downloads/)  
Make sure to check the box **“Add Python to PATH”** during installation!

---

## 🖥️ Step 2: Set Up the Environment

1. **Unzip the downloaded project folder** to any location on your computer.

2. **Open a terminal/command prompt** in the project folder.

3. Create a virtual environment (recommended):

### On Windows:
```
python -m venv venv
venv\Scripts\activate
```

### On Mac/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 Step 3: Install Required Libraries

Install everything the app needs by running:

```
pip install -r requirements.txt
```

---

## 🚀 Step 4: Run the Shot Analyzer

Launch the application with:

```
streamlit run nba_shot_analysis.py
```

Your browser will open the app automatically.

---

## 🎥 How to Use the App

1. Upload a basketball shooting video (`.mp4`, `.mov`, `.avi`).  
   ➤ Make sure the shooter and ball are clearly visible.

2. Click **“Analyze Shots”**.

3. View the metrics per shot: angles, jump height, release quality.

4. Download:
   - 📊 **CSV file** for data
   - 📄 **PDF report**
   - 🎞️ **Video with pose landmarks**

---

## ✅ Tips for Best Results

- Use clear, well-lit videos from the side or 45-degree angle.
- Make sure the shooter’s full body is visible during the shot.

---

## 💬 Support

This prototype was designed to be beginner-friendly. If you face any issues, feel free to contact the developer.

Enjoy your personalized NBA-level shooting analysis! 🏀📈


Regards,
[Precisionmlai]
