# AI Student Engagement Facial Recognition

This project uses computer vision to track and analyze student engagement levels in real-time through facial expression and gaze detection.

## 🚀 Installation & Setup

Follow these steps to get the environment running on your local machine.

### 1. Clone the Repository

Open your terminal and run:
```bash
git clone https://github.com/Shaurav-Vora/AI-student-engagement-facial-recognition.git
cd AI-student-engagement-facial-recognition
```

### 2. Create Your Own Branch

Before making any changes, create a new branch to keep your work separate from `main`:
```bash
git checkout -b your-branch-name
```

> 💡 Use a descriptive name like `feature/gaze-tracking` or `fix/expression-model`.

### 3. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install the required libraries (TensorFlow, OpenCV, Torch, etc.) from the requirements file:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application

Start the facial recognition engagement tracker:
```bash
python main.py
```

## 📂 Project Structure

- `main.py` — The main execution script.
- `requirements.txt` — Contains all external library dependencies.
- `.gitignore` — Configured to exclude the `venv/` folder and large binary files (> 100MB).
- `src/` — Contains the following folders:
  - `core/` — Defines the foundational data structures (like `StudentState`) that carry information predictably throughout the entire application.
  - `interfaces/` — Establishes strict blueprint templates using the Strategy Pattern, guaranteeing that any new AI model you add will plug perfectly into the system.
  - `models/` — Houses the primary face detection algorithms (like YuNet) whose sole job is to scan the raw video frame and locate where students are.
  - `processors/` — Contains the secondary analysis engines (like the Emotion Analyzer and Face Recognizer) that take cropped faces and extract business value, such as identifying who the student is and how engaged they are.

## ⚠️ Important: Handling Large Files

GitHub has a **100MB limit** per file.

- Do **NOT** commit the `venv/` folder. It contains heavy binaries like `torch_cpu.dll` and `tensorflow.dll` which will cause your push to fail.
- If you install new packages, update the list using:
```bash
pip freeze > requirements.txt
```
