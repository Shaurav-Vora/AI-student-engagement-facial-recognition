# AI Student Engagement Facial Recognition

This project uses computer vision to track and analyze student engagement levels in real-time through facial expression and gaze detection. The application features a powerful Python (Flask) backend AI pipeline and a modern, responsive React + Tailwind CSS frontend dashboard.

## 🌟 Features

- **Real-time Facial Detection & Recognition:** Identifies registered students in the camera feed using the YuNet face detection model.
- **Emotion & Engagement Analysis:** Analyzes student facial expressions to determine engagement levels (e.g., highly engaged, moderately engaged, distracted) using an Emotion Analyzer.
- **Live Dashboard:** A React frontend that displays the live video feed (MJPEG) and a dynamically updating list of students with their current engagement states.
- **Easy Registration:** Register new students directly from the frontend UI.

## 📂 Project Structure

The project is divided into two main components:

- `backend/` — The Flask application and AI pipeline.
  - `app.py`: The main Flask server that provides the API and MJPEG video feed.
  - `pipeline.py`: The core AI pipeline integrating face detection, recognition, and emotion analysis.
  - `src/`: Contains core data structures, model interfaces, and processors.
- `frontend/` — The React + Vite frontend application.
  - `src/App.jsx`: The main React component rendering the dashboard.
  - `src/components/`: Reusable UI components.

## 🚀 Installation & Setup

Follow these steps to get the environment running on your local machine.

### 1. Clone the Repository

Open your terminal and run:
```bash
git clone https://github.com/Shaurav-Vora/AI-student-engagement-facial-recognition.git
cd AI-student-engagement-facial-recognition
```

### 2. Backend Setup (AI Pipeline)

Navigate to the backend directory, create a virtual environment, and install the dependencies.

**Windows:**
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Frontend Setup (Dashboard)

Open a **new terminal tab/window**, navigate to the frontend directory, and install the Node modules.

```bash
cd frontend
npm install
```

## 🏃 Workflow & Execution

To run the full application, you need to start both the backend and frontend servers simultaneously.

### Step 1: Start the Backend server

In your backend terminal (with the virtual environment activated), run:
```bash
python app.py
```
*The backend will start processing the webcam feed and host the API on `http://localhost:5000`.*

### Step 2: Start the Frontend development server

In your frontend terminal, run:
```bash
npm run dev
```
*This will start the Vite development server. Open the provided local URL (usually `http://localhost:5173`) in your browser.*

### Step 3: Register Students and Monitor Engagement

1. **Dashboard:** You will see the live webcam feed and a list of detected students.
2. **Registration:** If a face is detected but unrecognized, click the "Register Student" button, enter their name, and submit. The system will save their facial encoding.
3. **Monitoring:** The dashboard will update in real-time, showing each registered student's current emotion and engagement status derived by the AI models.

## ⚠️ Important Notes

- **Camera Access:** Ensure no other application is using your webcam, as the backend needs exclusive access to it (`cv2.VideoCapture(0)`).
- **Handling Large Files:** The backend `venv/` contains large binary files (> 100MB) such as `torch` and `tensorflow`. Do **NOT** commit the `venv/` or `frontend/node_modules/` folders to GitHub. They are ignored by default via `.gitignore`.
