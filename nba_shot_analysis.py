# --- NBA Shot Analyzer ---
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import tempfile

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------- Angle Calculation ----------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ---------- Shot Detection Logic ----------
def detect_shots(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    shots, frame_count = [], 0
    shot_id, detecting_shot, release_detected = 0, False, False
    initial_hip_y = None
    max_jump = 0
    elbow_angles, wrist_angles, release_angles, confidences = [], [], [], []
    snapshot = {}
    last_release_frame = -50

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            confidence = np.mean([lm[i].visibility for i in range(33)])

            r_shoulder = [lm[12].x, lm[12].y]
            r_elbow = [lm[14].x, lm[14].y]
            r_wrist = [lm[16].x, lm[16].y]
            r_hip = [lm[24].x, lm[24].y]

            elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            wrist_angle = calculate_angle(r_elbow, r_wrist, [r_wrist[0], r_wrist[1]-0.1])
            release_angle = calculate_angle([r_elbow[0], r_elbow[1]-0.1], r_elbow, r_wrist)
            shoulder_angle = calculate_angle([r_shoulder[0]-0.1, r_shoulder[1]], r_shoulder, r_elbow)

            if initial_hip_y is None:
                initial_hip_y = r_hip[1]
            jump_height = max(0, initial_hip_y - r_hip[1])
            max_jump = max(max_jump, jump_height)

            if elbow_angle < 100 and not detecting_shot:
                detecting_shot = True
                elbow_angles, wrist_angles, release_angles, confidences = [], [], [], []
                max_jump = 0
                snapshot = {}

            if detecting_shot:
                elbow_angles.append(elbow_angle)
                wrist_angles.append(wrist_angle)
                release_angles.append(release_angle)
                confidences.append(confidence)

                if (r_wrist[1] < r_shoulder[1]) and elbow_angle > 140 and not release_detected:
                    release_detected = True
                    release_frame = frame_count
                    snapshot = {
                        "Elbow Angle": round(elbow_angle, 1),
                        "Wrist Angle": round(wrist_angle, 1),
                        "Shoulder Angle": round(shoulder_angle, 1),
                        "Pose Confidence": round(confidence, 2)
                    }

            if detecting_shot and release_detected and elbow_angle < 100:
                if frame_count - last_release_frame > 10:
                    shot_id += 1
                    avg_conf = np.mean(confidences)
                    avg_elbow = round(np.mean(elbow_angles), 1)
                    avg_wrist = round(np.mean(wrist_angles), 1)
                    avg_release = round(np.mean(release_angles), 1)
                    jump_cm = round(max_jump * 100, 1)
                    release_height_cm = round((1 - r_wrist[1]) * 100, 1)

                    # --- Intelligent classification ---
                    result = "MISS"
                    miss_type = "Short Right"
                    if avg_release > 50 and avg_conf > 0.8 and avg_elbow > 85:
                        result = "MAKE"
                        miss_type = ""

                    # Suggested drills
                    drills = "Form Shooting - Elbow Control, Wrist Flick Reps" if result == "MISS" else "Keep up the good form!"

                    shots.append({
                        "Shot ID": shot_id,
                        "Result": result,
                        "Release Frame": release_frame,
                        "Confidence": round(avg_conf, 2),
                        "Release Angle": avg_release,
                        "Elbow Angle": avg_elbow,
                        "Wrist Flick": avg_wrist,
                        "Jump Height (cm)": jump_cm,
                        "Release Height (cm)": release_height_cm,
                        "Miss Type": miss_type,
                        "Suggested Drills": drills,
                        "Pose Snapshot": snapshot
                    })
                    last_release_frame = frame_count

                detecting_shot = False
                release_detected = False
                elbow_angles, wrist_angles, release_angles, confidences = [], [], [], []
                snapshot = {}

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()
    return shots, output_video_path

# ---------- PDF Report ----------
def generate_pdf(shots):
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_pdf.name, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "NBA Shot Analysis Report")

    y = 720
    c.setFont("Helvetica", 12)
    for shot in shots:
        c.drawString(50, y, f"Shot ID: {shot['Shot ID']} | Result: {shot['Result']}")
        y -= 20
        for k, v in shot.items():
            if k not in ["Shot ID", "Result"]:
                if isinstance(v, dict):  # Pose Snapshot
                    c.drawString(60, y, "Pose Snapshot:")
                    y -= 15
                    for pk, pv in v.items():
                        c.drawString(70, y, f"{pk}: {pv}")
                        y -= 12
                else:
                    c.drawString(60, y, f"{k}: {v}")
                    y -= 15
        y -= 20

    c.save()
    return tmp_pdf.name

# ---------- STREAMLIT UI ----------
st.title("🏀 NBA Shot Analysis - Enhanced Detection")

uploaded_video = st.file_uploader("Upload your shooting video", type=["mp4", "mov", "avi"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    st.video(video_path)

    if st.button("Analyze Shots"):
        with st.spinner("Analyzing video..."):
            processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            shots, processed_path = detect_shots(video_path, processed_video_path)

        if not shots:
            st.warning("No shots detected. Ensure the shot motion is clear and visible.")
        else:
            df = pd.DataFrame(shots)
            st.subheader("Per-Shot Breakdown")
            st.dataframe(df)

            csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df.to_csv(csv_file.name, index=False)
            with open(csv_file.name, "rb") as f:
                st.download_button("Download CSV Data", f, file_name="shot_metrics.csv")

            pdf_file = generate_pdf(shots)
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF Report", f, file_name="shot_analysis_report.pdf")

            with open(processed_path, "rb") as f:
                st.download_button("Download Processed Video", f, file_name="pose_analysis.mp4")
