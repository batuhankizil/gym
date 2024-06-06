import cv2
import mediapipe as mp
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner
import tkinter as tk
from tkinter import messagebox

# Egzersiz seçimi için global değişken
selected_exercise = "squat"

def start_video():
    global selected_exercise
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    process_frame = ProcessFrame(thresholds=get_thresholds_beginner(), flip_frame=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kare okunamadı!")
            break

        if selected_exercise == "squat":
            processed_frame, play_sound = process_frame.process(frame, pose, exercise="squat")
        elif selected_exercise == "elbow-plank":
            processed_frame, play_sound = process_frame.process(frame, pose, exercise="elbow-plank")
        elif selected_exercise == "punches":
            processed_frame, play_sound = process_frame.process(frame, pose, exercise="punches")
        elif selected_exercise == "leg-curls":
            processed_frame, play_sound = process_frame.process(frame, pose, exercise="leg-curls")
        elif selected_exercise == "high-knees":
            processed_frame, play_sound = process_frame.process(frame, pose, exercise="high_knees")

        cv2.imshow('Egzersiz Algılama', processed_frame)

        if play_sound:
            print("Geri bildirim sesi:", play_sound)

        key = cv2.waitKey(1)

        if key == ord('q') or key == 27:  # 'q' tuşuna veya ESC tuşuna basılınca

            break

    cap.release()
    cv2.destroyAllWindows()

def select_squat():
    global selected_exercise
    selected_exercise = "squat"
    messagebox.showinfo("Seçim", "Squat seçildi! Hareketinizi yapın.")
    start_video()

def select_elbow_plank():
    global selected_exercise
    selected_exercise = "elbow-plank"
    messagebox.showinfo("Seçim", "Elbow-plank seçildi! Hareketinizi yapın.")
    start_video()

def select_punches():
    global selected_exercise
    selected_exercise = "punches"
    messagebox.showinfo("Seçim", "Punches seçildi! Hareketinizi yapın.")
    start_video()

def select_leg_curls():
    global selected_exercise
    selected_exercise = "leg-curls"
    messagebox.showinfo("Seçim", "Leg Curls seçildi! Hareketinizi yapın.")
    start_video()

def select_high_knees():
    global selected_exercise
    selected_exercise = "high-knees"
    messagebox.showinfo("Seçim", "High Knees seçildi! Hareketinizi yapın.")
    start_video()

def main():
    # Tkinter arayüzünü oluştur
    root = tk.Tk()
    root.title("Egzersiz Seçimi")

    label = tk.Label(root, text="Lütfen yapmak istediğiniz egzersizi seçin")
    label.pack(pady=10)

    squat_button = tk.Button(root, text="Squat", command=select_squat)
    squat_button.pack(pady=5)

    elbow_plank_button = tk.Button(root, text="Elbow Plank", command=select_elbow_plank)
    elbow_plank_button.pack(pady=5)

    punches_button = tk.Button(root, text="Punches", command=select_punches)
    punches_button.pack(pady=5)

    leg_curls_button = tk.Button(root, text="Leg Curls", command=select_leg_curls)
    leg_curls_button.pack(pady=5)

    high_knees_button = tk.Button(root, text="High Knees", command=select_high_knees)
    high_knees_button.pack(pady=5)

    exit_button = tk.Button(root, text="Çıkış", command=root.destroy)
    exit_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
