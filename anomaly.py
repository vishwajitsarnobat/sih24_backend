import cv2
import numpy as np
import os
import math
import dlib
from skimage.metrics import structural_similarity as ssim

def exponential_scale(x, min_val, max_val):
    if x < min_val:
        return 0
    if x > max_val:
        return 0.99
    normalized_x = (x - min_val) / (max_val - min_val)
    scaled_value = 0.99 * (1 - math.exp(-normalized_x))
    return scaled_value

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) * 2 + (point1[1] - point2[1]) * 2)

def calculate_misalignment_score(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    nose_tip = landmarks[30]
    left_eyebrow = landmarks[17:22]
    right_eyebrow = landmarks[23:27]
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]
    chin = landmarks[8]

    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    eye_distance = euclidean_distance(left_eye_center, right_eye_center)

    nose_to_left_eye = euclidean_distance(nose_tip, left_eye_center)
    nose_to_right_eye = euclidean_distance(nose_tip, right_eye_center)
    nose_alignment_score = abs(nose_to_left_eye - nose_to_right_eye)

    left_eyebrow_center = np.mean(left_eyebrow, axis=0)[1]
    right_eyebrow_center = np.mean(right_eyebrow, axis=0)[1]
    eyebrow_alignment_score = abs(left_eyebrow_center - right_eyebrow_center)

    mouth_center = np.mean([left_mouth, right_mouth], axis=0)[0]
    mouth_alignment_score = abs(mouth_center - (left_eye_center[0] + right_eye_center[0]) / 2)

    chin_alignment_score = abs(chin[1] - nose_tip[1])

    eye_distance_score = min(eye_distance / 100, 1)
    nose_alignment_score = min(nose_alignment_score / 50, 1)
    eyebrow_alignment_score = min(eyebrow_alignment_score / 10, 1)
    mouth_alignment_score = min(mouth_alignment_score / 50, 1)
    chin_alignment_score = min(chin_alignment_score / 50, 1)

    misalignment_score = (0.2 * eye_distance_score +
                          0.2 * nose_alignment_score +
                          0.2 * eyebrow_alignment_score +
                          0.2 * mouth_alignment_score +
                          0.2 * chin_alignment_score)
    
    return misalignment_score

def scale_scores(real_score, fake_score):
    difference = abs(real_score - fake_score)
    scaled_value = 0.99 * (1 - math.exp(-difference))
    return scaled_value

def calculate_brightness(image):
    return np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

def calculate_histogram_difference(face_region, surrounding_region):
    face_hist = cv2.calcHist([face_region], [0], None, [256], [0, 256])
    surrounding_hist = cv2.calcHist([surrounding_region], [0], None, [256], [0, 256])
    return cv2.compareHist(face_hist, surrounding_hist, cv2.HISTCMP_CORREL)

def calculate_sharpness(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges)

def calculate_ssim(face_region, frame):
    face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (frame_gray.shape[1], frame_gray.shape[0]))
    score, _ = ssim(face_gray, frame_gray, full=True)
    return score

def detect_lighting_and_shadow_mismatches(frame):
    faces = detect_face(frame)
    if len(faces) == 0:
        return 0
    scores = []
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        surrounding_region = frame[max(0, y-50):min(frame.shape[0], y+h+50),
                                   max(0, x-50):min(frame.shape[1], x+w+50)]
        face_brightness = calculate_brightness(face_region)
        surrounding_brightness = calculate_brightness(surrounding_region)
        brightness_diff = abs(face_brightness - surrounding_brightness)
        hist_diff = calculate_histogram_difference(face_region, surrounding_region)
        anomaly_score = (brightness_diff / 100) + (1 - hist_diff)
        scores.append(anomaly_score)
    return np.mean(scores)

def detect_resolution_artifacts(frame):
    faces = detect_face(frame)
    if len(faces) == 0:
        return 0
    scores = []
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        face_sharpness = calculate_sharpness(face_region)
        ssim_score = calculate_ssim(face_region, frame)
        resolution_score = 1 - (abs(face_sharpness - ssim_score) / 100)
        scores.append(resolution_score)
    return np.mean(scores)

def get_item(video_path):
    analysis_results = {
        "misalignment_score": None,
        "lighting_score": None,
        "resolution_score": None,
        "fake_score": None
    }

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)

    video_capture.release()

    if len(frames) > 0:
        frame = frames[0]

        faces = detect_face(frame)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                detections = detector(gray)
                if len(detections) > 0:
                    shape = predictor(gray, detections[0])
                    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                    misalignment_score = calculate_misalignment_score(landmarks)
                    lighting_score = detect_lighting_and_shadow_mismatches(frame)
                    resolution_score = detect_resolution_artifacts(frame)
                    fake_score = exponential_scale(misalignment_score, 0, 1)

                    # Convert results to strings before storing
                    analysis_results["misalignment_score"] = str(misalignment_score)
                    analysis_results["lighting_score"] = str(lighting_score)
                    analysis_results["resolution_score"] = str(resolution_score)
                    analysis_results["fake_score"] = str(fake_score)

    return analysis_results

if __name__ == '_main_':
    print(get_item('/home/vishwajit/Workspace/SIH/final_test_videos/A12348.mp4'))