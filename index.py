import os
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import anomaly
import ppg
from deep_fake_detect_app import predict_deepfake
from summary import generate_summary

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file for metadata extraction")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        aspect_ratio = f"{resolution[0]}:{resolution[1]}"
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # File size in MB

        # Extract thumbnail
        ret, frame = cap.read()
        thumbnail_base64 = None
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "video_name": os.path.basename(video_path),
            "video_length": f"{int(duration // 3600):02}:{int((duration % 3600) // 60):02}:{int(duration % 60):02}",
            "frame_count": frame_count,
            "frame_rate": int(fps),
            "resolution": f"{resolution[0]}x{resolution[1]}",
            "aspect_ratio": aspect_ratio,
            "file_size": f"{file_size:.2f}MB",
            "thumbnail": thumbnail_base64,
            "source": "Gen-3_DF-UAV"
        }
    finally:
        cap.release()

@app.route('/deepfake_json', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file in the request'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(video_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    try:
        # Extract metadata
        video_metadata = get_video_info(video_path)

        # Analyze video for anomalies
        anomalies = anomaly.get_item(video_path)

        # Analyze video for PPG signals
        ppg_data = ppg.get_item(video_path)

        # Detect deepfake using MRI method
        fake_confidence, real_confidence, is_fake = predict_deepfake(video_path, debug=False, verbose=True)
        video_metadata.update({
            "is_fake": "Fake" if is_fake else "Real",
            "fake_confidence": fake_confidence,
            "real_confidence": real_confidence
        })

        # Combine data in the required format
        response_data = {
            "video_metadata": video_metadata,
            "anomalies": {
                "fake_score": anomalies.get("fake_score"),
                "lighting_score": anomalies.get("lighting_score"),
                "misalignment_score": anomalies.get("misalignment_score"),
                "resolution_score": anomalies.get("resolution_score"),
            },
            "ppg": {
                "ppg_signals": ppg_data.get("ppg_signals"),
                "psd_signals": ppg_data.get("psd_signals"),
                "ppg_fake_confidence": f"{ppg_data.get('ppg_fake_confidence')}",
                "ppg_real_confidence": f"{ppg_data.get('ppg_real_confidence')}",
                "ppg_result": ppg_data.get("ppg_result"),
            }
        }

        # Generate summary using Gemini API
        summary_data = generate_summary(response_data, 'api_key') # api_key not included for security purposes
        response_data['summary'] = summary_data

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    app.run(debug=True)