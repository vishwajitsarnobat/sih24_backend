import torch
import torch.nn as nn
import cv2
from torchvision import transforms as T
from PIL import Image as pil_image

transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed_image = transform(pil_image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0)
    return preprocessed_image

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(268203, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def predict_video_class_from_stream(video, model, device, class_labels):
    frame_predictions = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        preprocessed_frame = preprocess_image(frame).to(device)

        with torch.no_grad():
            output = model(preprocessed_frame)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        frame_predictions.append((predicted_class.item(), confidence.item()))

    class_votes = {}
    for pred_class, _ in frame_predictions:
        class_votes[pred_class] = class_votes.get(pred_class, 0) + 1

    final_class_idx = max(class_votes, key=class_votes.get)
    final_confidence = sum(conf for cls, conf in frame_predictions if cls == final_class_idx) / class_votes[final_class_idx]

    final_class_name = class_labels.get(final_class_idx, "Unknown Class")

    result = {
        'source_deepfake': final_class_name,
        'source_deepfake_confidence': final_confidence
    }

    return result

def get_item(video_path):
    model_path = "models/source.pth"

    class_labels = {
        0: 'Gen-2_CNN_Based_Autoencoders(DeepFake-maker)',
        1: 'Gen-2_Face_Swapping_Autoencoders',
        2: 'Gen-2_Face_Swapping_Landmark_Extraction',
        3: 'Gen-2_Patch_Based_GAN_Loss(GAN-based)',
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    video = cv2.VideoCapture(video_path)
    result = predict_video_class_from_stream(video, model, device, class_labels)
    video.release()

    return result

if __name__ == "__main__":
    video_path = "/home/vishwajit/Workspace/SIH/final_test_videos/A12348.mp4"
    print(get_item(video_path))