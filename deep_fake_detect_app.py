import argparse
import os
from tabulate import tabulate
from data_utils.face_detection import *
from deep_fake_detect.utils import *
from deep_fake_detect.DeepFakeDetectModel import *
import torchvision
from data_utils.datasets import *
import warnings
import multiprocessing
import torch
import sys

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def predict_deepfake(input_videofile, debug=False, verbose=False):
    num_workers = multiprocessing.cpu_count() - 2
    model_params = {
        'batch_size': 16,  # Reduced batch size to 16
        'imsize': 160,     # Reduced image size to 160
        'encoder_name': 'tf_efficientnet_b0_ns'
    }

    prob_threshold_fake = 0.5
    fake_fraction = 0.5

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    vid = os.path.basename(input_videofile)[:-4]
    output_path = os.path.join("output", vid)
    os.makedirs(output_path, exist_ok=True)

    if verbose:
        print(f'Extracting faces from the video')
    extract_landmarks_from_video(input_videofile, output_path, overwrite=True)
    
    plain_faces_data_path = os.path.join(output_path, "plain_frames")
    os.makedirs(plain_faces_data_path, exist_ok=True)
    crop_faces_from_video(input_videofile, output_path, plain_faces_data_path, overwrite=True)

    if verbose:
        print(f'Generating MRIs of the faces')
    mri_output = os.path.join(output_path, 'mri')
    predict_mri_using_MRI_GAN(plain_faces_data_path, mri_output, vid, 256, overwrite=True)

    model_path = 'assets/weights/MRI_GAN_weights.chkpt'
    frames_path = mri_output

    if verbose:
        print(f'Detecting DeepFakes using MRI method')
    model = DeepFakeDetectModel(frame_dim=model_params['imsize'], encoder_name=model_params['encoder_name'])
    if verbose:
        print(f'Loading model weights {model_path}')
    check_point_dict = torch.load(model_path)
    model.load_state_dict(check_point_dict['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((model_params['imsize'], model_params['imsize'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    data_path = os.path.join(frames_path, vid)
    test_dataset = SimpleImageFolder(root=data_path, transforms_=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                             pin_memory=True)

    if len(test_loader) == 0:
        print('Cannot extract images. Dataloaders empty')
        return None, None, None

    probabilities = []
    all_filenames = []
    all_predicted_labels = []

    with torch.no_grad():
        for batch_id, samples in enumerate(test_loader):
            frames = samples[0].to(device)

            # Enable mixed precision
            with torch.cuda.amp.autocast():
                output = model(frames)

            predicted = get_predictions(output).to('cpu').detach().numpy()
            class_probability = get_probability(output).to('cpu').detach().numpy().astype(float)

            if len(predicted) > 1:
                all_predicted_labels.extend(predicted.squeeze())
                probabilities.extend(class_probability.squeeze())
                all_filenames.extend(samples[1])
            else:
                all_predicted_labels.append(predicted.squeeze())
                probabilities.append(class_probability.squeeze())
                all_filenames.append(samples[1])

            # Manually clear cache to free up memory
            torch.cuda.empty_cache()

        total_number_frames = len(probabilities)
        probabilities = np.array(probabilities)

        fake_frames_high_prob = probabilities[probabilities >= prob_threshold_fake]
        number_fake_frames = len(fake_frames_high_prob)
        fake_prob = round(sum(fake_frames_high_prob) / number_fake_frames, 4) if number_fake_frames > 0 else 0

        real_frames_high_prob = probabilities[probabilities < prob_threshold_fake]
        number_real_frames = len(real_frames_high_prob)
        real_prob = round(1 - sum(real_frames_high_prob) / number_real_frames, 4) if number_real_frames > 0 else 0
        pred = pred_strategy(number_fake_frames, number_real_frames, total_number_frames, fake_fraction=fake_fraction)

        return fake_prob, real_prob, pred

def individual_test():
    print_line()
    debug = False
    verbose = True
    fake_prob, real_prob, pred = predict_deepfake(args.input_videofile, debug=debug, verbose=verbose)
    if pred is None:
        print_red('Failed to detect DeepFakes')
        return

    label = "REAL" if pred == 0 else "DEEP-FAKE"

    probability = real_prob if pred == 0 else fake_prob
    probability = round(probability * 100, 4)
    print_line()
    if pred == 0:
        print_green(f'The video is {label}, probability={probability}%')
    else:
        print_red(f'The video is {label}, probability={probability}%')
    print_line()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DeepFakes detection App. Provide input_videofile as argument')
    parser.add_argument('--input_videofile', action='store', help='Input video file', required=True)
    args = parser.parse_args()

    if os.path.isfile(args.input_videofile):
        individual_test()
    else:
        print(f'Input file not found ({args.input_videofile})')
