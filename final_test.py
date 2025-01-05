import os
import requests
import json

results_folder = '/home/vishwajit/Workspace/sih/results'

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

for filename in os.listdir('/home/vishwajit/Workspace/sih'):
    file_path = os.path.join('/home/vishwajit/Workspace/sih', filename)
    
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            files = {
                'video': file,
            }
            response = requests.post('http://localhost:5000/deepfake_json', files=files)

            if response.status_code == 200:
                result_data = response.json()
                result_file_path = os.path.join(results_folder, f"{filename}_result.json")
                with open(result_file_path, 'w') as result_file:
                    json.dump(result_data, result_file, indent=4)
                print(f"Successfully uploaded {filename} and saved result")
            else:
                print(f"Failed to upload {filename}. Status code: {response.status_code}")