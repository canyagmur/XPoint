import os
import json
import csv

# Specify the base directory where your folders are located
base_path = "OUTPUTS_JOURNAL/VIS_SAR_hm3_5_10"

# Keys to extract from each .txt file
keys_to_extract = [
    "nn_map", "m_score", "threshold_keypoints", "threshold_homography",
    "h_correctness_th3", "h_correctness_th5", "h_correctness_th10",
    "repeatability_mean", "n_kp_optical", "n_kp_thermal", "n_kp_avg",
    "distance_threshold", "model_dir", "model_version", "height-width",
    "detection_th", "dataset", "reprojection_threshold", "nms",
    "two_forward_mean", "nms_mean", "interpolate_mean"
]

# Create a CSV file to store the results
#take the latest directory name in the base path
output_csv = os.path.basename(os.path.normpath(base_path)) + "_metrics.csv"

# Open the CSV file in write mode
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["filename"] + keys_to_extract)
    writer.writeheader()  # Write the header row

    # Loop through each folder and file in the base path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".txt"):  # Process only .txt files
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        # Load the Python dictionary from the file
                        data = json.load(f)

                        # Extract only the relevant keys
                        extracted_data = {key: data.get(key, None) for key in keys_to_extract}
                        extracted_data["filename"] = file  # Add the filename for reference

                        # Write the extracted data to the CSV file
                        writer.writerow(extracted_data)

                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_path}: {e}")
