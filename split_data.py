import os
import random
from shutil import copy2, rmtree
from pathlib import Path
import csv
import cv2

# Paths to original data and output
data_path = "/home/phoebe0723/rop/benign_malignant_data/cropped"
output_path = "/home/phoebe0723/rop/all_data"

# CSV file path
csv_file = os.path.join(output_path, "data_labels.csv")

# Split ratios
train_val_ratio = 0.9
train_ratio_within_train_val = 0.8
random_seed = 42

# Ensure output directory is clean
def clean_output_directory(output_path):
    if Path(output_path).exists():
        print(f"Cleaning up existing directory: {output_path}")
        rmtree(output_path)  # Remove the existing directory and its contents

    Path(output_path).mkdir(parents=True, exist_ok=True)  # Recreate the output directory
    print(f"Created new output directory: {output_path}")

# Create output directories
def create_directories(output_path, split_dirs, categories):
    for split in split_dirs:
        for category in categories:
            dir_path = Path(f"{output_path}/{split}/{category}")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")

# Function to check if an image is completely black
def is_black_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    return image is not None and cv2.countNonZero(image) == 0  # True if all pixels are zero

# Function to split data
def split_patients(data_path, output_path, train_val_ratio, train_ratio_within_train_val, csv_file, seed):
    random.seed(seed)
    labels = []

    categories = ["benign", "malignant"]
    for category in categories:
        category_path = os.path.join(data_path, category)
        patient_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

        # Shuffle and split patient folders
        random.shuffle(patient_folders)
        split_index = int(len(patient_folders) * train_val_ratio)

        train_val_patients = patient_folders[:split_index]
        test_patients = patient_folders[split_index:]

        print(f"Category: {category}")
        print(f"Train/Validation Patients: {len(train_val_patients)}, Test Patients: {len(test_patients)}")

        # Process test patients: Rename and copy all images to the test set
        for patient in test_patients:
            patient_path = os.path.join(category_path, patient)
            rename_and_copy_images(patient_path, os.path.join(output_path, "test", category), patient, category, labels)

        # Process train+validation patients: Mix images for splitting
        train_val_images = []
        for patient in train_val_patients:
            patient_path = os.path.join(category_path, patient)
            train_val_images.extend(
                [(os.path.join(patient_path, f), patient) for f in os.listdir(patient_path) if f.startswith("image_")]
            )

        # Shuffle train+validation images
        random.shuffle(train_val_images)
        train_size = int(len(train_val_images) * train_ratio_within_train_val)

        train_images = train_val_images[:train_size]
        validation_images = train_val_images[train_size:]

        # Copy train images
        for image_path, patient in train_images:
            rename_and_copy_image(image_path, os.path.join(output_path, "train", category), patient, category, labels)

        # Copy validation images
        for image_path, patient in validation_images:
            rename_and_copy_image(image_path, os.path.join(output_path, "validation", category), patient, category, labels)

    # Write labels to CSV
    write_csv(labels, csv_file)

# Helper function to rename and copy images
def rename_and_copy_images(src_path, dst_path, patient, category, labels):
    for i, file in enumerate(sorted(os.listdir(src_path))):
        if file.startswith("image_") and os.path.isfile(os.path.join(src_path, file)):
            src_file = os.path.join(src_path, file)

            # Skip black images
            if is_black_image(src_file):
                print(f"Skipping black image: {src_file}")
                continue

            new_name = f"{patient}_{i}_image.jpg"
            dst_file = os.path.join(dst_path, new_name)
            Path(dst_path).mkdir(parents=True, exist_ok=True)
            copy2(src_file, dst_file)
            labels.append({"image": new_name, "label": category})
            # print(f"Copied {src_file} to {dst_file}")

# Helper function to rename and copy a single image
def rename_and_copy_image(src_file, dst_folder, patient, category, labels):
    if os.path.isfile(src_file):
        base_name = os.path.basename(src_file)
        i = base_name.split("_")[-1].split(".")[0]  # Extract the image number from the file name
        new_name = f"{patient}_{i}_image.jpg"
        dst_file = os.path.join(dst_folder, new_name)
        Path(dst_folder).mkdir(parents=True, exist_ok=True)
        copy2(src_file, dst_file)
        labels.append({"image": new_name, "label": category})
        # print(f"Copied {src_file} to {dst_file}")

# Helper function to write labels to CSV
def write_csv(labels, csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows(labels)
    print(f"Labels written to {csv_file}")

# Main execution
if __name__ == "__main__":
    clean_output_directory(output_path)
    create_directories(output_path, ["train", "validation", "test"], ["benign", "malignant"])
    split_patients(data_path, output_path, train_val_ratio, train_ratio_within_train_val, csv_file, random_seed)
