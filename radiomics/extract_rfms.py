import os
import numpy as np
import cv2
import SimpleITK as sitk
from radiomics import featureextractor


def sliding_window_radiomics(image, window_size, stride, params):
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    # Pad the image to handle edge cases
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)

    # Initialize feature maps
    feature_maps = {}

    # Sliding window loop
    rows, cols = image.shape
    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            # Extract patch
            patch_image = padded_image[i:i + window_size, j:j + window_size]

            # Skip uniform or empty patches
            if np.min(patch_image) == np.max(patch_image):
                continue

            # Convert patch to SimpleITK image
            sitk_patch_image = sitk.GetImageFromArray(patch_image)

            # Create a dummy mask
            patch_mask = np.ones_like(patch_image, dtype=np.uint8)
            sitk_patch_mask = sitk.GetImageFromArray(patch_mask)

            # Compute radiomics features
            try:
                features = extractor.execute(sitk_patch_image, sitk_patch_mask)
            except ValueError as e:
                print(f"Skipping patch due to error: {e}")
                continue

            # Store features
            for feature_name, feature_value in features.items():
                try:
                    float(feature_value)
                except:
                    continue

                if feature_name not in feature_maps:
                    feature_maps[feature_name] = np.zeros(image.shape, dtype=np.float64)
                feature_maps[feature_name][i, j] = min(float(feature_value), np.finfo(np.float64).max)

    return feature_maps


if __name__ == "__main__":
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for noise_level in noise_levels:
        image_dir = f"/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_{noise_level}/test"
        output_dir = f"/home/phoebe0723/rop2/speckle_noise_radiomics_feature_maps/add_speckle_noise_rfm_{noise_level}/test"

        os.makedirs(output_dir, exist_ok=True)

        for class_label in ["benign"]:
            class_input_dir = os.path.join(image_dir, class_label)
            class_output_dir = os.path.join(output_dir, class_label)
            os.makedirs(class_output_dir, exist_ok=True)

            # Loop through images
            for image_file in os.listdir(class_input_dir):
                if image_file.endswith("_image.jpg"):
                    # Paths for image
                    image_path = os.path.join(class_input_dir, image_file)

                    # Define output directories for the current image
                    base_name = os.path.splitext(image_file)[0].replace("_image", "")
                    image_output_dir = os.path.join(class_output_dir, base_name)
                    visualization_dir = os.path.join(image_output_dir, "visualization")
                    numpy_dir = os.path.join(image_output_dir, "numpy")
                    feature1_dir = os.path.join(numpy_dir, "original_glcm_SumEntropy.npy")
                    feature2_dir = os.path.join(numpy_dir, "original_glrlm_RunLengthNonUniformity.npy")

                    # Check if the image has already been processed
                    if os.path.exists(feature1_dir) and os.path.exists(feature2_dir):
                        print(f"Skipping already processed image: {image_path}")
                        continue

                    print(f"Processing image: {image_path}")

                    if not os.path.exists(image_path):
                        print(f"Image file not found: {image_path}. Skipping...")
                        continue

                    # Read and resize the image
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

                    # Define parameters
                    window_size = 5
                    stride = 1
                    params = "./params.yaml"

                    # Generate feature maps
                    feature_maps = sliding_window_radiomics(image, window_size, stride, params)

                    # Create output directories
                    os.makedirs(visualization_dir, exist_ok=True)
                    os.makedirs(numpy_dir, exist_ok=True)

                    # Save feature maps
                    for feature_name, feature_map in feature_maps.items():
                        npy_save_path = os.path.join(numpy_dir, f"{feature_name}.npy")
                        np.save(npy_save_path, feature_map)

                        # Save as .jpg for visualization
                        normalized_map = cv2.normalize(feature_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                        normalized_map = normalized_map.astype(np.uint8)
                        jpg_save_path = os.path.join(visualization_dir, f"{feature_name}.jpg")
                        cv2.imwrite(jpg_save_path, normalized_map)

                        print(f"Saved {feature_name} as .npy ({npy_save_path})")