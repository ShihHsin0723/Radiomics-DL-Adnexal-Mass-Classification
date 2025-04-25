import os
import cv2
import numpy as np
import shutil

# Define source directories
image_dirs = [
    "/home/phoebe0723/rop2/segmented_data/test/benign",
    "/home/phoebe0723/rop2/segmented_data/test/malignant"
]

def add_speckle_noise(image, std):
    # Preserve original dtype
    original_dtype = image.dtype

    # Convert to float32 for processing
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Generate speckle noise
    noise = np.random.randn(*image.shape) * std

    # Apply speckle noise (multiplicative)
    noisy_image = image + image * noise

    # Clip to valid range
    noisy_image = np.clip(noisy_image, 0, 1) * 255

    # Restore original dtype
    return noisy_image.astype(original_dtype)

def add_gaussian_noise(image, std, mean=0):
    # Preserve original dtype
    original_dtype = image.dtype

    # Normalize image to [0, 1]
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Generate normalized Gaussian noise
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)

    # Apply additive noise
    noisy_image = image + noise

    # Clip to [0, 1], then scale back to [0, 255]
    noisy_image = np.clip(noisy_image, 0, 1) * 255.0

    # Return to original dtype
    return noisy_image.astype(original_dtype)


speckle_noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
gaussian_noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

def apply_noise_batch(noise_func, noise_levels, output_root):
    for noise_level in noise_levels:
        output_base_dir = os.path.join(output_root, f"{noise_func.__name__}_data_{noise_level}")
        if os.path.exists(output_base_dir):
            shutil.rmtree(output_base_dir)

        for source_dir in image_dirs:
            relative_path = os.path.relpath(source_dir, "/home/phoebe0723/rop2/segmented_data")
            output_dir = os.path.join(output_base_dir, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            for img_name in os.listdir(source_dir):
                img_path = os.path.join(source_dir, img_name)
                output_path = os.path.join(output_dir, img_name)

                if os.path.isfile(img_path) and "image" in img_name.lower():
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Warning: Failed to load {img_path}. Skipping.")
                        continue
                    noisy_image = noise_func(image, std=noise_level)
                    cv2.imwrite(output_path, noisy_image)
                else:
                    shutil.copy2(img_path, output_path)
        print(f"Noise added: {noise_func.__name__} | Level: {noise_level} → {output_base_dir}")

apply_noise_batch(add_speckle_noise, speckle_noise_levels, "/home/phoebe0723/rop2/speckle_noise_data")
apply_noise_batch(add_gaussian_noise, gaussian_noise_levels, "/home/phoebe0723/rop2/gaussian_noise_data")
