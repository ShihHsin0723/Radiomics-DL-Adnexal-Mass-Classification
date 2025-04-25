import os
import shutil
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import Counter
from torch.optim import Adam
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Parameters setting
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_EPOCHS = 35
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 1e-2
THRESHOLD = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30), 
    transforms.RandomHorizontalFlip(0.5), 
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.ToTensor(),
])

def calculate_metrics(labels, preds, probs):
    auc = roc_auc_score(labels, probs[:, 1]) 
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 
    return auc, sensitivity, specificity

def balance_validation_set(val_dataset):
    benign_indices = [i for i, label in enumerate(val_dataset.labels) if label == 0]
    malignant_indices = [i for i, label in enumerate(val_dataset.labels) if label == 1]

    min_samples = min(len(benign_indices), len(malignant_indices))

    benign_sampled = random.sample(benign_indices, min_samples)
    malignant_sampled = random.sample(malignant_indices, min_samples)

    balanced_indices = benign_sampled + malignant_sampled
    random.shuffle(balanced_indices)

    balanced_image_paths = [val_dataset.image_paths[i] for i in balanced_indices]
    balanced_labels = [val_dataset.labels[i] for i in balanced_indices]

    return MultiChannelDataset(balanced_image_paths, val_dataset.radiomics_dir, val_dataset.feature_map1, val_dataset.feature_map2)

class MultiChannelDataset(Dataset):
    def __init__(self, image_paths, radiomics_dir, feature_map1, feature_map2, transform=None):
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.radiomics_dir = radiomics_dir
        self.feature_map1 = feature_map1
        self.feature_map2 = feature_map2
        self.transform = transform
        self.classes = ["benign", "malignant"]

        for img_path in image_paths:
            class_label = os.path.basename(os.path.dirname(img_path))  # 'benign' or 'malignant'
            label = 0 if class_label == "benign" else 1

            mask_path = img_path.replace("_image.jpg", "_mask.jpg")  # Get mask path
            img_base_name = os.path.basename(img_path).rsplit('_', 1)[0]  # Extract patient ID
            radiomics_folder = os.path.join(radiomics_dir, class_label, img_base_name, "numpy")

            path1 = os.path.join(radiomics_folder, feature_map1)
            path2 = os.path.join(radiomics_folder, feature_map2)

            if os.path.exists(mask_path) and os.path.exists(path1) and os.path.exists(path2):
                rfm1 = np.load(path1)
                rfm2 = np.load(path2)

                if np.all(rfm1 == 0) or np.all(rfm2 == 0):
                    continue

                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]

        img_name = os.path.basename(image_path).rsplit('_', 1)[0]
        class_label = os.path.basename(os.path.dirname(image_path))
        radiomics_folder = os.path.join(self.radiomics_dir, class_label, img_name, "numpy")

        # Load ultrasound image
        ultrasound_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        ultrasound_img = cv2.resize(ultrasound_img, (224, 224)) / 255.0

        # Load segmentation mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))
        mask = (mask > 0).astype(np.uint8)

        # Load radiomics feature maps
        radiomics_map1 = np.load(os.path.join(radiomics_folder, self.feature_map1))
        radiomics_map2 = np.load(os.path.join(radiomics_folder, self.feature_map2))

        # Resize radiomics maps
        radiomics_map1 = cv2.resize(radiomics_map1, (224, 224))
        radiomics_map2 = cv2.resize(radiomics_map2, (224, 224))

        # Apply mask to radiomics maps
        radiomics_map1 = np.where(mask, radiomics_map1, 0)
        radiomics_map2 = np.where(mask, radiomics_map2, 0)

        # Normalize radiomics maps
        if np.std(radiomics_map1) > 0:
            radiomics_map1 = (radiomics_map1 - np.mean(radiomics_map1)) / np.std(radiomics_map1)
        if np.std(radiomics_map2) > 0:
            radiomics_map2 = (radiomics_map2 - np.mean(radiomics_map2)) / np.std(radiomics_map2)

        # Stack the inputs: [Ultrasound Image, Radiomics Map 1, Radiomics Map 2]
        combined_input = np.stack([ultrasound_img, radiomics_map1, radiomics_map2], axis=0)

        return torch.tensor(combined_input, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.65):
        super(EfficientNetClassifier, self).__init__()
        
        # Load the pretrained EfficientNet-B0 model
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze the pretrained layers
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        # Modify the classifier to include a Dropout layer
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)

def train_and_evaluate(model, model_name, train_loader, val_loader):
    print(f"\nTraining {model_name}...\n")
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Lists to store metrics per epoch
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_aucs, val_aucs = [], []
    train_sensitivities, val_sensitivities = [], []
    train_specificities, val_specificities = [], []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds, all_probs = [], [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = (probs[:, 1] > THRESHOLD).astype(int)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            correct += (preds == labels.cpu().numpy()).sum()
            total += labels.size(0)

        # Compute metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total * 100
        train_auc, train_sensitivity, train_specificity = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))

        # Print training results
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.2f}% | Sensitivity: {train_sensitivity:.2f}% | Specificity: {train_specificity:.2f}% | AUC: {train_auc:.4f}")

        # Evaluate on Validation Set
        model.eval()
        val_loss, val_accuracy, val_auc, val_sensitivity, val_specificity = evaluate_metrics(model, val_loader, criterion, phase="Validation")

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_aucs.append(train_auc)
        train_sensitivities.append(train_sensitivity)
        train_specificities.append(train_specificity)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_aucs.append(val_auc)
        val_sensitivities.append(val_sensitivity)
        val_specificities.append(val_specificity)

    # Save model
    torch.save(model.state_dict(), f"{model_name.lower()}_model.pth")
    print(f"Model {model_name} saved.")

def evaluate_metrics(model, loader, criterion, phase="Validation"):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = (probs[:, 1] > THRESHOLD).astype(int) 

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            correct += (preds == labels.cpu().numpy()).sum()
            total += labels.size(0)

    # Compute metrics
    accuracy = correct / total * 100
    auc, sensitivity, specificity = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))

    print(f"{phase} Loss: {loss/len(loader):.4f} | Accuracy: {accuracy:.2f}% | Sensitivity: {sensitivity:.2f}% | Specificity: {specificity:.2f}% | AUC: {auc:.4f}")

    return loss / len(loader), accuracy, auc, sensitivity, specificity


if __name__ == "__main__":
    # Define selected feature maps
    feature_map1 = "original_glcm_SumEntropy.npy"
    feature_map2 = "original_glrlm_RunLengthNonUniformity.npy"

    # Define paths
    image_dir = "/home/phoebe0723/rop2/segmented_data/merged"
    radiomics_dir = "/home/phoebe0723/rop2/radiomics_feature_maps/merged"

    # Get list of image paths
    full_dataset = datasets.ImageFolder(image_dir, transform=None)
    print(f"Class Mapping: {full_dataset.class_to_idx}") 

    # Extract Patient IDs from Filenames
    image_paths = [img_path for img_path, _ in full_dataset.samples]
    patient_ids = np.array([os.path.basename(path).split("_")[2] for path in image_paths]) 

    # Get Unique Patient IDs and Split by Patient
    unique_patients = np.unique(patient_ids)  # Get unique patient IDs
    K = 5  # Number of folds
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    # Track fold results
    fold_results = []

    for fold, (train_patients, val_patients) in enumerate(kf.split(unique_patients)):
        print(f"\n----- Fold {fold+1}/{K} -----\n")

        train_patient_ids = unique_patients[train_patients] 
        val_patient_ids = unique_patients[val_patients]

        print(f"Train Patients: {len(train_patient_ids)} patients")
        print(f"Validation Patients: {len(val_patient_ids)} patients")

        # Convert Patient IDs to Image Indices
        train_indices = np.where(np.isin(patient_ids, train_patient_ids))[0]
        val_indices = np.where(np.isin(patient_ids, val_patient_ids))[0]

        # Get actual image paths
        train_image_paths = [image_paths[i] for i in train_indices]
        val_image_paths = [image_paths[i] for i in val_indices]

        # Create Train & Validation Datasets
        train_dataset = MultiChannelDataset(train_image_paths, radiomics_dir, feature_map1, feature_map2, transform=transform)
        val_dataset = MultiChannelDataset(val_image_paths, radiomics_dir, feature_map1, feature_map2)

        # Balance the training set using WeightedRandomSampler
        train_labels = np.array([train_dataset.labels[i] for i in range(len(train_dataset))])
        class_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = np.array([class_weights[label] for label in train_labels])
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        # Apply undersampling to balance the validation set
        val_dataset_balanced = balance_validation_set(val_dataset)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset_balanced, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model
        model = EfficientNetClassifier(num_classes=2).to(DEVICE)

        # Train and evaluate for this fold
        train_and_evaluate(model, f"Fold_{fold+1}", train_loader, val_loader)

        # Store validation results
        val_loss, val_accuracy, val_auc, val_sensitivity, val_specificity = evaluate_metrics(model, val_loader, nn.CrossEntropyLoss(), phase="Validation")
        fold_results.append({
            "Fold": fold + 1,
            "Accuracy": val_accuracy,
            "AUC": val_auc,
            "Sensitivity": val_sensitivity,
            "Specificity": val_specificity
        })

    # Print & save final results
    results_df = pd.DataFrame(fold_results)
    print("\nFinal K-Fold Validation For Radiomics Results:\n", results_df.mean())
    results_df.to_csv("radiomics_with_masks_kfold_results_new_split.csv", index=False)