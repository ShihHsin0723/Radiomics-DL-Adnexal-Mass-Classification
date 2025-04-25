import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import Counter
from torch.optim import Adam
from pre_trained_cnns.EfficientNet import EfficientNetClassifier
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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters setting
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-2
THRESHOLD = 0.4

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

VAL_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def calculate_metrics(labels, preds, probs):
    auc = roc_auc_score(labels, probs[:, 1]) 
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
    return auc, sensitivity, specificity

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.labels = [dataset.samples[i][1] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        img = self.transform(img)
        return img, label

def balance_validation_set(val_dataset):
    benign_indices = [i for i, label in enumerate(val_dataset.labels) if label == 0]
    malignant_indices = [i for i, label in enumerate(val_dataset.labels) if label == 1]

    min_samples = min(len(benign_indices), len(malignant_indices))

    benign_sampled = random.sample(benign_indices, min_samples)
    malignant_sampled = random.sample(malignant_indices, min_samples)

    balanced_indices = benign_sampled + malignant_sampled
    random.shuffle(balanced_indices)

    balanced_dataset = CustomDataset(val_dataset.dataset, [val_dataset.indices[i] for i in balanced_indices], val_dataset.transform)
    return balanced_dataset

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
            preds = (probs[:, 1] >= THRESHOLD).astype(int)

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

            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = (probs[:, 1] >= THRESHOLD).astype(int)

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
    # Define dataset path
    DATA_DIR = "/home/phoebe0723/rop2/all_data/merged"

    # Load dataset (entire dataset in one folder)
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=None)
    print(f"Class Mapping: {full_dataset.class_to_idx}") 

    # Extract Patient IDs from Filenames
    image_paths = [img_path[0] for img_path in full_dataset.samples]
    patient_ids = np.array([os.path.basename(path).split("_")[2] for path in image_paths])

    # Get Unique Patient IDs and Split by Patient
    unique_patients = np.unique(patient_ids)
    print(f"Total unique patients: {len(unique_patients)}")
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    # Track fold results
    fold_results = []

    for fold, (train_patients, val_patients) in enumerate(kf.split(unique_patients)):
        print(f"\n----- Fold {fold+1}/{K} -----\n")

        train_patient_ids = unique_patients[train_patients]
        val_patient_ids = unique_patients[val_patients]

        print(f"\n----- Fold {fold+1}/{K} -----")
        print(f"Train Patients: {len(train_patient_ids)} patients")
        print(f"Validation Patients: {len(val_patient_ids)} patients")

        # Convert Patient IDs to Image Indices
        train_indices = np.where(np.isin(patient_ids, unique_patients[train_patients]))[0]
        val_indices = np.where(np.isin(patient_ids, unique_patients[val_patients]))[0]

        # Create Train & Validation Datasets
        train_dataset = CustomDataset(full_dataset, train_indices, TRAIN_TRANSFORM)
        val_dataset = CustomDataset(full_dataset, val_indices, VAL_TEST_TRANSFORM)

        # Balance the training set using WeightedRandomSampler
        train_labels = np.array([full_dataset.samples[i][1] for i in train_indices])
        class_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = np.array([class_weights[label] for label in train_labels])
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        # Apply undersampling to balance the validation set
        val_dataset_balanced = balance_validation_set(val_dataset)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_dataset_balanced, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = EfficientNetClassifier(num_classes=2).to("cuda" if torch.cuda.is_available() else "cpu")

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
    print("\nFinal K-Fold Validation Results:\n", results_df.mean())
    results_df.to_csv("kfold_results_new_split.csv", index=False)
