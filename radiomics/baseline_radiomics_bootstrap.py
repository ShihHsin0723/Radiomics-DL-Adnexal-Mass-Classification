import os
import cv2
import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
import random
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import datasets
import sys
sys.path.append("../baseline_model/pre_trained_cnns")
from EfficientNet import EfficientNetClassifier as Baseline_EfficientNetClassifier
torch.cuda.empty_cache()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Training Parameters
NUM_CLASSES = 2
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-3
THRESHOLD = 0.4

# Define the transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(0.5), 
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.ToTensor(),
])

# Dataset class to handle loading images and radiomics feature maps
class MultiChannelFromPaths(Dataset):
    def __init__(self, image_paths, labels, radiomics_dir, feature_map1, feature_map2):
        self.image_paths = image_paths
        self.labels = labels
        self.radiomics_dir = radiomics_dir
        self.feature_map1 = feature_map1
        self.feature_map2 = feature_map2
        self.classes = ["benign", "malignant"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        cls = self.classes[label]
        base_name = os.path.basename(image_path).rsplit('_', 1)[0]
        radiomics_folder = os.path.join(self.radiomics_dir, cls, base_name, "numpy")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224)) / 255.0

        rfm1 = np.load(os.path.join(radiomics_folder, self.feature_map1))
        rfm2 = np.load(os.path.join(radiomics_folder, self.feature_map2))
        rfm1 = (rfm1 - np.mean(rfm1)) / np.std(rfm1)
        rfm2 = (rfm2 - np.mean(rfm2)) / np.std(rfm2)

        combined = np.stack([img, rfm1, rfm2], axis=0)
        return torch.tensor(combined, dtype=torch.float32), label

class BasicImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = datasets.folder.default_loader(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_balanced_test_paths(data_dir):
    from collections import defaultdict

    test_images = defaultdict(list)
    for class_idx, class_name in enumerate(["benign", "malignant"]):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            if "image" in img_name.lower():
                test_images[class_idx].append(os.path.join(class_path, img_name))

    test_min_count = min(len(test_images[0]), len(test_images[1]))
    image_paths, labels = [], []

    for class_idx in [0, 1]:
        selected = random.sample(test_images[class_idx], test_min_count)
        image_paths.extend(selected)
        labels.extend([class_idx] * test_min_count)

    return image_paths, labels


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

def calculate_metrics(labels, preds, probs):
    auc = roc_auc_score(labels, probs) 
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 

    return auc, sensitivity, specificity

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_sensitivities, train_specificities, train_aucs = [], [], []
    val_sensitivities, val_specificities, val_aucs = [], [], []
    f1_scores = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_train_labels, all_train_preds, all_train_probs = [], [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int) 
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)
            all_train_probs.extend(probs)
            correct += (preds == labels.cpu().numpy()).sum()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Compute train AUC, Sensitivity, and Specificity
        train_auc, train_sensitivity, train_specificity = calculate_metrics(
            all_train_labels, all_train_preds, all_train_probs
        )
        train_aucs.append(train_auc)
        train_sensitivities.append(train_sensitivity)
        train_specificities.append(train_specificity)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_labels, all_val_probs, all_val_preds = [], [], []

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_labels)
                val_loss += v_loss.item()

                y_probs = torch.softmax(val_outputs, dim=1)[:, 1]
                val_predicted = (y_probs >= 0.4).long() 
                val_correct += (val_predicted == val_labels).sum().item()
                val_total += val_labels.size(0)

                all_val_labels.extend(val_labels.cpu().numpy())
                all_val_probs.extend(y_probs.cpu().numpy())
                all_val_preds.extend(val_predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Compute Validation AUC, Sensitivity, and Specificity
        val_auc, val_sensitivity, val_specificity = calculate_metrics(
            all_val_labels, all_val_preds, all_val_probs
        )
        val_aucs.append(val_auc)
        val_sensitivities.append(val_sensitivity)
        val_specificities.append(val_specificity)

        # Compute F1-score
        f1 = f1_score(all_val_labels, all_val_preds)
        f1_scores.append(f1)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Train AUC: {train_auc:.4f} | Train Sensitivity: {train_sensitivity:.4f} | Train Specificity: {train_specificity:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f} | "
              f"Val Sensitivity: {val_sensitivity:.4f} | Val Specificity: {val_specificity:.4f} | "
              f"F1-Score: {f1:.4f}")


# Evaluate function
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() 

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    TP, TN, FP, FN = 0, 0, 0, 0 
    
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            y_probs = torch.softmax(outputs, dim=1)[:, 1]
            y_probs = torch.softmax(outputs, dim=1)[:, 1]
            predicted = (y_probs >= THRESHOLD).long() 
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(y_probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            # Compute TP, TN, FP, FN
            for i in range(len(labels)):
                if labels[i] == 1 and predicted[i] == 1:
                    TP += 1
                elif labels[i] == 0 and predicted[i] == 0:
                    TN += 1
                elif labels[i] == 0 and predicted[i] == 1:
                    FP += 1
                elif labels[i] == 1 and predicted[i] == 0:
                    FN += 1

    test_loss /= len(test_loader)
    test_acc = correct / total * 100
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0 
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # Compute F1-score and AUC
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    print("\nTest Set Evaluation:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    return test_loss, test_acc, sensitivity, specificity, f1, auc

# Function to balance test set
def balance_test_set(dataset):
    label_counts = Counter(dataset.labels)
    min_samples = min(label_counts.values())
    
    benign_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    malignant_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

    benign_indices = random.sample(benign_indices, min_samples)
    malignant_indices = random.sample(malignant_indices, min_samples)

    balanced_indices = benign_indices + malignant_indices
    return torch.utils.data.Subset(dataset, balanced_indices)

# Save test metrics
def save_results_to_csv(csv_file, test_loss, test_accuracy, test_auc, test_sensitivity, test_specificity, test_f1):
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Batch Size", "Epochs", "Learning Rate", "Test Loss",
                             "Test Accuracy", "Test AUC", "Test Sensitivity", "Test Specificity", "Test F1"])

        writer.writerow([BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, test_loss,
                         test_accuracy, test_auc, test_sensitivity, test_specificity, test_f1])

    print(f"Test results saved to {csv_file}")


if __name__ == "__main__":
    feature_map1 = "original_glcm_SumEntropy.npy"
    feature_map2 = "original_glrlm_RunLengthNonUniformity.npy"

    # Define paths
    test_image_dir = "/home/phoebe0723/rop2/all_data/test"
    test_radiomics_dir = "/home/phoebe0723/rop2/radiomics_feature_maps/test"
    
    # Load trained radiomics model
    radiomics_model = EfficientNetClassifier(num_classes=2)
    radiomics_model.load_state_dict(torch.load("radiomics_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    radiomics_model.eval()
    print("Loaded trained radioimics model from radiomics_model.pth")

    # Load trained baseline model
    baseline_model = Baseline_EfficientNetClassifier(num_classes=2)
    baseline_model.load_state_dict(torch.load("../baseline_model/efficientnet_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    baseline_model.eval()
    print("Loaded trained baseline model from efficientnet_model.pth")

    # Prepare balanced image paths & labels once
    test_image_paths, test_labels = get_balanced_test_paths(test_image_dir)

    # Extract dataset name dynamically from the path
    radiomics_csv_file = f"./resample_results/resample_radiomics_test_results.csv"
    baseline_csv_file = f"./resample_results/resample_baseline_test_results.csv"

    # Perform bootstrapping on both
    for i in range(5000):
        print(f"\nEvaluation Run {i+1}/5000")

        # Sample with replacement for bootstrapping
        indices = np.random.choice(len(test_image_paths), size=len(test_image_paths), replace=True)
        sampled_paths = [test_image_paths[idx] for idx in indices]
        sampled_labels = [test_labels[idx] for idx in indices]

        # Create test datasets and loaders for both models
        radiomics_test_dataset = MultiChannelFromPaths(
            sampled_paths, sampled_labels, test_radiomics_dir, feature_map1, feature_map2
        )
        baseline_test_dataset = BasicImageDataset(sampled_paths, sampled_labels, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ]))

        radiomics_loader = DataLoader(radiomics_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        baseline_loader = DataLoader(baseline_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Evaluate and save results for both models
        r_loss, r_acc, r_sens, r_spec, r_f1, r_auc = evaluate_model(radiomics_model, radiomics_loader)
        save_results_to_csv(radiomics_csv_file, r_loss, r_acc, r_auc, r_sens, r_spec, r_f1)

        b_loss, b_acc, b_sens, b_spec, b_f1, b_auc = evaluate_model(baseline_model, baseline_loader)
        save_results_to_csv(baseline_csv_file, b_loss, b_acc, b_auc, b_sens, b_spec, b_f1)