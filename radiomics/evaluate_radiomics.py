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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

torch.cuda.empty_cache()

# Training Parameters
NUM_CLASSES = 2
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-3
THRESHOLD = 0.4


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.ToTensor(),
])


class MultiChannelDataset(Dataset):
    def __init__(self, image_dir, radiomics_dir, feature_map1, feature_map2, transform=None):
        self.image_dir = image_dir
        self.radiomics_dir = radiomics_dir
        self.feature_map1 = feature_map1
        self.feature_map2 = feature_map2
        self.transform = transform
        self.classes = ["benign", "malignant"]
        self.image_paths = []
        self.labels = []
        
        for label, cls in enumerate(self.classes):
            class_path = os.path.join(image_dir, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img_base_name = os.path.basename(img_name).rsplit('_', 1)[0] 
                radiomics_folder = os.path.join(self.radiomics_dir, cls, img_base_name, "numpy")
                
                path1 = os.path.join(radiomics_folder, self.feature_map1)
                path2 = os.path.join(radiomics_folder, self.feature_map2)
                
                if os.path.exists(path1) and os.path.exists(path2):
                    rfm1 = np.load(path1)
                    rfm2 = np.load(path2)

                    if np.all(rfm1 == 0) or np.all(rfm2 == 0):
                        continue

                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        img_name = os.path.basename(image_path).rsplit('_', 1)[0]
        radiomics_folder = os.path.join(self.radiomics_dir, self.classes[label], img_name, "numpy")
        
        radiomics_path1 = os.path.join(radiomics_folder, self.feature_map1)
        radiomics_path2 = os.path.join(radiomics_folder, self.feature_map2)
        
        # Load images and feature maps
        ultrasound_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        radiomics_map1 = np.load(radiomics_path1)
        radiomics_map2 = np.load(radiomics_path2)

        # Preprocess input
        ultrasound_img = cv2.resize(ultrasound_img, (224, 224)) / 255.0
        radiomics_map1 = (radiomics_map1 - np.mean(radiomics_map1)) / np.std(radiomics_map1)
        radiomics_map2 = (radiomics_map2 - np.mean(radiomics_map2)) / np.std(radiomics_map2)
        
        combined_input = np.stack([ultrasound_img, radiomics_map1, radiomics_map2], axis=0)  
        return torch.tensor(combined_input, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.65):
        super(EfficientNetClassifier, self).__init__()
        
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
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
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
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


def balance_test_set(dataset):
    label_counts = Counter(dataset.labels)
    min_samples = min(label_counts.values())
    
    benign_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    malignant_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

    benign_indices = random.sample(benign_indices, min_samples)
    malignant_indices = random.sample(malignant_indices, min_samples)

    balanced_indices = benign_indices + malignant_indices
    return torch.utils.data.Subset(dataset, balanced_indices)


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

    # train_image_dir = "/home/phoebe0723/rop2/all_data/train"
    # train_radiomics_dir = "/home/phoebe0723/rop2/radiomics_feature_maps/train"
    # val_image_dir = "/home/phoebe0723/rop2/all_data/validation"
    # val_radiomics_dir = "/home/phoebe0723/rop2/radiomics_feature_maps/validation"

    TEST_IMAGE_DIRS = [
        "/home/phoebe0723/rop2/all_data/test",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.05/test",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.1/test",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.2/test",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.3/test",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.4/test",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.5/test",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.05/test",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.1/test",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.2/test",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.3/test",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.4/test",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.5/test",
    ]
    TEST_RADIOMICS_DIRS = [
        "/home/phoebe0723/rop2/radiomics_feature_maps/test",
        "/home/phoebe0723/rop2/speckle_noise_radiomics_feature_maps/add_speckle_noise_rfm_0.05/test",
        "/home/phoebe0723/rop2/speckle_noise_radiomics_feature_maps/add_speckle_noise_rfm_0.1/test",
        "/home/phoebe0723/rop2/speckle_noise_radiomics_feature_maps/add_speckle_noise_rfm_0.2/test",
        "/home/phoebe0723/rop2/speckle_noise_radiomics_feature_maps/add_speckle_noise_rfm_0.3/test",
        "/home/phoebe0723/rop2/speckle_noise_radiomics_feature_maps/add_speckle_noise_rfm_0.4/test",
        "/home/phoebe0723/rop2/speckle_noise_radiomics_feature_maps/add_speckle_noise_rfm_0.5/test",
        "/home/phoebe0723/rop2/gaussian_noise_radiomics_feature_maps/add_gaussian_noise_rfm_0.05/test",
        "/home/phoebe0723/rop2/gaussian_noise_radiomics_feature_maps/add_gaussian_noise_rfm_0.1/test",
        "/home/phoebe0723/rop2/gaussian_noise_radiomics_feature_maps/add_gaussian_noise_rfm_0.2/test",
        "/home/phoebe0723/rop2/gaussian_noise_radiomics_feature_maps/add_gaussian_noise_rfm_0.3/test",
        "/home/phoebe0723/rop2/gaussian_noise_radiomics_feature_maps/add_gaussian_noise_rfm_0.4/test",
        "/home/phoebe0723/rop2/gaussian_noise_radiomics_feature_maps/add_gaussian_noise_rfm_0.5/test"
    ]
    levels = 13

    # # Create datasets
    # train_dataset = MultiChannelDataset(train_image_dir, train_radiomics_dir, feature_map1, feature_map2, transform=transform)
    # val_dataset = MultiChannelDataset(val_image_dir, val_radiomics_dir, feature_map1, feature_map2)

    # # Compute class weights for training sampler
    # class_counts = Counter(train_dataset.labels)
    # class_weights = {label: 1.0 / count for label, count in class_counts.items()}
    # sample_weights = [class_weights[label] for label in train_dataset.labels]
    # sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
   
    # # Create DataLoaders with sampler for oversampling
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train the model
    model = EfficientNetClassifier(num_classes=2)
    # train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    # torch.save(model.state_dict(), "radiomics_model.pth")
    # print("Model saved to radiomics_model.pth")

    # Loading the model
    model_path = "radiomics_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for level in range(levels):
        # Obtain the radiomics and image dir for specific noise and level
        test_radiomics_dir = TEST_RADIOMICS_DIRS[level]
        test_image_dir = TEST_IMAGE_DIRS[level]

        # Extract dataset name dynamically from the path
        dataset_name = os.path.basename(os.path.dirname(test_radiomics_dir))
        csv_file = f"./noisy_results_new_split/{dataset_name}.csv"

        # Creat test dataloader
        test_dataset = MultiChannelDataset(test_image_dir, test_radiomics_dir, feature_map1, feature_map2)
        balanced_test_dataset = balance_test_set(test_dataset)
        test_loader = DataLoader(balanced_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Evaluate on Test Set
        test_loss, test_acc, sensitivity, specificity, f1, auc = evaluate_model(model, test_loader)
        save_results_to_csv(csv_file, test_loss, test_acc, auc, sensitivity, specificity, f1)
     