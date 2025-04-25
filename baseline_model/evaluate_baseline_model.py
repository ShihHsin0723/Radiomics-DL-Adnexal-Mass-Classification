import os
import csv
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import defaultdict
from datetime import datetime
from pre_trained_cnns.EfficientNet import EfficientNetClassifier

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

# Training Parameters
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


def create_dataloaders(data_dir):
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=TRAIN_TRANSFORM)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "validation"), transform=VAL_TEST_TRANSFORM)

    # Compute class weights for imbalanced datasets
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in train_dataset.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def create_balanced_test_loader(data_dir):
    test_images = defaultdict(list)
    
    for class_idx, class_name in enumerate(["benign", "malignant"]):
        class_path = os.path.join(data_dir, "test", class_name)
        for img_name in os.listdir(class_path):
            if "image" in img_name.lower():
                test_images[class_idx].append(os.path.join(class_path, img_name))

    test_min_count = min(len(test_images[0]), len(test_images[1]))
    balanced_test_images, balanced_test_labels = [], []

    for class_idx in [0, 1]:
        selected_images = random.sample(test_images[class_idx], test_min_count)
        balanced_test_images.extend(selected_images)
        balanced_test_labels.extend([class_idx] * test_min_count)

    class BalancedTestDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            image = datasets.folder.default_loader(img_path)
            if self.transform:
                image = self.transform(image)
            return image, label

    test_dataset = BalancedTestDataset(balanced_test_images, balanced_test_labels, transform=VAL_TEST_TRANSFORM)
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

def calculate_metrics(labels, preds, probs):
    auc = roc_auc_score(labels, probs[:, 1]) 
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate

    return auc, sensitivity, specificity

def evaluate_metrics(model, data_loader, criterion, phase="Test"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            correct += (preds == labels.cpu().numpy()).sum()
            total += labels.size(0)

    accuracy = correct / total * 100
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"{phase} Metrics:\nLoss: {running_loss / len(data_loader):.4f} | "
          f"Accuracy: {accuracy:.2f}% | AUC: {auc:.4f} | "
          f"Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}\n")

    return running_loss / len(data_loader), accuracy, auc, sensitivity, specificity

def save_results_to_csv(model_name, file_name, test_loss, test_accuracy, test_auc, test_sensitivity, test_specificity):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header if the file does not exist
        if not file_exists:
            writer.writerow(["Model", "Batch Size", "Epochs", "Learning Rate", "Test Loss",
                             "Test Accuracy", "Test AUC", "Test Sensitivity", "Test Specificity"])

        # Write test results
        writer.writerow([model_name, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, test_loss,
                         test_accuracy, test_auc, test_sensitivity, test_specificity])
    
    print(f"Test results saved to {file_name}")

def train_and_evaluate(model, model_name, train_data_dir):
    print(f"\nTraining {model_name} on {train_data_dir}...\n")
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader, val_loader = create_dataloaders(train_data_dir)
    test_loader = create_balanced_test_loader(train_data_dir)

    # Lists to store metrics per epoch
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_aucs, train_sensitivities, train_specificities = [], [], []
    val_aucs, val_sensitivities, val_specificities = [], [], []

    # Store training and validation metrics per epoch
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
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            correct += (preds == labels.cpu().numpy()).sum()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total * 100
        train_auc, train_sensitivity, train_specificity = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))

        # Print epoch results for train set
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("Training Metrics:")
        print(f"Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.2f}% | AUC: {train_auc:.4f} | Sensitivity: {train_sensitivity:.4f} | Specificity: {train_specificity:.4f}")
    
        # Evaluate on Validation Set after each epoch
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


if __name__ == "__main__":
    train_data_dir = "/home/phoebe0723/rop2/all_data"

    TEST_DATA_DIRS = {
        "/home/phoebe0723/rop2/all_data": "clean",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.05": "speckle_0.05",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.1": "speckle_0.1",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.2": "speckle_0.2",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.3": "speckle_0.3",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.4": "speckle_0.4",
        "/home/phoebe0723/rop2/speckle_noise_data/add_speckle_noise_data_0.5": "speckle_0.5",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.05": "gaussian_0.05",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.1": "gaussian_0.1",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.2": "gaussian_0.2",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.3": "gaussian_0.3",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.4": "gaussian_0.4",
        "/home/phoebe0723/rop2/gaussian_noise_data/add_gaussian_noise_data_0.5": "gaussian_0.5",
    }

    output_dir = "./EfficientNet_Results_noise"
    os.makedirs(output_dir, exist_ok=True)

    model = EfficientNetClassifier(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("./efficientnet_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Loaded pre-trained EfficientNet model.")

    for test_data_dir, label_name in TEST_DATA_DIRS.items():
        csv_file = os.path.join(output_dir, f"{label_name}_efficientNet_test_results.csv")
        test_loader = create_balanced_test_loader(test_data_dir)
        test_loss, test_accuracy, test_auc, test_sensitivity, test_specificity = evaluate_metrics(
            model, test_loader, nn.CrossEntropyLoss(), phase="Test"
        )

        save_results_to_csv("EfficientNet", csv_file, test_loss, test_accuracy, test_auc, test_sensitivity, test_specificity)

    # model = EfficientNetClassifier(num_classes=NUM_CLASSES)
    # csv_file = os.path.join("./EfficientNet_Results", "EfficientNet_test_results_pre.csv")
    # train_and_evaluate(model, "EfficientNet", train_data_dir, csv_file)