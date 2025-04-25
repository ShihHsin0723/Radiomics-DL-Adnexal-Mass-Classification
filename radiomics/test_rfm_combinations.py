import os
import itertools
import cv2
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
from sklearn.metrics import roc_auc_score, f1_score

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
NUM_EPOCHS = 35
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 1e-2
THRESHOLD = 0.4

# Define paths
train_image_dir = "/home/phoebe0723/rop2/segmented_data/train"
train_radiomics_dir = "/home/phoebe0723/rop/radiomics_feature_maps/train"
val_image_dir = "/home/phoebe0723/rop2/segmented_data/validation"
val_radiomics_dir = "/home/phoebe0723/rop2/radiomics_feature_maps/validation"
test_image_dir = "/home/phoebe0723/rop2/segmented_data/test"
test_radiomics_dir = "/home/phoebe0723/rop2/radiomics_feature_maps/test"

# Get all available feature maps
sample_patient_folder = os.path.join(test_radiomics_dir, "benign")
first_patient = os.listdir(sample_patient_folder)[0]
radiomics_folder = os.path.join(sample_patient_folder, first_patient, "numpy")

feature_maps = [f for f in os.listdir(radiomics_folder)]
feature_combinations = list(itertools.combinations(feature_maps, 2))

print(f"Total feature map pairs to test: {len(feature_combinations)}")

# Define transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# Dataset class to handle loading images and radiomics feature maps
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
                
                # Check if both selected feature maps exist
                path1 = os.path.join(radiomics_folder, self.feature_map1)
                path2 = os.path.join(radiomics_folder, self.feature_map2)
                
                if os.path.exists(path1) and os.path.exists(path2):
                    rfm1 = np.load(path1)
                    rfm2 = np.load(path2)

                    # Skip feature maps that are completely black (all zeros)
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

# Define model
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

# Train function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            # _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())

            probs = F.softmax(outputs, dim=1)[:, 1] 
            predicted = (probs >= THRESHOLD).long() 
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) 

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

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

    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = float('nan') 

    # Calculate F1-score
    f1 = f1_score(all_labels, all_predictions, average='binary')

    return test_loss, test_acc, sensitivity, specificity, auc_score, f1

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


results = []
csv_file = "feature_map_combinations_results.csv"

# Iterate over feature map combinations
for feature_map1, feature_map2 in feature_combinations:
    print(f"Testing combination: {feature_map1} & {feature_map2}")

    # Create datasets
    train_dataset = MultiChannelDataset(train_image_dir, train_radiomics_dir, feature_map1, feature_map2, transform)
    val_dataset = MultiChannelDataset(val_image_dir, val_radiomics_dir, feature_map1, feature_map2)
    test_dataset = MultiChannelDataset(test_image_dir, test_radiomics_dir, feature_map1, feature_map2)

    # Compute class weights for training sampler
    class_counts = Counter(train_dataset.labels)
    class_weights = {label: 1.0 / count for label, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Balance test set
    balanced_test_dataset = balance_test_set(test_dataset)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(balanced_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train the model
    model = EfficientNetClassifier(num_classes=2)
    train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    test_loss, test_acc, sensitivity, specificity, auc_score, f1 = evaluate_model(model, test_loader)

    # Write result immediately to CSV
    result_row = [feature_map1, feature_map2, test_loss, test_acc, sensitivity, specificity, auc_score]
    df = pd.DataFrame([result_row], columns=["Feature Map 1", "Feature Map 2", "Test Loss", "Test Accuracy", "Sensitivity", "Specificity", "AUC"])
    
    # Append to CSV, only write header if file doesn't exist
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)