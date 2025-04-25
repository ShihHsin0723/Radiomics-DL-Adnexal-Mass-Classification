import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(EfficientNetClassifier, self).__init__()

        # Load the pretrained EfficientNet-B0 model
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze the pretrained layers
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Modify the classifier head
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)
