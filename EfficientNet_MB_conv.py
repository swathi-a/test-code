import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Swish activation function used in EfficientNet
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze and Excitation block used in EfficientNet
class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = input_channels // squeeze_factor
        self.fc1 = nn.Conv2d(input_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, input_channels, kernel_size=1)

    def forward(self, x):
        scale = torch.mean(x, (2, 3), keepdim=True)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

# MBConv block used in EfficientNet
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.stride = stride
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(expanded_channels)
        self.swish = Swish()

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size,
                                        stride=stride, padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)

        # Squeeze and Excitation
        self.se = SqueezeExcitation(expanded_channels, int(in_channels * se_ratio))

        # Output phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x

        x = self.swish(self.bn0(self.expand_conv(x)))
        x = self.swish(self.bn1(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.bn2(self.project_conv(x))

        if self.use_residual:
            return x + identity
        else:
            return x

# EfficientNet-B0 model from scratch
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetB0, self).__init__()
        
        # Stem
        self.stem_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)
        self.swish = Swish()

        # MBConv blocks
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, kernel_size=3, stride=1),
            MBConvBlock(16, 24, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(24, 24, expand_ratio=6, kernel_size=3, stride=1),
            MBConvBlock(24, 40, expand_ratio=6, kernel_size=5, stride=2),
            MBConvBlock(40, 40, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(40, 80, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=3, stride=1),
            MBConvBlock(80, 112, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(112, 112, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(112, 192, expand_ratio=6, kernel_size=5, stride=2),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(192, 320, expand_ratio=6, kernel_size=3, stride=1),
        )

        # Head
        self.head_conv = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        # Stem
        x = self.swish(self.bn0(self.stem_conv(x)))

        # MBConv blocks
        x = self.blocks(x)

        # Head
        x = self.swish(self.bn1(self.head_conv(x)))
        x = self.pool(x).flatten(1)
        x = self.classifier(x)

        return x

# Custom Dataset class
class DefectDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Generate label from mask: 1 if defect exists, otherwise 0
        label = 1 if 'noncrack' not in image_name else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print('Confusion Matrix:\n', confusion_matrix(all_labels, all_preds))
    print('Test Accuracy:', accuracy_score(all_labels, all_preds))
    print('Precision:', precision_score(all_labels, all_preds))
    print('Recall:', recall_score(all_labels, all_preds))
    print('Classification Report:\n', classification_report(all_labels, all_preds))


# Training and validation function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved!')

# Main function
if __name__ == "__main__":
    # Define transformations for dataset
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Paths to the image and mask directories (adjust accordingly)
    image_dir = 'path_to_images'
    mask_dir = 'path_to_masks'

    # Create dataset and dataloader
    dataset = DefectDataset(image_dir=image_dir, mask_dir=mask_dir, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model
    model = EfficientNetB0(num_classes=2).cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Example training phase
    train_and_validate(model, dataloader, dataloader, criterion, optimizer, num_epochs=5)

    # Example evaluation
    evaluate(model, dataloader)
