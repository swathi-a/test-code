import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from imblearn.metrics import specificity_score
import cv2, argparse
from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision.transforms.functional import to_tensor
import torchvision.models as models
from torchvision.models.inception import InceptionOutputs
from timm import create_model
from sklearn.manifold import TSNE
import math

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
        mask_path = os.path.join(self.mask_dir,
                                 image_name)  # Assuming mask files have the same name but with a .png extension

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Generate label from mask: 1 if defect exists, otherwise 0
        mask_np = np.array(mask)
        if 'noncrack' in image_name:
            label = 1
        else:
            label = 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# Model definition
# 定义DefectClassifier类
class DefectClassifier(nn.Module):
    def __init__(self, model_type='efficientnet'):
        super(DefectClassifier, self).__init__()
        if model_type == 'efficientnet':
            self.model = efficientnet_b0(pretrained=True)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()  # 移除原来的分类器

        elif model_type == 'efficientnetv2_rw_m':
            self.model = create_model('efficientnetv2_rw_m', pretrained=True)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()  # 移除原来的分类器

        elif model_type == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            in_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Identity()  # 移除原来的分类器

        elif model_type == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()  # 移除原来的分类器

        elif model_type == 'inception_v3':
            self.model = models.inception_v3(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()  # 移除原来的分类器

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 使用正确的输入通道数
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),  # 第一层
            nn.ReLU(),
            nn.Dropout(0.5),  # 可选的Dropout层，防止过拟合

            nn.Linear(512, 256),  # 第二层
            nn.ReLU(),
            nn.Dropout(0.5),  # 可选的Dropout层

            nn.Linear(256, 2)  # 第三层，用于二分类
        )

    def save_gram(self, feature, filename):
        """
        将特征向量保存为PNG图像。

        参数:
        - feature: 2D张量 (batch_size, channels)
        - filename: 保存的PNG文件名
        """
        if len(feature.shape) == 2:
            # 将每个特征向量展平为二维图像
            batch_size, channels = feature.shape
            for i in range(batch_size):
                single_feature = feature[i]

                # 重塑为接近正方形的2D网格
                side_length = math.ceil(math.sqrt(channels))
                padded_feature = torch.zeros(side_length * side_length)
                padded_feature[:channels] = single_feature
                padded_feature = padded_feature.view(side_length, side_length)

                # 正则化到0-1
                padded_feature = (padded_feature - padded_feature.min()) / (padded_feature.max() - padded_feature.min())

                # 转换为PIL图像格式
                to_pil = transforms.ToPILImage()
                image = to_pil(padded_feature.unsqueeze(0))  # 添加一个通道维度

                # 创建保存路径的文件夹
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                # 保存图像为PNG文件
                image.save(f"{filename}_sample{i}.png")

    def forward(self, x):
        mid_fea = self.model(x)
        final_acc = self.classifier(mid_fea)
        return mid_fea, final_acc


# Evaluation on the test set
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
    print('Specificity:', specificity_score(all_labels, all_preds))
    print('Classification Report:\n',
          classification_report(all_labels, all_preds, target_names=['Non-defective', 'Defective']))



# Training and validation function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, model_dir='ckpts',
                       model_type='efficientnet'):
    best_model_path = 'best_model.pth'
    best_val_accuracy = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            fea, outputs = model(images)
            logits = outputs.logits if isinstance(outputs, InceptionOutputs) else outputs

            loss = criterion(logits, labels)
            train_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        vis_features = []
        vis_labels = []

        # Visualization test image
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                feature, outputs = model(images)
                logits = outputs.logits if isinstance(outputs, InceptionOutputs) else outputs
                model.save_gram(feature, "output_features/feature_output")

                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                vis_features.append(feature.cpu().numpy())
                vis_labels.append(labels.cpu().numpy())

            vis_features = np.concatenate(vis_features)
            vis_labels = np.concatenate(vis_labels)

            # 使用 t-SNE 进行降维
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(vis_features)

            # 可视化 t-SNE 结果
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=vis_labels, cmap='viridis')
            plt.legend(*scatter.legend_elements(), title="Classes")
            plt.title('t-SNE Visualization of Image Features')
            plt.savefig('tsne_visualization_step_%d.png' % epoch, dpi=300, bbox_inches='tight')
            plt.show()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

            # Save the best model and test it
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(model_dir, best_model_path))
                print(f'Best model saved at epoch {epoch + 1}')

                # evaluate(model, test_loader)

                '''
                # plot gradcam
                target_layer = model.model.classifier[0]
                image_path = os.path.join(source_dir, test_image_dir, 'CFD_001.jpg') # 替换为你的图像路径
                save_path = './gradcam_image.jpg'  # 替换为你想保存的路
                visualize_gradcam(model, image_path, transform, target_layer, save_path)
                '''
    # Plotting training loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.show()


# Main function to handle arguments and call appropriate functions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test an image classification model.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Train or test the model')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save the model')
    parser.add_argument('--model_type', type=str, default='vgg16', choices=['vgg16', 'inception_v3',
                                                                                   'resnet18', 'efficientnetv2_rw_m',
                                                                                   'efficientnet'],
                        help='chose different models')

    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_image_dir = 'train/images'
    train_mask_dir = 'train/masks'
    test_image_dir = 'test/images'
    test_mask_dir = 'test/masks'

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = DefectDataset(os.path.join(args.data_dir, train_image_dir),
                                  os.path.join(args.data_dir, train_mask_dir), transform=transform)
    test_dataset = DefectDataset(os.path.join(args.data_dir, test_image_dir),
                                 os.path.join(args.data_dir, test_mask_dir), transform=transform)
    val_dataset = DefectDataset(os.path.join(args.data_dir, test_image_dir), 
                                os.path.join(args.data_dir, test_mask_dir), transform=transform)

    # train_dataset = CocoClassificationDataset(image='data/images/train2024', annotation='data/annotations/instances_train2024.json', transforms=transform)
    # val_dataset = CocoClassificationDataset(image='data/images/val2024', annotation='data/annotations/instances_val2024.json', transforms=transform)
    # test_dataset = CocoClassificationDataset(image='data/images/test2024', annotation='data/annotations/instances_test2024.json', transforms=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  vgg16, inception_v3, resnet18, efficientnetv2_rw_m efficientnet
    model = DefectClassifier(args.model_type).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.mode == 'train':
        train_and_validate(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, args.model_dir,
                           args.model_type)
    elif args.mode == 'test':
        evaluate(model, test_loader, args.model_dir)




