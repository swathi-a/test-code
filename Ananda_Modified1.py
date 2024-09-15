import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.metrics import specificity_score
import cv2, argparse
from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision.transforms.functional import to_tensor
import torchvision.models as models
from torchvision.models.inception import InceptionOutputs
from timm import create_model
from sklearn.manifold import TSNE


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
        mask_path = os.path.join(self.mask_dir, image_name)  # Assuming mask files have the same name but with a .png extension

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
class DefectClassifier(nn.Module):
    def __init__(self, model_type='efficientnet'):
        super(DefectClassifier, self).__init__()
        if model_type == 'efficientnet':
            self.model = efficientnet_b0(pretrained=True)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, 512), 
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(512, 256),  
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(256, 2)  
            )
        elif model_type == 'efficientnetv2_rw_m':
            self.model = create_model('efficientnetv2_rw_m', pretrained=True)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),  
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(512, 256),  
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(256, 2)  
            )
        elif model_type == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            in_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),  
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(512, 256),  
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(256, 2)  
            )
        elif model_type == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, 512),  
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(512, 256),  
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(256, 2)  
            )
        elif model_type == 'inception_v3':
            self.model = models.inception_v3(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, 512),  
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(512, 256),  
                nn.ReLU(),
                nn.Dropout(0.5),  

                nn.Linear(256, 2)  
            )
    def forward(self, x):
        return self.model(x)


# Grad-CAM 可视化类
class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.hook()

    def hook(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_image, class_index=None):
        self.model.eval()
        input_image.requires_grad_()  # 确保输入张量需要梯度

        model_output = self.model(input_image)
        if class_index is None:
            class_index = torch.argmax(model_output)

        self.model.zero_grad()
        model_output[0, class_index].backward(retain_graph=True)  # 计算梯度

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
        heatmap = cv2.resize(heatmap, (input_image.size(3), input_image.size(2)))
        heatmap = heatmap / np.max(heatmap)

        return heatmap


# 可视化函数
def visualize_gradcam(model, image_path, transform, target_layer, save_path):
    model.eval()

    # 加载图像并预处理
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).cuda()
    input_tensor.requires_grad_()  # 确保输入张量需要梯度

    # 实例化GradCam并生成热力图
    grad_cam = GradCam(model, target_layer)
    heatmap = grad_cam.generate(input_tensor)

    # 应用热力图并叠加在原始图像上
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_image = heatmap + np.float32(transforms.ToTensor()(image).permute(1, 2, 0).numpy())
    cam_image = cam_image / np.max(cam_image)

    # 显示图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title('Grad-CAM')

    # 保存图像
    cam_image = np.uint8(255 * cam_image)
    cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    plt.show()


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
    print('Classification Report:\n', classification_report(all_labels, all_preds, target_names=['Non-defective', 'Defective']))


# Training and validation function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, model_dir='ckpts', model_type='efficientnet'):
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

            outputs = model(images)
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
        if model_type == 'efficientnet':
            model = efficientnet_b0(pretrained=True).to(device)
        elif model_type == 'efficientnetv2_rw_m':
            model = create_model('efficientnetv2_rw_m', pretrained=True).to(device)
        elif model_type == 'vgg16':
            model = models.vgg16(pretrained=True).to(device)
        elif model_type == 'resnet18':
            model = models.resnet18(pretrained=True).to(device)
        elif model_type == 'inception_v3':
            model = models.inception_v3(pretrained=True).to(device)

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                logits = outputs.logits if isinstance(outputs, InceptionOutputs) else outputs

                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                feature = model(images)
                vis_features.append(feature.cpu().numpy())
                vis_labels.append(labels.cpu().numpy())

            vis_features = np.concatenate(vis_features)
            vis_labels = np.concatenate(vis_labels)

         
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(vis_features)

            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=vis_labels, cmap='viridis')
            plt.legend(*scatter.legend_elements(), title="Classes")
            plt.title('t-SNE Visualization of Image Features')
            plt.savefig('tsne_visualization_step_%d.png'%epoch, dpi=300, bbox_inches='tight')
            plt.show()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

            # Save the best model and test it
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(),  os.path.join(model_dir, best_model_path) )
                print(f'Best model saved at epoch {epoch+1}')

                #evaluate(model, test_loader)
       
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
    parser.add_argument('--model_type', type=str, default='efficientnet', choices=['vgg16', 'inception_v3',
                                                                            'resnet18', 'efficientnetv2_rw_m', 'efficientnet'],
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


    # Datasets
    train_dataset = DefectDataset(os.path.join(args.data_dir, train_image_dir), os.path.join(args.data_dir, train_mask_dir), transform=transform)
    test_dataset = DefectDataset(os.path.join(args.data_dir, test_image_dir), os.path.join(args.data_dir, test_mask_dir), transform=transform)
    val_dataset = DefectDataset(os.path.join(args.data_dir, test_image_dir), os.path.join(args.data_dir, test_mask_dir), transform=transform)

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
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    if args.mode == 'train':
        train_and_validate(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, args.model_dir, args.model_type)
    elif args.mode == 'test':
        evaluate(model, test_loader, args.model_dir)



