import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np

# U-Net Architecture


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle layer
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        enc1 = self.encoder(x)
        middle = self.middle(enc1)
        dec1 = self.decoder(middle)

        return dec1


# Transformation for both input images and masks
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Transformation for resizing masks to match model output size
transform_mask = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

#   this function is performed to enhance the ground truth images for perfect evaluation otherwise we'll recieve just wrong accuracy


def dilate_annotations(annotations_path, dilation_kernel_size):
    annotations = cv2.imread(annotations_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_annotations = cv2.dilate(annotations, kernel, iterations=1)
    return dilated_annotations

# Dataset class for your custom dataset


class CustomDataset(Dataset):

    def __init__(self, root, transform, transform_mask, ef=64, subset_fraction=0.2):
        self.dataset = ImageFolder(root, transform=transform)
        self.transform_mask = transform_mask
        self.enhancement_factor = ef
        self.subset_fraction = subset_fraction
        import random
        # Randomly select a subset of indices   [to slice the dataset and get random 20%]
        self.indices = random.sample(range(len(self.dataset)), int(
            subset_fraction * len(self.dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, _ = self.dataset[original_idx]
        mask_path = self.dataset.imgs[original_idx][0].replace("images", "gt")

        # print(mask_path)   #for checking

        annotations_path = mask_path
        original_annotations = cv2.imread(
            annotations_path, cv2.IMREAD_GRAYSCALE)

        # print("Min Value (Original):", np.min(original_annotations))   #for checking
        # print("Max Value (Original):", np.max(original_annotations))   #for checking

        annotations_normalized = (original_annotations - np.min(original_annotations)) / (
            np.max(original_annotations) - np.min(original_annotations)) * 255

        dilation_kernel_size = 5
        kernel = np.ones(
            (dilation_kernel_size, dilation_kernel_size), np.uint8)
        dilated_annotations = cv2.dilate(
            annotations_normalized, kernel, iterations=1)

        # Convert dilated_annotations to PIL Image
        dilated_annotations_pil = Image.fromarray(dilated_annotations)

        # print("Before transformation:", dilated_annotations_pil)   #for checking

        # Apply the mask transformation
        mask = self.transform_mask(dilated_annotations_pil)

        return img, mask


# Set the path to your dataset
dataset_path = 'DIRECTORY TO THE DATASET I PROVIDED IN README FILE VIA LINK'
custom_dataset = CustomDataset(
    root=dataset_path, transform=transform, transform_mask=transform_mask)
dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=True)

# Initialize model, loss function, and optimizer
model = UNet()
# Binary Cross Entropy loss for binary segmentation
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the trained model
model.load_state_dict(torch.load('MODEL PATH [pth FILE]'))
# set it for evaluation
model.eval()

# Initialize variables for confusion matrix
all_targets = []
all_predictions = []

# Testing loop
with torch.no_grad():
    for images, targets in tqdm(dataloader, desc='Testing'):
        try:
            # Your existing code here
            outputs = model(images)
            # Apply sigmoid activation to get probabilities
            predictions = torch.sigmoid(outputs)
            all_targets.extend(targets.view(-1).cpu().numpy().tolist())
            all_predictions.extend(predictions.view(-1).cpu().numpy().tolist())
        except Exception as e:
            print(f"Error processing batch: {e}")

        # Visualize the result
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(predictions[0, 0].cpu().numpy(), cmap='gray')
        plt.title('predicted')
        plt.subplot(1, 2, 2)
        plt.imshow(targets[0, 0].cpu().numpy(), cmap='gray')
        plt.title('ground truth')
        plt.show()

        # Visualize original predictions and thresholded predictions (optional)
        # Uncomment the following lines if you want to visualize the threshold predictions

        # plt.imshow(predictions_binary[0, 0].cpu().numpy(), cmap='gray')
        # plt.title('Thresholded Predictions')
        # plt.show()

# Convert to binary format
all_targets = [1 if t > 0.1 else 0 for t in all_targets]
all_predictions = [1 if p > 0.01 else 0 for p in all_predictions]

# adjust and sync length
min_length = min(len(all_predictions), len(all_targets))
all_predictions = all_predictions[:min_length]
all_targets = all_targets[:min_length]

cm = confusion_matrix(all_targets, all_predictions)

# Calculate accuracy
accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)

# Plot confusion matrix with a darker background
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Background', 'Object'], yticklabels=['True', 'Actual'])
plt.title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%')
plt.show()

# Display model summary
print(model)
