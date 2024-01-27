from libs.models import DualSeg_res101
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm


# Transformation for both input images and masks
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Transformation for resizing masks to match model output size
transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Dataset class for your custom dataset
class CustomDataset(Dataset):
    def __init__(self, root, transform_img, transform_mask):
        self.dataset = ImageFolder(root, transform=None)
        self.transform_mask = transform_mask
        self.transform_img = transform_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        mask_path = self.dataset.imgs[idx][0].replace("images", "gt")

        # Load PNG mask images and resize
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        mask = self.transform_mask(mask)

        img = self.transform_img(img)

        return img, mask


# Set the path to your dataset
dataset_path = 'DIRECTORY TO THE DATASET I PROVIDED IN README FILE VIA LINK'
custom_dataset = CustomDataset(
    root=dataset_path, transform_img=transform_img, transform_mask=transform_mask)
dataloader = DataLoader(custom_dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
model = DualSeg_res101(num_classes=1)
# Binary Cross Entropy loss for binary segmentation
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# from torchsummary import summary
# summary(model,input_size=(3, 128, 128))

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()

        outputs = model(images)
        m_out = outputs[0]
        t_out = targets
        # print(m_out,t_out)
        # Trim the lists to the minimum length
        min_length = min(len(m_out), len(t_out))
        m_out = m_out[:min_length]
        t_out = t_out[:min_length]

        # t_out = t_out.unsqueeze(1)  # Add a channel dimension for compatibility

        loss = criterion(m_out, t_out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # outputs, outputs_dsn = model(images)
        # outputs = torch.sigmoid(outputs[0])  # Applying sigmoid activation
        # loss1 = criterion(outputs, targets.to(torch.float32))
        # loss2 = criterion(outputs_dsn, targets.to(torch.float32))
        # loss = loss1 + loss2

    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

# Save the trained model
torch.save(model.state_dict(), 'segmentation_model.pth')
