import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

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
        # Encoder
        enc1 = self.encoder(x)
        # Middle
        middle = self.middle(enc1)
        # Decoder
        dec1 = self.decoder(middle)

        return dec1


# Transformation for both input images and masks
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Transformation for resizing masks to match model output size
transform_mask = transforms.Compose([
    transforms.Resize((64, 64)),  # Adjusted to match the model's output size
    transforms.ToTensor()
])

# Dataset class for your custom dataset


class CustomDataset(Dataset):
    def __init__(self, root, transform, transform_mask):
        self.dataset = ImageFolder(root, transform=transform)
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        mask_path = self.dataset.imgs[idx][0].replace("images", "gt")

        # Load PNG mask images and resize
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        mask = self.transform_mask(mask)

        return img, mask


# Set the path to your dataset
dataset_path = 'DIRECTORY TO THE DATASET I PROVIDED IN README FILE VIA LINK'
custom_dataset = CustomDataset(
    root=dataset_path, transform=transform, transform_mask=transform_mask)

# DataLoader
dataloader = DataLoader(custom_dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
model = UNet()
# Binary Cross Entropy loss for binary segmentation
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

# Save the trained model
torch.save(model.state_dict(), 'segmentation_model.pth')
