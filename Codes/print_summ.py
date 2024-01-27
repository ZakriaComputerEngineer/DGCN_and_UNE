#   THIS PROGRAM PRINTS THE COMPLETE DETAIL ABOUT THE MODEL IN TABULAR FORM SUCH THAT EACH LAYER IN SHOWN ACCORDING TO GIVEN INPUT IMAGE SHAPE AND SIZE


from torchsummary import summary
from libs.models import DualSeg_res101
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Transformation for both input images and masks
transform_img = transforms.Compose([
    transforms.Resize((508, 508)),
    transforms.ToTensor()
])

# Transformation for resizing masks to match model output size
transform_mask = transforms.Compose([
    transforms.Resize((64, 64)),
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

# DataLoader
dataloader = DataLoader(custom_dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
model = DualSeg_res101(num_classes=1)
# Binary Cross Entropy loss for binary segmentation
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

summary(model, input_size=(3, 128, 128))

# Count layers and parameters
num_layers = len(list(model.parameters()))
num_parameters = sum(p.numel() for p in model.parameters())

print(f"Number of layers: {num_layers}")
print(f"Number of parameters: {num_parameters}")
