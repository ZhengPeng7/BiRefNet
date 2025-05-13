import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from models.birefnet_multiclass_pytorch import BiRefNetMultiClass
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# Dataset
train_dataset = VOCSegmentation(
    root='./data', year='2012', image_set='train', download=not os.path.exists('./data/VOCdevkit'),
    transform=T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ]),
    target_transform=T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor(),
    ])
)
val_dataset = VOCSegmentation(
    root='./data', year='2012', image_set='val', download=not os.path.exists('./data/VOCdevkit'),
    transform=T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ]),
    target_transform=T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor(),
    ])
)
# DataLoader
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Dynamically set the number of classes based on the dataset
num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 21
model = BiRefNetMultiClass(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
model.train()
for epoch in range(50):
    # Training phase
    model.train()
    train_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]") as t:
        for images, labels in t:
            images, labels = images.to(device), labels.squeeze(1).long().to(device)

            optimizer.zero_grad()
            outputs = model(images, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            t.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]") as t:
            for images, labels in t:
                images, labels = images.to(device), labels.squeeze(1).long().to(device)

                outputs = model(images, labels=labels)
                loss = outputs['loss']

                val_loss += loss.item()
                t.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch+1}, Average Validation Loss: {avg_val_loss:.4f}')

# Save model
torch.save(model.state_dict(), "./birefnet_multiclass_50epochs.pth")
print("Model has been saved successfully to './birefnet_multiclass.pth'")

# Create a test dataset using the 'val' split
test_dataset = VOCSegmentation(
    root='./data', year='2012', image_set='val', download=not os.path.exists('./data/VOCdevkit'),
    transform=T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ]),
    target_transform=T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor(),
    ])
)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# Set model to evaluation mode
model.eval()

count = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        pred_logits = outputs['logits']
        pred_segs = torch.argmax(pred_logits, dim=1)  # shape: (batch, height, width)

        for i in range(images.size(0)):
            img_disp = images[i].cpu().permute(1, 2, 0).numpy()
            gt_seg = labels[i].squeeze(0).cpu().numpy()
            pred_seg = pred_segs[i].cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_disp)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(gt_seg, cmap=plt.cm.get_cmap('tab20'))
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred_seg)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(f"output_{count}.png")
            plt.close()

            count += 1
            if count >= 41:
                break
        if count >= 41:
            break
        