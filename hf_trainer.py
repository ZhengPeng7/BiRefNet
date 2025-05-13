from transformers import Trainer, TrainingArguments
import torch
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.birefnet_multiclass_pytorch import BiRefNetMultiClass
import numpy as np
import evaluate

# Custom collate_fn for Hugging Face Trainer
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch]).squeeze(1).long()
    return {'pixel_values': images, 'labels': labels}

# Dataset preparation
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

target_transform = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
    T.PILToTensor(),
])

train_dataset = VOCSegmentation(
    root='./data', year='2012', image_set='train', download=True,
    transform=transform,
    target_transform=target_transform
)

val_dataset = VOCSegmentation(
    root='./data', year='2012', image_set='val', download=True,
    transform=transform,
    target_transform=target_transform
)

# Model Initialization
model = BiRefNetMultiClass(num_classes=21)

# Hugging Face Trainer arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=100,
    learning_rate=1e-4,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Metrics function
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    predictions = torch.argmax(logits, dim=1).numpy()
    labels = labels
    
    metrics = metric.compute(predictions=predictions, references=labels, num_labels=21, ignore_index=255)
    return metrics

# Hugging Face Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

# Training
trainer.train()

# Save the best model
trainer.save_model("./birefnet_multiclass_best")
