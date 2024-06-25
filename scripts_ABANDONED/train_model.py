from torch.utils.data import DataLoader
import torch
import torchvision
from dataset_class import LabelDetectionDataset
import torchvision.models.detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# set up dataset and dataloader
image_dir = './data/resized_images'
annotation_dir = './data/labelled_images'

dataset = LabelDetectionDataset(image_dir, annotation_dir, transforms=torchvision.transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load a pre-trained model and modify it
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
num_classes = 2  # Include background class
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move the model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Update the learning rate
    lr_scheduler.step()

    print(f"Epoch: {epoch}, Loss: {losses.item()}")

torch.save(model.state_dict(), 'fasterrcnn_model.pth')
