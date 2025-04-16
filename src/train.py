# Below is a revised training script that also measures mAP on the validation set each epoch,
# logging the mAP to Weights & Biases (WandB). We use torchmetrics.detection.MeanAveragePrecision
# for the COCO-style calculation.

import argparse
import os
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models.detection import FasterRCNN
from typing import Any, Optional
from torchvision.models.resnet import resnet50
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.resnet import ResNet50_Weights
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

from datasets import COCODatasets

# Torchmetrics for mAP calculation
from torchmetrics.detection.mean_ap import MeanAveragePrecision



def _digit_anchorgen():
    anchor_sizes = ((12,), (24,), (36,), (56,), (104,))
    aspect_ratios = (0.5, 0.75, 1.0) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

def fasterrcnn_resnet50_fpn_v2_customanchors(
    *,
    weights: Optional[FasterRCNN_ResNet50_FPN_V2_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    norm_layer: nn.Module = nn.BatchNorm2d,
    **kwargs: Any,
) -> FasterRCNN:
    """
    Constructs an improved Faster R-CNN model with a ResNet-50-FPN backbone from `Benchmarking Detection
    Transfer Learning with Vision Transformers <https://arxiv.org/abs/2111.11429>`__ paper.

    .. betastatus:: detection module

    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.

    Args:
        weights (:class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.faster_rcnn.FasterRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights
        :members:
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    backbone = resnet50(weights=weights_backbone, progress=progress)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=norm_layer)
    rpn_anchor_generator = _digit_anchorgen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=norm_layer
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

# Helper function for visualization
def visualize_detections(image, targets=None, predictions=None, score_threshold=0.5):
    """
    Visualize detections on an image.

    Args:
        image: PyTorch tensor [C, H, W]
        targets: Dict with 'boxes' and 'labels' keys (ground truth)
        predictions: Dict with 'boxes', 'labels', and 'scores' keys (model predictions)
        score_threshold: Minimum confidence score for showing predictions

    Returns:
        PIL Image with drawn boxes
    """
    # Convert image tensor to numpy and then to PIL
    img_np = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    # Normalize to 0-255 range if needed
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    # Create PIL image for drawing
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)

    # Try to get a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    # Draw ground truth boxes (if provided)
    if targets is not None:
        boxes = targets['boxes'].cpu().numpy()
        labels = targets['labels'].cpu().numpy()

        # Draw each ground truth box in green
        for box, label in zip(boxes, labels):
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], 
                           outline="green", width=2)
            draw.text((box[0], box[1]), f"GT: {int(label)-1}", 
                      fill="green", font=font)

    # Draw predicted boxes (if provided)
    if predictions is not None:
        pred_boxes = predictions['boxes'].cpu().numpy()
        pred_labels = predictions['labels'].cpu().numpy()
        pred_scores = predictions['scores'].cpu().numpy()

        # Filter predictions by score threshold
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # Draw each prediction box in red
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], 
                           outline="red", width=2)
            draw.text((box[0], box[1] - 15), f"{int(label)-1}", 
                      fill="red", font=font)

    return pil_img

# Convert PIL image to wandb Image
def get_wandb_image(pil_img):
    return wandb.Image(pil_img)


def evaluate_mAP(model, dataloader, device, score_threshold=0.5):
    """
    Evaluate the model on the given dataloader using torchmetrics MeanAveragePrecision.
    This function returns the computed metric (a dict with keys like 'map', 'map_50', etc.).
    """
    model.eval()

    metric = MeanAveragePrecision(box_format='xyxy')  # default COCO style [0.5:0.95]

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validating for mAP', leave=False):
            images = list(img.to(device) for img in images)
            target_list = []
            for t in targets:
                target_list.append({
                    'boxes': t['boxes'].to(device),
                    'labels': t['labels'].to(device)
                })

            predictions = model(images)

            # Convert predictions to correct format for metric
            pred_list = []
            for pred in predictions:
                # filter out low-score boxes
                keep = pred['scores'] >= score_threshold
                filtered_boxes = pred['boxes'][keep]
                filtered_scores = pred['scores'][keep]
                filtered_labels = pred['labels'][keep]
                pred_list.append({
                    'boxes': filtered_boxes,
                    'scores': filtered_scores,
                    'labels': filtered_labels
                })

            metric.update(pred_list, target_list)

    results = metric.compute()
    model.train()  # set back to train mode
    return results

# --- Main Script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the COCO dataset with WandB logging, including mAP measurement')
    # Data and Path Arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the COCO dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the trained model and logs')
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for the optimizer')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'], help='Optimizer to use for training')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'onecycle'], help='Learning rate scheduler to use')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of dataloader workers')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes (including background)') # Match COCODatasets and model
    parser.add_argument('--trainable_backbone_layers', type=int, default=3, help='Number of trainable layers in backbone')
    parser.add_argument('--min_size', type=int, default=150, help='Minimum size of the image for training')
    parser.add_argument('--max_size', type=int, default=300, help='Maximum size of the image for training')
    # Visualization parameters
    parser.add_argument('--vis_interval', type=int, default=100, help='Visualization interval (iterations)')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Confidence score threshold for visualizations/mAP')
    # WandB Arguments
    parser.add_argument('--wandb_project', type=str, default='vrdl-hw2', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='jayinnn', help='WandB entity (username or team)') # Optional: Defaults to your default entity
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name (optional)')

    args = parser.parse_args()

    # Check if num_classes is appropriate
    print(f"Training with {args.num_classes} classes (including background)")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args) # Log all hyperparameters from args
    )
    print("WandB initialized.")

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    print("Transforms defined.")

    # --- Datasets and DataLoaders ---
    try:
        train_dataset = COCODatasets(root=args.data_dir, mode='train', transform=train_transform)
        valid_dataset = COCODatasets(root=args.data_dir, mode='valid', transform=valid_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure 'datasets.py' with COCODatasets class is available or replace placeholder.")
        exit()

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn # Use custom collate_fn for detection tasks
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )
    print("Datasets and DataLoaders created.")

    # --- Model Definition ---
    model = fasterrcnn_resnet50_fpn_v2_customanchors(
        trainable_backbone_layers=args.trainable_backbone_layers,
        # weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        weights_backbone=ResNet50_Weights.DEFAULT,
        # norm_layer=tv.ops.FrozenBatchNorm2d, # FrozenBatchNorm2d,
        norm_layer=nn.BatchNorm2d, 
        progress=True,
        min_size=args.min_size,
        max_size=args.max_size,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (number of classes + 1 for background)
    model.roi_heads.box_predictor = tv.models.detection.faster_rcnn.FastRCNNPredictor(in_features, args.num_classes)

    model.to(device)
    print(f"Model loaded and moved to {device}.")

    # --- Optimizer and Scheduler ---
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=0.9,
        )
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=0.0005
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    if args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,  # Peak learning rate
            total_steps=args.epochs * len(train_loader),
            pct_start=0.1,  # Spend 10% of training warming up
            div_factor=10,  # Start at max_lr/10
            final_div_factor=100,  # End at max_lr/1000
            anneal_strategy='cos'  # Cosine annealing
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * len(train_loader),
            eta_min=args.lr * 0.001
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    print("Optimizer and Scheduler created.")

    print("Starting training...")
    global_step = 0

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", unit="batch")

        for batch_idx, (images, targets) in enumerate(pbar_train):
            images = list(image.to(device) for image in images)
            targets = [ {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            loss_item = losses.item()
            train_loss_accum += loss_item

            log_dict = {
                "train/batch_loss": loss_item,
                "train/learning_rate": current_lr,
                "epoch": epoch + 1,
                "batch": batch_idx,
                "global_step": global_step
            }

            # Add individual losses if needed
            for loss_name, value in loss_dict.items():
                log_dict[f"train/loss_{loss_name}"] = value.item()

            # Visualization every args.vis_interval iterations
            if global_step % args.vis_interval == 0:
                model.eval()
                with torch.no_grad():
                    sample_img = images[0]
                    sample_target = targets[0]
                    predictions = model([sample_img])[0]
                    vis_img = visualize_detections(
                        sample_img.cpu(),
                        sample_target,
                        predictions,
                        score_threshold=args.score_threshold
                    )
                    log_dict["train/visualization"] = get_wandb_image(vis_img)

                model.train()

            wandb.log(log_dict)
            pbar_train.set_postfix(loss=loss_item, lr=current_lr)
            global_step += 1

   

        avg_train_loss = train_loss_accum / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch + 1})

        # --- Validation Phase (Loss) ---
        model.eval()
        val_loss_accum = 0.0
        val_step = 0
        pbar_valid = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validate]", unit="batch")

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(pbar_valid):
                images = list(img.to(device) for img in images)
                targets = [ {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                # In order to get the validation loss, we temporarily switch to train mode
                # for the forward pass with targets. However, we do NOT backprop.
                model.train()
                loss_dict = model(images, targets)
                model.eval()

                losses = sum(loss for loss in loss_dict.values())
                loss_item = losses.item()
                val_loss_accum += loss_item

                val_log_dict = {
                    "val/batch_loss": loss_item,
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "global_step": global_step
                }

                for loss_name, value in loss_dict.items():
                    val_log_dict[f"val/loss_{loss_name}"] = value.item()

                if val_step % args.vis_interval == 0:
                    sample_img = images[0]
                    sample_target = targets[0]
                    predictions = model([sample_img])[0]

                    vis_img = visualize_detections(
                        sample_img.cpu(),
                        sample_target,
                        predictions,
                        score_threshold=args.score_threshold
                    )

                    val_log_dict["val/visualization"] = get_wandb_image(vis_img)

                wandb.log(val_log_dict)
                pbar_valid.set_postfix(loss=loss_item)
                val_step += 1

        avg_val_loss = val_loss_accum / len(valid_loader)
        print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val/epoch_loss": avg_val_loss, "epoch": epoch + 1})

        # --- Validation Phase (mAP) ---
        # Evaluate full dataloader for mAP
        mAP_results = evaluate_mAP(model, valid_loader, device, score_threshold=args.score_threshold)
        # mAP_results is a dictionary, typical keys: 'map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large'
        # We'll log the ones that exist.
        map_log = {}
        for k, v in mAP_results.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                map_log[f"val/{k}"] = v.item()
            else:
                # If it's not a single scalar, skip or handle differently
                pass

        # Also log them to wandb
        map_log["epoch"] = epoch + 1
        wandb.log(map_log)

        print("Validation mAP:")
        for key, value in map_log.items():
            if key.startswith("val/"):
                print(f"  {key}: {value:.4f}")

        # --- Save Model Checkpoint ---
        checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
        wandb.log({"checkpoint_path": checkpoint_path, "epoch": epoch + 1})

    print("Training finished.")
    wandb.finish()
    print("WandB run finished.")