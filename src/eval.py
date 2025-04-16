import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
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
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
from datasets import COCODatasets, COCOTestDatasets
from sklearn.cluster import DBSCAN

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


def evaluate_model(model, valid_loader, device, score_threshold=0.5):
    """
    Evaluate the model on validation set using TorchMetrics MeanAveragePrecision
    """
    # Initialize TorchMetrics mAP metric
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
    
    print("Evaluating model on validation set...")
    model.eval()
    
    with torch.no_grad():
        for images, targets in tqdm(valid_loader):
            images = list(image.to(device) for image in images)
            
            # Get model predictions
            predictions = model(images)
            
            # Format targets for torchmetrics
            target_list = []
            for target in targets:
                target_list.append({
                    'boxes': target['boxes'].to(device),
                    'labels': target['labels'].to(device)
                })
            
            # Format predictions for torchmetrics
            pred_list = []
            for pred in predictions:
                # Filter predictions based on score threshold
                keep = pred['scores'] > score_threshold
                pred_list.append({
                    'boxes': pred['boxes'][keep],
                    'scores': pred['scores'][keep],
                    'labels': pred['labels'][keep]
                })
            
            # Update metric
            metric.update(pred_list, target_list)
    
    # Compute mAP
    result = metric.compute()
    
    # Print results
    print("\nValidation Results:")
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"{key}: {value.item():.4f}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: {value.cpu().numpy()}")
    
    return result

def predict_on_test_set(model, test_loader, device, score_threshold=0.5):
    """
    Make predictions on the test set
    """
    print("Making predictions on test set...")
    model.eval()
    
    all_predictions = []
    all_image_ids = []
    num_batches = 10
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader):
            # Get image ids from the filenames (assuming filenames are like "1.png", "2.png", etc.)
            all_image_ids.extend(image_ids)
            
            # Move images to device
            images = list(image.to(device) for image in images)
            
            # Get model predictions
            predictions = model(images)
            all_predictions.extend(predictions)
    
    return all_predictions, all_image_ids

def prepare_test_predictions_for_submission(all_predictions, image_ids, score_threshold=0.5, 
                                           nms_threshold=0.2, apply_clustering=True):
    """
    Prepare predictions for Task 1 (COCO format) with spatial clustering to filter outliers.
    
    Args:
        all_predictions: List of prediction dictionaries from the model
        image_ids: List of image IDs corresponding to predictions
        score_threshold: Confidence threshold for keeping predictions
        nms_threshold: IoU threshold for NMS to remove duplicate detections
        apply_clustering: Whether to apply spatial clustering to filter outliers
    """
    coco_predictions = []
    print(f"Processing {len(all_predictions)} predictions across {len(image_ids)} images")
    
    for image_id, pred in zip(image_ids, all_predictions):
        # Get predictions
        boxes = pred['boxes'].cpu()
        scores = pred['scores'].cpu()
        labels = pred['labels'].cpu()
        
        # Initial filtering by confidence score
        keep = scores > score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Apply NMS for each class separately (more stringent than the model's built-in NMS)
        final_boxes = []
        final_scores = []
        final_labels = []
        
        # Process each digit class separately for NMS
        for class_id in range(1, 11):  # Assuming 10 digit classes (0-9) with IDs 1-10
            class_mask = labels == class_id
            if not class_mask.any():
                continue
                
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            # Apply NMS
            keep_indices = tv.ops.nms(class_boxes, class_scores, nms_threshold)
            
            # Save the kept predictions
            final_boxes.append(class_boxes[keep_indices])
            final_scores.append(class_scores[keep_indices])
            final_labels.append(torch.full((len(keep_indices),), class_id))
        
        if not final_boxes:  # No detections after NMS
            continue
            
        # Concatenate the filtered predictions from all classes
        boxes = torch.cat(final_boxes)
        scores = torch.cat(final_scores)
        labels = torch.cat(final_labels)
        
        # Apply spatial clustering to filter outliers
        if apply_clustering and len(boxes) > 2:  # Only cluster if there are enough boxes
            import numpy as np
            from sklearn.cluster import DBSCAN
            from collections import Counter
            
            # Get center points of bounding boxes
            centers = torch.zeros((len(boxes), 2))
            for i, box in enumerate(boxes):
                centers[i, 0] = (box[0] + box[2]) / 2  # x-center
                centers[i, 1] = (box[1] + box[3]) / 2  # y-center

            distances = []
            for i in range(len(centers) - 1):
                for j in range(i + 1, len(centers)):
                    distances.append(np.linalg.norm(centers[i] - centers[j]))

            # Set cluster distance based on mean pairwise distance
            if distances:
                cluster_distance = np.min(distances) * 2
                print(f"Image {image_id}: Cluster distance = {cluster_distance:.2f}")
            
            # # Calculate the average width and height of boxes
            # widths = boxes[:, 2] - boxes[:, 0]
            # heights = boxes[:, 3] - boxes[:, 1]
            # avg_size = torch.mean(torch.cat([widths, heights])).item()
            
            # # Set cluster distance based on average box size
            # cluster_distance = avg_size * 3  # Allow digits to be 3x their size apart
            
            # Apply DBSCAN clustering to identify groups of digits
            clustering = DBSCAN(eps=cluster_distance, min_samples=1).fit(centers.numpy())
            labels_cluster = clustering.labels_
            
            # Count number of elements in each cluster
            cluster_counts = Counter(labels_cluster)
            
            # Find the largest cluster
            largest_cluster = cluster_counts.most_common(1)[0][0]
            
            # Filter to keep only boxes in the largest cluster
            cluster_mask = torch.tensor([bool(label == largest_cluster) for label in labels_cluster])
            
            # Get the percentage of boxes kept after clustering
            pct_kept = 100 * cluster_mask.sum().item() / len(cluster_mask)
            
            # Only apply clustering filter if it doesn't remove too many detections
            # This prevents over-filtering in case of legitimate spread-out digits
            if pct_kept >= 50:  # Keep clustering result if at least 50% of boxes remain
                boxes = boxes[cluster_mask]
                scores = scores[cluster_mask]
                labels = labels[cluster_mask]
                print(f"Image {image_id}: Cluster distance = {cluster_distance:.2f}, kept {pct_kept:.1f}% of detections")
            else:
                print(f"Image {image_id}: Clustering would remove too many boxes ({pct_kept:.1f}% kept), skipping")
        
        # Add Task 1 predictions to COCO format
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            
            coco_predictions.append({
                'image_id': int(image_id),
                'category_id': int(labels[i]),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(scores[i])
            })
    
    return coco_predictions


def generate_task2_from_json(json_file, score_threshold=0.5):
    """
    Generate Task 2 predictions (pred.csv) from Task 1 predictions (pred.json).
    
    Args:
        json_file: Path to the Task 1 predictions (COCO format JSON)
        min_cluster_pct: Minimum percentage of boxes to keep after clustering
        
    Returns:
        List of dictionaries containing image_id and pred_label for Task 2
    """
    import json
    import numpy as np
    from collections import defaultdict, Counter
    from sklearn.cluster import DBSCAN
    
    print(f"Generating Task 2 predictions from {json_file}")
    
    # Load Task 1 predictions
    with open(json_file, 'r') as f:
        coco_predictions = json.load(f)
    
    # Group predictions by image_id
    predictions_by_image = defaultdict(list)
    for pred in coco_predictions:
        predictions_by_image[pred['image_id']].append(pred)
    
    # Process each image to generate Task 2 predictions
    task2_predictions = []
    
    for image_id, preds in predictions_by_image.items():
        # If no predictions for this image
        if len(preds) == 0:
            task2_predictions.append({
                'image_id': image_id,
                'pred_label': -1
            })
            continue
        
        # Extract bounding boxes, scores, and labels
        boxes = []
        labels = []
        scores = []
        
        for pred in preds:
            if pred['score'] < score_threshold:
                continue
            x, y, w, h = pred['bbox']
            # Convert from COCO format [x,y,w,h] to [x1,y1,x2,y2]
            boxes.append([x, y, x+w, y+h])
            labels.append(pred['category_id'])
            scores.append(pred['score'])

        if len(boxes) == 0:
            task2_predictions.append({
                'image_id': image_id,
                'pred_label': -1
            })
            continue

        boxes = np.array(boxes)
        labels = np.array(labels)
        scores = np.array(scores)
        
        # Use clustering to identify outliers if there are multiple digits
        if len(boxes) > 1:
            # Get center points of bounding boxes
            centers = np.zeros((len(boxes), 2))
            for i, box in enumerate(boxes):
                centers[i, 0] = (box[0] + box[2]) / 2  # x-center
                centers[i, 1] = (box[1] + box[3]) / 2  # y-center
            
            distances = []
            for i in range(len(centers) - 1):
                for j in range(i + 1, len(centers)):
                    distances.append(np.linalg.norm(centers[i] - centers[j]))

            # Set cluster distance based on mean pairwise distance
            if distances:
                cluster_distance = np.min(distances) * 2
            # # Calculate the average width and height of boxes
            # widths = boxes[:, 2] - boxes[:, 0]
            # heights = boxes[:, 3] - boxes[:, 1]
            # avg_size = np.mean(np.concatenate([widths, heights]))
            
            # # Set cluster distance based on average box size
            # cluster_distance = avg_size * 3  # Allow digits to be 3x their size apart
            
            print(f"Image {image_id}: Cluster distance = {cluster_distance:.2f}")
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=cluster_distance, min_samples=1).fit(centers)
            cluster_labels = clustering.labels_
            
            # Find the largest cluster
            cluster_counts = Counter(cluster_labels)
            largest_cluster = cluster_counts.most_common(1)[0][0]
            
            # Filter to keep only boxes in the largest cluster
            cluster_mask = np.array([bool(label == largest_cluster) for label in cluster_labels])
            
            # Get the percentage of boxes kept after clustering
            pct_kept = 100 * np.sum(cluster_mask) / len(cluster_mask)
            boxes = boxes[cluster_mask]
            labels = labels[cluster_mask]
            # Only apply clustering filter if it doesn't remove too many detections
            # if pct_kept > 60:  # Keep clustering result if at least 60% of boxes remain
                
                # print(f"Image {image_id}: Kept {pct_kept:.1f}% of detections")
            # else:
                # print(f"Image {image_id}: Clustering would remove too many boxes ({pct_kept:.1f}% kept), skipping")
        
        # Sort remaining detections by x-coordinate
        sorted_indices = np.argsort(boxes[:, 0])
        sorted_labels = labels[sorted_indices]
        
        # Convert labels to digit values (category_id - 1)
        digits = [int(label - 1) for label in sorted_labels]
        
        # Combine digits into a single number
        number = int(''.join(map(str, digits))) if digits else -1
        
        task2_predictions.append({
            'image_id': image_id,
            'pred_label': number
        })
    
    # Check if any images might be missing from the Task 1 predictions
    # In this case, we need to add entries with -1 prediction
    all_image_ids = set(range(1, max(predictions_by_image.keys()) + 1))
    missing_image_ids = all_image_ids - set(predictions_by_image.keys())
    
    for image_id in missing_image_ids:
        task2_predictions.append({
            'image_id': image_id,
            'pred_label': -1
        })
    
    # Sort by image_id for consistency
    task2_predictions.sort(key=lambda x: x['image_id'])
    
    return task2_predictions


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if args.custom_anchors:
        # Load model
        model = fasterrcnn_resnet50_fpn_v2_customanchors(
            min_size=args.min_size,
            max_size=args.max_size,
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one (number of classes + 1 for background)
        model.roi_heads.box_predictor = tv.models.detection.faster_rcnn.FastRCNNPredictor(in_features, args.num_classes)
    else:
        model = tv.models.detection.fasterrcnn_resnet50_fpn_v2(
            num_classes=args.num_classes,
        )
        
    
    # Load model weights
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    
    print(f"Model loaded from {args.checkpoint_path}")

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Evaluate on validation set if requested
    if args.eval_validation:
        # Load validation dataset
        valid_dataset = COCODatasets(root=args.data_dir, mode='valid', transform=transform)
        
        # Create validation data loader
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Evaluate model
        results = evaluate_model(model, valid_loader, device, args.score_threshold)
        
        # Save results
        results_dict = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    results_dict[key] = value.item()
                else:
                    results_dict[key] = value.cpu().numpy().tolist()
        
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"Validation results saved to {args.output_file}")
    
    # Make predictions on test set if requested
    if args.test_predictions:
        # Load test dataset
        test_dataset = COCOTestDatasets(root=args.data_dir, mode='test', transform=transform)
        
        # Create test data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Step 1: Generate Task 1 predictions (pred.json)
        print("Step 1: Generating Task 1 predictions (pred.json)")
        all_predictions, image_ids = predict_on_test_set(model, test_loader, device, args.score_threshold)
        
        # Prepare Task 1 predictions in COCO format with clustering
        coco_predictions = prepare_test_predictions_for_submission(
            all_predictions, image_ids, args.score_threshold, args.nms_threshold, 
            apply_clustering=args.apply_clustering
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.pred_json) if os.path.dirname(args.pred_json) else '.', exist_ok=True)
        
        # Save Task 1 predictions (COCO format)
        with open(args.pred_json, 'w') as f:
            json.dump(coco_predictions, f)
        
        print(f"Task 1 predictions saved to {args.pred_json}")
        
        # Step 2: Generate Task 2 predictions (pred.csv) from Task 1 predictions
        print("Step 2: Generating Task 2 predictions (pred.csv) from Task 1 predictions")
        task2_predictions = generate_task2_from_json(args.pred_json, args.score_threshold)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.pred_csv) if os.path.dirname(args.pred_csv) else '.', exist_ok=True)
        
        # Save Task 2 predictions (CSV format)
        pd.DataFrame(task2_predictions).to_csv(args.pred_csv, index=False)
        
        print(f"Task 2 predictions saved to {args.pred_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a Faster R-CNN model and make test predictions')
    # Data and Path Arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset root directory')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--output_file', type=str, default='./eval_results.json', help='Path to save evaluation results')
    # Model Parameters
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes (including background)')
    parser.add_argument('--min_size', type=int, default=200, help='Minimum size of the image for training')
    parser.add_argument('--max_size', type=int, default=400, help='Maximum size of the image for training')
    # Evaluation Parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Confidence score threshold for predictions')
    parser.add_argument('--nms_threshold', type=float, default=0.2, help='IoU threshold for NMS')
    # Clustering Parameters
    parser.add_argument('--apply_clustering', action='store_true', help='Apply spatial clustering to Task 1')
    parser.add_argument('--min_cluster_pct', type=float, default=60.0, 
                      help='Minimum percentage of boxes to keep after clustering')
    # Evaluation Modes
    parser.add_argument('--eval_validation', action='store_true', help='Evaluate on validation set')
    parser.add_argument('--test_predictions', action='store_true', help='Make predictions on test set')
    # Submission Files
    parser.add_argument('--pred_json', type=str, default='./pred.json', help='Path to save Task 1 predictions in COCO format')
    parser.add_argument('--pred_csv', type=str, default='./pred.csv', help='Path to save Task 2 predictions')
    # Flag to generate Task 2 from existing JSON
    parser.add_argument('--generate_csv_only', action='store_true', help='Only generate pred.csv from existing pred.json')
    parser.add_argument('--custom_anchors', action='store_true', help='Use custom anchors for the model')
    
    args = parser.parse_args()
    
    # Handle the special case where we only want to generate the CSV from existing JSON
    if args.generate_csv_only:
        if not os.path.exists(args.pred_json):
            print(f"Error: Cannot find pred.json at {args.pred_json}")
            exit(1)
        
        task2_predictions = generate_task2_from_json(args.pred_json, args.score_threshold)
        os.makedirs(os.path.dirname(args.pred_csv) if os.path.dirname(args.pred_csv) else '.', exist_ok=True)
        pd.DataFrame(task2_predictions).to_csv(args.pred_csv, index=False)
        print(f"Task 2 predictions saved to {args.pred_csv}")
        exit(0)
    
    # Ensure at least one mode is selected
    if not (args.eval_validation or args.test_predictions):
        print("Warning: No action selected. Please use --eval_validation or --test_predictions")
        exit(1)
        
    main(args)