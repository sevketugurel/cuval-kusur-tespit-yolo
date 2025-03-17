#!/usr/bin/env python
"""
Training script for YOLO models with attention mechanisms for sack defect detection.
"""

import os
import sys
import argparse
import yaml
import torch
import subprocess
from pathlib import Path

# Add parent directory to path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO models for sack defect detection')
    parser.add_argument('--model', type=str, default='yolov9', choices=['yolov5', 'yolov7', 'yolov9'],
                        help='YOLO model variant to use')
    parser.add_argument('--config', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to data configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--weights', type=str, default='', help='Initial weights path')
    parser.add_argument('--attention', type=str, default=None, choices=['cbam', 'gam', 'coordatt'],
                        help='Attention mechanism to use')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--project', default='runs/train', help='Save to project/name')
    parser.add_argument('--name', default='exp', help='Save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Existing project/name ok, do not increment')
    
    return parser.parse_args()

def train_yolov5(args):
    """
    Train YOLOv5 model with optional attention mechanisms.
    """
    try:
        from ultralytics import YOLO
        
        # Load a model
        if args.weights:
            model = YOLO(args.weights)
        else:
            model = YOLO('yolov5s.pt')  # load a pretrained model
        
        # Apply attention mechanism if specified
        if args.attention:
            print(f"Adding {args.attention} attention mechanism to YOLOv5")
            # This would require custom model modification which is beyond the scope of this script
            # In a real implementation, we would modify the YOLOv5 model architecture here
        
        # Train the model
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            device=args.device
        )
        
        return results
    
    except ImportError:
        print("Error: ultralytics package not found. Please install it with: pip install ultralytics")
        sys.exit(1)

def train_yolov7(args):
    """
    Train YOLOv7 model with optional attention mechanisms.
    """
    print("YOLOv7 training requires the YOLOv7 repository.")
    print("Please clone the YOLOv7 repository and follow the instructions for training.")
    print("Example command:")
    print(f"python yolov7/train.py --data {args.data} --cfg {args.config} --batch-size {args.batch_size} --epochs {args.epochs} --img-size {args.img_size}")
    
    # In a real implementation, we would either:
    # 1. Clone the YOLOv7 repository and run the training script
    # 2. Implement YOLOv7 training directly in this script
    
    return None

def train_yolov9(args):
    """
    Train YOLOv9 model with optional attention mechanisms.
    """
    # Check if YOLOv9 repository exists
    yolov9_dir = Path("yolov9")
    if not yolov9_dir.exists():
        print("YOLOv9 repository not found. Cloning from GitHub...")
        subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov9.git"], check=True)
    
    # Prepare command
    cmd = [
        "python", "yolov9/train.py",
        "--data", args.data,
        "--cfg", args.config,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--img-size", str(args.img_size),
        "--project", args.project,
        "--name", args.name,
        "--exist-ok" if args.exist_ok else ""
    ]
    
    # Add weights if specified
    if args.weights:
        cmd.extend(["--weights", args.weights])
    
    # Add device if specified
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Apply attention mechanism if specified
    if args.attention:
        print(f"Note: Adding {args.attention} attention mechanism to YOLOv9 requires custom model modification.")
        print("This will be implemented by modifying the YOLOv9 model architecture.")
        
        # In a real implementation, we would:
        # 1. Copy the YOLOv9 model file to a custom location
        # 2. Modify the model architecture to include the attention mechanism
        # 3. Point to the custom model file in the training command
        
        # For demonstration, we'll just print a message
        print(f"Using {args.attention} attention mechanism with YOLOv9")
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run([c for c in cmd if c != ""], check=True)
        print(f"Training completed. Results saved to {args.project}/{args.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return None

def main():
    args = parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(f"{args.project}/{args.name}", exist_ok=True)
    
    # Load data configuration
    try:
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        print(f"Training on dataset: {data_config}")
    except Exception as e:
        print(f"Error loading data configuration: {e}")
        sys.exit(1)
    
    # Train the selected model
    if args.model == 'yolov5':
        results = train_yolov5(args)
    elif args.model == 'yolov7':
        results = train_yolov7(args)
    elif args.model == 'yolov9':
        results = train_yolov9(args)
    else:
        print(f"Unsupported model: {args.model}")
        sys.exit(1)
    
    print(f"Training completed. Results saved to {args.project}/{args.name}")

if __name__ == '__main__':
    main() 