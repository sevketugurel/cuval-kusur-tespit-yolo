#!/usr/bin/env python
"""
Training script for YOLO models for sack defect detection.
"""

import os
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for sack defect detection')
    parser.add_argument('--data', type=str, default='configs/data.yaml', 
                       help='Path to data configuration file')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, 
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, 
                       help='Image size')
    parser.add_argument('--project', default='runs/train', 
                       help='Save to project/name')
    parser.add_argument('--name', default='v1_initial', 
                       help='Save to project/name')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Model oluştur
    model = YOLO('yolov8n.pt')
    
    # Eğitimi başlat
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=args.project,
        name=args.name
    )

if __name__ == '__main__':
    main() 