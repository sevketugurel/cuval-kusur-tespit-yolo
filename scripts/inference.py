#!/usr/bin/env python
"""
Inference script for real-time sack defect detection using YOLO models.
"""

import os
import sys
import argparse
import time
import cv2
import torch
import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with YOLO models for sack defect detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to image, video, or directory')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='Display results')
    parser.add_argument('--save-txt', action='store_true', help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='Save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='Save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='Do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default='runs/detect', help='Save results to project/name')
    parser.add_argument('--name', default='exp', help='Save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='Bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='Hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='Hide confidences')
    parser.add_argument('--webcam', action='store_true', help='Use webcam as source')
    parser.add_argument('--model-type', type=str, default='yolov9', choices=['yolov5', 'yolov7', 'yolov9'],
                        help='YOLO model variant to use')
    
    return parser.parse_args()

def run_inference_yolov5(args):
    """
    Run inference with YOLOv5 model.
    """
    try:
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO(args.model)
        
        # Set up source
        if args.webcam:
            source = 0  # Use webcam
        else:
            source = args.source
        
        # Run inference
        results = model.predict(
            source=source,
            conf=args.conf_thres,
            iou=args.iou_thres,
            imgsz=args.img_size,
            stream=True,
            show=args.view_img,
            save=not args.nosave,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            classes=args.classes,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            line_thickness=args.line_thickness,
            hide_labels=args.hide_labels,
            hide_conf=args.hide_conf,
            device=args.device
        )
        
        # Process results
        if args.webcam or isinstance(source, int):
            # Real-time processing for webcam or video
            for result in results:
                # Display FPS
                if args.view_img:
                    frame = result.orig_img
                    fps = 1.0 / (result.speed['inference'] + 1e-10)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display defect counts
                    if result.boxes is not None and len(result.boxes) > 0:
                        class_counts = {}
                        for box in result.boxes:
                            cls = int(box.cls.item())
                            cls_name = model.names[cls]
                            if cls_name in class_counts:
                                class_counts[cls_name] += 1
                            else:
                                class_counts[cls_name] = 1
                        
                        y_pos = 80
                        for cls_name, count in class_counts.items():
                            cv2.putText(frame, f"{cls_name}: {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            y_pos += 40
                    
                    cv2.imshow("Sack Defect Detection", frame)
                    
                    # Break loop on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        else:
            # Batch processing for images
            for result in tqdm(results, desc="Processing"):
                pass
        
        # Clean up
        if args.view_img:
            cv2.destroyAllWindows()
        
        return True
    
    except ImportError:
        print("Error: ultralytics package not found. Please install it with: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

def run_inference_yolov9(args):
    """
    Run inference with YOLOv9 model.
    """
    try:
        # Check if YOLOv9 repository exists
        yolov9_dir = Path("yolov9")
        if not yolov9_dir.exists():
            print("YOLOv9 repository not found. Cloning from GitHub...")
            subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov9.git"], check=True)
        
        # Prepare command
        cmd = [
            "python", "yolov9/detect.py",
            "--weights", args.model,
            "--source", "0" if args.webcam else args.source,
            "--img-size", str(args.img_size),
            "--conf-thres", str(args.conf_thres),
            "--iou-thres", str(args.iou_thres),
            "--project", args.project,
            "--name", args.name,
            "--exist-ok" if args.exist_ok else "",
            "--line-thickness", str(args.line_thickness),
            "--hide-labels" if args.hide_labels else "",
            "--hide-conf" if args.hide_conf else ""
        ]
        
        # Add device if specified
        if args.device:
            cmd.extend(["--device", args.device])
        
        # Add save options
        if args.nosave:
            cmd.append("--nosave")
        if args.save_txt:
            cmd.append("--save-txt")
        if args.save_conf:
            cmd.append("--save-conf")
        if args.save_crop:
            cmd.append("--save-crop")
        if args.view_img:
            cmd.append("--view-img")
        
        # Add classes if specified
        if args.classes:
            cmd.extend(["--classes"] + [str(c) for c in args.classes])
        
        # Run the command
        print(f"Running command: {' '.join([c for c in cmd if c != ''])}")
        subprocess.run([c for c in cmd if c != ""], check=True)
        
        print(f"Inference completed successfully. Results saved to {args.project}/{args.name}")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error during inference: {e}")
        return False
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

def main():
    args = parse_args()
    
    # Create output directories if they don't exist
    if not args.nosave:
        os.makedirs(f"{args.project}/{args.name}", exist_ok=True)
    
    # Run inference with the selected model
    if args.model_type == 'yolov5':
        success = run_inference_yolov5(args)
    elif args.model_type == 'yolov9':
        success = run_inference_yolov9(args)
    else:
        print(f"Inference not implemented for model type: {args.model_type}")
        success = False
    
    if success:
        print(f"Inference completed successfully. Results saved to {args.project}/{args.name}")
    else:
        print("Inference failed.")

if __name__ == '__main__':
    main() 