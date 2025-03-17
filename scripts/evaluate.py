#!/usr/bin/env python
"""
Evaluation script for YOLO models on sack defect detection.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLO models for sack defect detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='Save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='Save results to *.json')
    parser.add_argument('--project', default='runs/val', help='Save to project/name')
    parser.add_argument('--name', default='exp', help='Save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Existing project/name ok, do not increment')
    parser.add_argument('--model-type', type=str, default='yolov9', choices=['yolov5', 'yolov7', 'yolov9'],
                        help='YOLO model variant to use')
    
    return parser.parse_args()

def evaluate_yolov5(args):
    """
    Evaluate YOLOv5 model performance.
    """
    try:
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO(args.model)
        
        # Run validation
        results = model.val(
            data=args.data,
            conf=args.conf_thres,
            iou=args.iou_thres,
            imgsz=args.img_size,
            batch=args.batch_size,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_json=args.save_json,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            device=args.device
        )
        
        # Extract metrics
        metrics = results.box
        
        # Print results
        print("\n--- Evaluation Results ---")
        print(f"mAP@0.5: {metrics.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.map:.4f}")
        print(f"Precision: {metrics.p:.4f}")
        print(f"Recall: {metrics.r:.4f}")
        print(f"F1-Score: {2 * (metrics.p * metrics.r) / (metrics.p + metrics.r + 1e-16):.4f}")
        
        # Plot confusion matrix if available
        if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
            plot_confusion_matrix(metrics.confusion_matrix.matrix, 
                                  metrics.names, 
                                  f"{args.project}/{args.name}/confusion_matrix.png")
        
        return metrics
    
    except ImportError:
        print("Error: ultralytics package not found. Please install it with: pip install ultralytics")
        sys.exit(1)

def evaluate_yolov9(args):
    """
    Evaluate YOLOv9 model performance.
    """
    # Check if YOLOv9 repository exists
    yolov9_dir = Path("yolov9")
    if not yolov9_dir.exists():
        print("YOLOv9 repository not found. Cloning from GitHub...")
        subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov9.git"], check=True)
    
    # Prepare command
    cmd = [
        "python", "yolov9/val.py",
        "--data", args.data,
        "--weights", args.model,
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.img_size),
        "--conf-thres", str(args.conf_thres),
        "--iou-thres", str(args.iou_thres),
        "--project", args.project,
        "--name", args.name,
        "--exist-ok" if args.exist_ok else ""
    ]
    
    # Add device if specified
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Add save options
    if args.save_txt:
        cmd.append("--save-txt")
    if args.save_conf:
        cmd.append("--save-conf")
    if args.save_json:
        cmd.append("--save-json")
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run([c for c in cmd if c != ""], check=True, capture_output=True, text=True)
        output = result.stdout
        
        # Parse metrics from output
        metrics = parse_yolov9_metrics(output, f"{args.project}/{args.name}")
        
        # Print results
        print("\n--- Evaluation Results ---")
        print(f"mAP@0.5: {metrics['map50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['map']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-16):.4f}")
        
        return metrics
    
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        print(f"Error output: {e.stderr}")
        return None

def parse_yolov9_metrics(output, output_dir):
    """
    Parse YOLOv9 metrics from command output.
    """
    metrics = {
        'map': 0.0,
        'map50': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'names': ['tear', 'print', 'stitch']  # Default class names
    }
    
    # Try to parse metrics from output
    for line in output.split('\n'):
        if 'Average Precision' in line and '@0.5' in line and '@0.5:0.95' not in line:
            try:
                metrics['map50'] = float(line.split('=')[-1].strip())
            except:
                pass
        elif 'Average Precision' in line and '@0.5:0.95' in line:
            try:
                metrics['map'] = float(line.split('=')[-1].strip())
            except:
                pass
        elif 'Precision' in line and 'Recall' not in line:
            try:
                metrics['precision'] = float(line.split('=')[-1].strip())
            except:
                pass
        elif 'Recall' in line:
            try:
                metrics['recall'] = float(line.split('=')[-1].strip())
            except:
                pass
    
    # Try to load class names from data.yaml
    try:
        data_yaml = list(Path(output_dir).glob('*.yaml'))[0]
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                metrics['names'] = data['names']
    except:
        pass
    
    return metrics

def plot_confusion_matrix(confusion_matrix, class_names, save_path):
    """
    Plot and save confusion matrix.
    """
    try:
        # Create a DataFrame for better visualization
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def calculate_fps(args):
    """
    Calculate inference speed (FPS) of the model.
    """
    if args.model_type == 'yolov5':
        try:
            from ultralytics import YOLO
            import time
            
            # Load the model
            model = YOLO(args.model)
            
            # Create a dummy input tensor
            device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
            dummy_input = torch.zeros((1, 3, args.img_size, args.img_size), device=device)
            
            # Warm-up runs
            for _ in range(10):
                _ = model(dummy_input)
            
            # Measure inference time
            num_runs = 100
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(dummy_input)
            end_time = time.time()
            
            # Calculate FPS
            fps = num_runs / (end_time - start_time)
            
            print(f"Inference Speed: {fps:.2f} FPS")
            
            return fps
        
        except ImportError:
            print("Error: ultralytics package not found. Please install it with: pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            print(f"Error calculating FPS: {e}")
            return None
    
    elif args.model_type == 'yolov9':
        try:
            # Check if YOLOv9 repository exists
            yolov9_dir = Path("yolov9")
            if not yolov9_dir.exists():
                print("YOLOv9 repository not found. Cloning from GitHub...")
                subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov9.git"], check=True)
            
            # Prepare command for speed test
            cmd = [
                "python", "yolov9/detect.py",
                "--weights", args.model,
                "--img-size", str(args.img_size),
                "--conf-thres", str(args.conf_thres),
                "--iou-thres", str(args.iou_thres),
                "--device", args.device if args.device else '0',
                "--benchmark", "100"  # Run 100 iterations for benchmark
            ]
            
            # Run the command
            print(f"Running FPS benchmark: {' '.join(cmd)}")
            result = subprocess.run([c for c in cmd if c != ""], check=True, capture_output=True, text=True)
            output = result.stdout
            
            # Parse FPS from output
            fps = None
            for line in output.split('\n'):
                if 'FPS:' in line:
                    try:
                        fps = float(line.split('FPS:')[1].strip().split()[0])
                        break
                    except:
                        pass
            
            if fps is not None:
                print(f"Inference Speed: {fps:.2f} FPS")
            else:
                print("Could not determine FPS from output")
            
            return fps
        
        except subprocess.CalledProcessError as e:
            print(f"Error during FPS calculation: {e}")
            print(f"Error output: {e.stderr}")
            return None
        except Exception as e:
            print(f"Error calculating FPS: {e}")
            return None
    
    else:
        print(f"FPS calculation not implemented for model type: {args.model_type}")
        return None

def main():
    args = parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(f"{args.project}/{args.name}", exist_ok=True)
    
    # Evaluate the model
    if args.model_type == 'yolov5':
        metrics = evaluate_yolov5(args)
        
        # Calculate inference speed
        fps = calculate_fps(args)
        
        # Save results to CSV
        results = {
            'Model': [os.path.basename(args.model)],
            'mAP@0.5': [metrics.map50],
            'mAP@0.5:0.95': [metrics.map],
            'Precision': [metrics.p],
            'Recall': [metrics.r],
            'F1-Score': [2 * (metrics.p * metrics.r) / (metrics.p + metrics.r + 1e-16)],
            'FPS': [fps]
        }
        
    elif args.model_type == 'yolov9':
        metrics = evaluate_yolov9(args)
        
        # Calculate inference speed
        fps = calculate_fps(args)
        
        if metrics:
            # Save results to CSV
            results = {
                'Model': [os.path.basename(args.model)],
                'mAP@0.5': [metrics['map50']],
                'mAP@0.5:0.95': [metrics['map']],
                'Precision': [metrics['precision']],
                'Recall': [metrics['recall']],
                'F1-Score': [2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-16)],
                'FPS': [fps]
            }
        else:
            print("Evaluation failed, no metrics to save.")
            return
    
    else:
        print(f"Evaluation not implemented for model type: {args.model_type}")
        return
    
    df = pd.DataFrame(results)
    csv_path = f"{args.project}/{args.name}/results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main() 