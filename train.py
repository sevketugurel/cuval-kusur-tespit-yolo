import torch
from ultralytics import YOLO

def main():
    # YOLOv8n (nano) modelini kullan
    model = YOLO('yolov8n.pt')  
    
    # Eğitimi başlat
    results = model.train(
        data='configs/data.yaml',
        epochs=20,  
        imgsz=640,
        batch=8,    
        save_period=5,  
        project='runs/train',
        name='initial_test'
    )

if __name__ == "__main__":
    main() 