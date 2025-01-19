from ultralytics import YOLO

# YOLOv11 modelini yükle
model = YOLO('yolov11.pt')  # Önceden eğitilmiş YOLOv11 ağırlıkları

# Modeli eğit
results = model.train(
    data='data.yaml',       # Dataset yapılandırma dosyası
    epochs=50,              # Eğitim tekrarı
    imgsz=640,              # Görüntü boyutu
    batch=16,               # Batch boyutu
    device=0                # GPU kullanımı için '0', CPU için 'cpu'
)

# Eğitimden sonra en iyi modeli kaydet
model.save('custom_yolov11_model.pt')
