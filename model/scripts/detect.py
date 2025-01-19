from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO('runs/train/exp/weights/best.pt')

# Test Görüntüleri Üzerinde Tahmin
results = model.predict(
    source='test_images/',  # Test görüntülerinin bulunduğu klasör
    conf=0.25,              # Güven eşiği
    save=True               # Tahmin edilen sonuçları kaydet
)