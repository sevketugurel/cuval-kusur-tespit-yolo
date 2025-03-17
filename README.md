# Sack Defect Detection Project

## Project Overview
Bu proje, endüstriyel üretimde kullanılan çuvallardaki kusurları (yırtık, baskı hataları, dikiş problemleri) YOLOv9 tabanlı derin öğrenme modelleri kullanarak tespit etmeyi ve sınıflandırmayı amaçlamaktadır. Sistem, üretim ortamlarında kalite kontrolü geliştirmek için gerçek zamanlı, yüksek doğrulukta tespit sağlar.

## Özellikler
- Çuvallardaki birden fazla kusur tipinin gerçek zamanlı tespiti
- YOLOv9 modelinin performans optimizasyonu
- Dikkat mekanizmalarının (CBAM, GAM, CoordATT) entegrasyonu ile geliştirilmiş tespit
- Entegre veri üretimi ve artırma pipeline'ı:
  - Cycle-GAN ile sentetik kusurlu görüntü üretimi
  - Gelişmiş veri artırma teknikleri (Mozaik, Mixup, Cutout vb.)
- Kapsamlı değerlendirme metrikleri (Precision, Recall, F1, mAP, IoU)
- Görselleştirme ve izleme için web tabanlı arayüz

## Proje Yapısı
```
cuval-kusur-tespit-yolo/
├── data/                      # Veri seti dizini
│   ├── raw/                   # Orijinal görüntüler
│   │   ├── normal/            # Kusursuz çuval görüntüleri
│   │   └── defect/            # Kusurlu çuval görüntüleri
│   ├── processed/             # İşlenmiş ve etiketlenmiş görüntüler
│   ├── synthetic/             # Cycle-GAN ile üretilen sentetik görüntüler
│   │   ├── images/            # Sentetik görüntüler
│   │   └── labels/            # Sentetik görüntü etiketleri
│   ├── train/                 # Eğitim veri seti
│   │   ├── images/            # Eğitim görüntüleri
│   │   └── labels/            # Eğitim etiketleri (YOLO formatı)
│   ├── val/                   # Doğrulama veri seti
│   │   ├── images/            # Doğrulama görüntüleri
│   │   └── labels/            # Doğrulama etiketleri (YOLO formatı)
│   ├── test/                  # Test veri seti
│   │   ├── images/            # Test görüntüleri
│   │   └── labels/            # Test etiketleri (YOLO formatı)
│   └── augmented/             # Artırılmış veri seti
│       ├── images/            # Artırılmış görüntüler
│       └── labels/            # Artırılmış etiketler
├── models/                    # Model implementasyonları
│   ├── attention/             # Dikkat mekanizması implementasyonları
│   │   ├── cbam.py            # Convolutional Block Attention Module
│   │   ├── gam.py             # Global Attention Module
│   │   └── coordatt.py        # Coordinate Attention
│   └── cyclegan/              # Cycle-GAN model dosyaları
│       └── latest_net_G.pth   # Eğitilmiş Cycle-GAN modeli
├── configs/                   # Konfigürasyon dosyaları
│   ├── data.yaml              # Veri seti konfigürasyonu
│   ├── yolov9.yaml            # YOLOv9 konfigürasyonu
│   ├── data_generation.yaml   # Veri üretimi ve artırma konfigürasyonu
│   └── label_templates/       # Etiket şablonları
│       └── defect_template.txt # Sentetik görüntüler için etiket şablonu
├── utils/                     # Yardımcı scriptler
│   ├── cyclegan/              # Sentetik veri için Cycle-GAN implementasyonu
│   │   └── cyclegan.py        # Cycle-GAN sınıfı
│   └── augmentation/          # Gelişmiş veri artırma pipeline'ı
│       ├── augment_pipeline.py # Veri artırma sınıfı
│       └── config.yaml        # Veri artırma yapılandırması
├── scripts/                   # Eğitim ve çıkarım scriptleri
│   ├── train.py               # Eğitim scripti
│   ├── evaluate.py            # Değerlendirme scripti
│   ├── inference.py           # Çıkarım scripti
│   ├── augment_data.py        # Veri artırma scripti
│   └── generate_and_augment.py # Entegre veri üretimi ve artırma scripti
├── requirements.txt           # Python bağımlılıkları
└── README.md                  # Proje dokümantasyonu
```

## Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/yourusername/cuval-kusur-tespit-yolo.git
cd cuval-kusur-tespit-yolo

# Sanal ortam oluşturun ve aktifleştirin
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# YOLOv9 repository'sini klonlayın
git clone https://github.com/WongKinYiu/yolov9.git
```

## Kullanım

### Veri Hazırlama, Üretimi ve Artırma
```bash
# Veri setini düzenleyin ve ön işleyin
python scripts/prepare_data.py --source /path/to/raw/images --dest data/processed

# Entegre veri üretimi ve artırma iş akışını çalıştırın
python scripts/generate_and_augment.py --config configs/data_generation.yaml
```

Entegre iş akışı şu adımları gerçekleştirir:
1. **Cycle-GAN ile Sentetik Görüntü Üretimi**: Kusursuz çuval görüntülerinden sentetik kusurlu görüntüler üretir.
2. **Veri Artırma**: Hem gerçek hem de sentetik görüntüleri çeşitli tekniklerle artırır:
   - Standart artırma (döndürme, çevirme, parlaklık/kontrast ayarları vb.)
   - Mozaik (4 görüntüyü birleştirme)
   - Mixup (2 görüntüyü karıştırma)
   - Cutout (rastgele bölgeleri maskeleme)

İş akışını özelleştirmek için `configs/data_generation.yaml` dosyasını düzenleyebilirsiniz:
```yaml
# Cycle-GAN yapılandırması
cyclegan_model: 'models/cyclegan/latest_net_G.pth'  # Önceden eğitilmiş model
source_images: 'data/raw/normal'  # Kusursuz çuval görüntüleri
num_synthetic: 100  # Üretilecek görüntü sayısı

# Veri artırma yapılandırması
variations: 3  # Görüntü başına varyasyon sayısı
```

Ayrıca, iş akışının belirli adımlarını atlamak için komut satırı argümanlarını kullanabilirsiniz:
```bash
# Sadece Cycle-GAN ile sentetik görüntüler üret
python scripts/generate_and_augment.py --skip-augmentation

# Sadece veri artırma uygula
python scripts/generate_and_augment.py --skip-cyclegan
```

### Eğitim
```bash
# YOLOv9 modelini eğitin
python scripts/train.py --model yolov9 --config configs/yolov9.yaml --data configs/data.yaml
```

### Değerlendirme
```bash
# Model performansını değerlendirin
python scripts/evaluate.py --model path/to/trained/model --data data/test --model-type yolov9
```

### Çıkarım (Inference)
```bash
# Yeni görüntüler üzerinde çıkarım yapın
python scripts/inference.py --model path/to/trained/model --source path/to/images --model-type yolov9
```

### Web Arayüzü
```bash
# Backend sunucusunu başlatın
cd backend
python app.py

# Frontend geliştirme sunucusunu başlatın
cd frontend
npm install
npm start
```

## Model Performansı

| Model | Precision | Recall | F1 Score | mAP@0.5 | FPS |
|-------|-----------|--------|----------|---------|-----|
| YOLOv9 | TBD | TBD | TBD | TBD | TBD |

## Dikkat Mekanizmaları Karşılaştırması

| Dikkat Mekanizması | Precision | Recall | F1 Score | mAP@0.5 | FPS |
|-------------------|-----------|--------|----------|---------|-----|
| Temel YOLOv9 | TBD | TBD | TBD | TBD | TBD |
| YOLOv9 + CBAM | TBD | TBD | TBD | TBD | TBD |
| YOLOv9 + GAM | TBD | TBD | TBD | TBD | TBD |
| YOLOv9 + CoordATT | TBD | TBD | TBD | TBD | TBD |

## Lisans
[Lisansınızı belirtin]

## Katkıda Bulunanlar
[Katkıda bulunanların listesi] 