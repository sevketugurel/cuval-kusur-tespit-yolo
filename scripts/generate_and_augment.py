#!/usr/bin/env python
"""
Cycle-GAN ve veri artırma pipeline'ını entegre eden iş akışı.
Önce Cycle-GAN ile sentetik kusurlu görüntüler üretir, ardından bu görüntüleri veri artırma pipeline'ı ile çoğaltır.
"""

import os
import sys
import argparse
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm

# Ana dizini ekle
sys.path.append(str(Path(__file__).parent.parent))

# Cycle-GAN ve veri artırma modüllerini içe aktar
from utils.cyclegan.cyclegan import CycleGAN
from utils.augmentation.augment_pipeline import AugmentationPipeline

def parse_args():
    """
    Komut satırı argümanlarını ayrıştırır.
    """
    parser = argparse.ArgumentParser(description='Cycle-GAN ve veri artırma pipeline entegrasyonu')
    
    # Genel parametreler
    parser.add_argument('--config', type=str, default='configs/data_generation.yaml',
                        help='Yapılandırma dosyası yolu')
    parser.add_argument('--skip-cyclegan', action='store_true',
                        help='Cycle-GAN adımını atla (sadece veri artırma uygula)')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Veri artırma adımını atla (sadece Cycle-GAN uygula)')
    
    # Cycle-GAN parametreleri
    parser.add_argument('--cyclegan-model', type=str, help='Önceden eğitilmiş Cycle-GAN model yolu')
    parser.add_argument('--source-images', type=str, help='Kaynak görüntü dizini (kusursuz çuvallar)')
    parser.add_argument('--synthetic-output', type=str, help='Sentetik görüntü çıktı dizini')
    parser.add_argument('--num-synthetic', type=int, help='Üretilecek sentetik görüntü sayısı')
    
    # Veri artırma parametreleri
    parser.add_argument('--augmentation-config', type=str, 
                        default='utils/augmentation/config.yaml',
                        help='Veri artırma yapılandırma dosyası')
    parser.add_argument('--input-images', type=str, help='Veri artırma için giriş görüntü dizini')
    parser.add_argument('--input-labels', type=str, help='Veri artırma için giriş etiket dizini')
    parser.add_argument('--output-images', type=str, help='Veri artırma için çıkış görüntü dizini')
    parser.add_argument('--output-labels', type=str, help='Veri artırma için çıkış etiket dizini')
    parser.add_argument('--variations', type=int, help='Görüntü başına varyasyon sayısı')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Yapılandırma dosyasını yükler.
    
    Args:
        config_path (str): Yapılandırma dosyası yolu
        
    Returns:
        dict: Yapılandırma parametreleri
    """
    if os.path.exists(config_path):
        print(f"Yapılandırma dosyası yükleniyor: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Uyarı: Yapılandırma dosyası bulunamadı: {config_path}")
        return {}

def run_cyclegan(config, args):
    """
    Cycle-GAN ile sentetik kusurlu görüntüler üretir.
    
    Args:
        config (dict): Yapılandırma parametreleri
        args (argparse.Namespace): Komut satırı argümanları
        
    Returns:
        str: Sentetik görüntülerin kaydedildiği dizin
    """
    # Cycle-GAN parametrelerini ayarla
    cyclegan_model = args.cyclegan_model or config.get('cyclegan_model')
    source_images = args.source_images or config.get('source_images')
    synthetic_output = args.synthetic_output or config.get('synthetic_output', 'data/synthetic/images')
    num_synthetic = args.num_synthetic or config.get('num_synthetic', 100)
    
    if not cyclegan_model or not source_images:
        print("Hata: Cycle-GAN modeli ve kaynak görüntü dizini belirtilmelidir.")
        print("Kullanım: --cyclegan-model MODEL_PATH --source-images SOURCE_DIR")
        return None
    
    # Çıktı dizinini oluştur
    os.makedirs(synthetic_output, exist_ok=True)
    
    print(f"Cycle-GAN ile sentetik kusurlu görüntüler üretiliyor...")
    print(f"Model: {cyclegan_model}")
    print(f"Kaynak görüntüler: {source_images}")
    print(f"Çıktı dizini: {synthetic_output}")
    print(f"Üretilecek görüntü sayısı: {num_synthetic}")
    
    # Cycle-GAN modelini yükle ve sentetik görüntüler üret
    cyclegan = CycleGAN(model_path=cyclegan_model)
    cyclegan.generate_images(
        source_dir=source_images,
        output_dir=synthetic_output,
        num_images=num_synthetic
    )
    
    print(f"Sentetik görüntüler başarıyla üretildi: {synthetic_output}")
    
    # Sentetik görüntüler için etiket dosyaları oluştur
    synthetic_labels = synthetic_output.replace('images', 'labels')
    os.makedirs(synthetic_labels, exist_ok=True)
    
    # Eğer etiket şablonu varsa, her sentetik görüntü için bir etiket dosyası oluştur
    label_template = config.get('label_template')
    if label_template and os.path.exists(label_template):
        print(f"Etiket şablonu kullanılarak sentetik görüntüler için etiketler oluşturuluyor...")
        
        with open(label_template, 'r') as f:
            template_content = f.read()
        
        for img_file in os.listdir(synthetic_output):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(img_file)[0]
                label_file = os.path.join(synthetic_labels, f"{base_name}.txt")
                
                with open(label_file, 'w') as f:
                    f.write(template_content)
    
    return {
        'images': synthetic_output,
        'labels': synthetic_labels
    }

def run_augmentation(config, args, input_dirs=None):
    """
    Veri artırma pipeline'ını çalıştırır.
    
    Args:
        config (dict): Yapılandırma parametreleri
        args (argparse.Namespace): Komut satırı argümanları
        input_dirs (dict, optional): Giriş dizinleri (Cycle-GAN çıktısı)
        
    Returns:
        bool: İşlem başarılı ise True, değilse False
    """
    # Veri artırma yapılandırmasını yükle
    aug_config_path = args.augmentation_config or config.get('augmentation_config', 'utils/augmentation/config.yaml')
    aug_config = load_config(aug_config_path)
    
    # Giriş ve çıkış dizinlerini ayarla
    if input_dirs:
        # Cycle-GAN çıktısını kullan
        input_images = input_dirs['images']
        input_labels = input_dirs['labels']
    else:
        # Komut satırı argümanlarını veya yapılandırma dosyasını kullan
        input_images = args.input_images or config.get('input_images')
        input_labels = args.input_labels or config.get('input_labels')
    
    output_images = args.output_images or config.get('output_images', 'data/augmented/images')
    output_labels = args.output_labels or config.get('output_labels', 'data/augmented/labels')
    variations = args.variations or config.get('variations')
    
    if not input_images or not input_labels:
        print("Hata: Veri artırma için giriş görüntü ve etiket dizinleri belirtilmelidir.")
        print("Kullanım: --input-images INPUT_DIR --input-labels LABELS_DIR")
        return False
    
    # Veri artırma yapılandırmasını güncelle
    aug_config['input_image_dir'] = input_images
    aug_config['input_label_dir'] = input_labels
    aug_config['output_image_dir'] = output_images
    aug_config['output_label_dir'] = output_labels
    
    if variations:
        aug_config['variations_per_image'] = variations
    
    print(f"Veri artırma pipeline'ı çalıştırılıyor...")
    print(f"Giriş görüntüleri: {input_images}")
    print(f"Giriş etiketleri: {input_labels}")
    print(f"Çıkış görüntüleri: {output_images}")
    print(f"Çıkış etiketleri: {output_labels}")
    
    # Veri artırma pipeline'ını çalıştır
    pipeline = AugmentationPipeline(aug_config)
    pipeline.apply_augmentation()
    
    print(f"Veri artırma işlemi tamamlandı!")
    return True

def main():
    """
    Ana fonksiyon.
    """
    args = parse_args()
    
    # Yapılandırma dosyasını yükle
    config = load_config(args.config)
    
    # Cycle-GAN ile sentetik görüntüler üret
    synthetic_dirs = None
    if not args.skip_cyclegan:
        synthetic_dirs = run_cyclegan(config, args)
        if not synthetic_dirs:
            print("Cycle-GAN adımı başarısız oldu.")
            if not args.skip_augmentation and (args.input_images and args.input_labels):
                print("Veri artırma adımına devam ediliyor...")
            else:
                return
    
    # Veri artırma pipeline'ını çalıştır
    if not args.skip_augmentation:
        # Eğer Cycle-GAN çalıştırıldıysa ve başarılıysa, onun çıktısını kullan
        if synthetic_dirs:
            success = run_augmentation(config, args, synthetic_dirs)
        else:
            # Değilse, doğrudan belirtilen giriş dizinlerini kullan
            success = run_augmentation(config, args)
        
        if not success:
            print("Veri artırma adımı başarısız oldu.")
            return
    
    print("İş akışı başarıyla tamamlandı!")

if __name__ == "__main__":
    main() 