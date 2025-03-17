#!/usr/bin/env python
"""
Veri artırma pipeline'ını çalıştırmak için script.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Ana dizini ekle
sys.path.append(str(Path(__file__).parent.parent))

# Veri artırma pipeline'ını içe aktar
from utils.augmentation.augment_pipeline import AugmentationPipeline

def parse_args():
    """
    Komut satırı argümanlarını ayrıştırır.
    """
    parser = argparse.ArgumentParser(description='Veri artırma pipeline\'ını çalıştır')
    parser.add_argument('--config', type=str, default='utils/augmentation/config.yaml',
                        help='Yapılandırma dosyası yolu')
    parser.add_argument('--input-images', type=str, help='Giriş görüntü dizini')
    parser.add_argument('--input-labels', type=str, help='Giriş etiket dizini')
    parser.add_argument('--output-images', type=str, help='Çıkış görüntü dizini')
    parser.add_argument('--output-labels', type=str, help='Çıkış etiket dizini')
    parser.add_argument('--variations', type=int, help='Görüntü başına varyasyon sayısı')
    parser.add_argument('--preserve-original', action='store_true', help='Orijinal görüntüleri koru')
    
    return parser.parse_args()

def main():
    """
    Ana fonksiyon.
    """
    args = parse_args()
    
    # Yapılandırma dosyasını yükle
    config = {}
    if args.config and os.path.exists(args.config):
        print(f"Yapılandırma dosyası yükleniyor: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Uyarı: Yapılandırma dosyası bulunamadı: {args.config}")
        print("Varsayılan yapılandırma kullanılacak.")
    
    # Komut satırı argümanlarını yapılandırmaya ekle
    if args.input_images:
        config['input_image_dir'] = args.input_images
    if args.input_labels:
        config['input_label_dir'] = args.input_labels
    if args.output_images:
        config['output_image_dir'] = args.output_images
    if args.output_labels:
        config['output_label_dir'] = args.output_labels
    if args.variations:
        config['variations_per_image'] = args.variations
    if args.preserve_original is not None:
        config['preserve_original'] = args.preserve_original
    
    # Pipeline'ı oluştur ve çalıştır
    print("Veri artırma pipeline'ı başlatılıyor...")
    pipeline = AugmentationPipeline(config)
    pipeline.apply_augmentation()
    
    print("İşlem tamamlandı!")

if __name__ == "__main__":
    main() 