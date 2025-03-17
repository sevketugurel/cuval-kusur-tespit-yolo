#!/usr/bin/env python
"""
Gelişmiş veri artırma (augmentation) pipeline'ı.
YOLOv9 için çuval kusur tespiti veri setini zenginleştirmek amacıyla kullanılır.
"""

import os
import cv2
import numpy as np
import albumentations as A
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import random
import shutil

class AugmentationPipeline:
    """
    Gelişmiş veri artırma pipeline sınıfı.
    Çeşitli veri artırma teknikleri ve yapılandırma seçenekleri sunar.
    """
    
    def __init__(self, config=None):
        """
        Veri artırma pipeline'ını başlatır.
        
        Args:
            config (dict, optional): Yapılandırma parametreleri. None ise varsayılan değerler kullanılır.
        """
        self.config = config if config else self._get_default_config()
        self.augmentation_pipeline = self._create_pipeline()
        
    def _get_default_config(self):
        """
        Varsayılan yapılandırma parametrelerini döndürür.
        """
        return {
            'input_image_dir': 'data/train/images',
            'input_label_dir': 'data/train/labels',
            'output_image_dir': 'data/augmented/images',
            'output_label_dir': 'data/augmented/labels',
            'variations_per_image': 3,
            'augmentation_probability': 0.5,
            'rotation_limit': 30,
            'brightness_contrast_limit': 0.3,
            'noise_probability': 0.3,
            'shift_limit': 0.1,
            'scale_limit': 0.2,
            'hue_saturation_value_probability': 0.5,
            'blur_probability': 0.2,
            'cutout_probability': 0.2,
            'mosaic_probability': 0.3,
            'mixup_probability': 0.2,
            'preserve_original': True
        }
    
    def _create_pipeline(self):
        """
        Veri artırma pipeline'ını oluşturur.
        """
        return A.Compose([
            A.HorizontalFlip(p=self.config['augmentation_probability']),
            A.VerticalFlip(p=self.config['augmentation_probability'] * 0.4),  # Dikey çevirme daha az olsun
            A.Rotate(limit=self.config['rotation_limit'], p=self.config['augmentation_probability']),
            A.RandomBrightnessContrast(
                brightness_limit=self.config['brightness_contrast_limit'],
                contrast_limit=self.config['brightness_contrast_limit'],
                p=self.config['augmentation_probability']
            ),
            A.GaussNoise(p=self.config['noise_probability']),
            A.ShiftScaleRotate(
                shift_limit=self.config['shift_limit'],
                scale_limit=self.config['scale_limit'],
                rotate_limit=self.config['rotation_limit'],
                p=self.config['augmentation_probability']
            ),
            A.RandomSizedCrop(
                min_max_height=(300, 500),
                height=640,
                width=640,
                p=self.config['augmentation_probability'] * 0.5
            ),
            A.HueSaturationValue(p=self.config['hue_saturation_value_probability']),
            A.Blur(blur_limit=7, p=self.config['blur_probability']),
            A.Cutout(
                num_holes=8,
                max_h_size=64,
                max_w_size=64,
                fill_value=0,
                p=self.config['cutout_probability']
            ),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    def read_yolo_labels(self, label_path, img_width, img_height):
        """
        YOLO formatındaki etiketleri okur ve Pascal VOC formatına dönüştürür.
        
        Args:
            label_path (str): Etiket dosyasının yolu
            img_width (int): Görüntü genişliği
            img_height (int): Görüntü yüksekliği
            
        Returns:
            tuple: (bboxes, class_labels) - Sınırlayıcı kutular ve sınıf etiketleri
        """
        bboxes = []
        class_labels = []
        
        if not os.path.exists(label_path):
            print(f"Uyarı: Etiket dosyası bulunamadı: {label_path}")
            return bboxes, class_labels
        
        with open(label_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # YOLO formatını piksel koordinatlarına çevir
                x_min = int((x_center - width/2) * img_width)
                y_min = int((y_center - height/2) * img_height)
                x_max = int((x_center + width/2) * img_width)
                y_max = int((y_center + height/2) * img_height)
                
                # Sınırları kontrol et
                x_min = max(0, min(x_min, img_width - 1))
                y_min = max(0, min(y_min, img_height - 1))
                x_max = max(x_min + 1, min(x_max, img_width))
                y_max = max(y_min + 1, min(y_max, img_height))
                
                bboxes.append([x_min, y_min, x_max, y_max])
                class_labels.append(class_id)
                
        return bboxes, class_labels
    
    def save_yolo_labels(self, output_path, bboxes, class_labels, img_width, img_height):
        """
        Etiketleri YOLO formatında kaydeder.
        
        Args:
            output_path (str): Çıktı dosyasının yolu
            bboxes (list): Sınırlayıcı kutular (Pascal VOC formatında)
            class_labels (list): Sınıf etiketleri
            img_width (int): Görüntü genişliği
            img_height (int): Görüntü yüksekliği
        """
        with open(output_path, "w") as file:
            for bbox, class_id in zip(bboxes, class_labels):
                # Pascal VOC formatından YOLO formatına dönüştür
                x_min, y_min, x_max, y_max = bbox
                x_center = (x_min + x_max) / (2 * img_width)
                y_center = (y_min + y_max) / (2 * img_height)
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Değerleri 0-1 aralığında sınırla
                x_center = max(0, min(x_center, 1.0))
                y_center = max(0, min(y_center, 1.0))
                width = max(0, min(width, 1.0))
                height = max(0, min(height, 1.0))
                
                file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def apply_mosaic(self, images, bboxes_list, class_labels_list, target_size=(640, 640)):
        """
        Mozaik veri artırma tekniğini uygular (4 görüntüyü birleştirir).
        
        Args:
            images (list): Görüntü listesi
            bboxes_list (list): Her görüntü için sınırlayıcı kutu listesi
            class_labels_list (list): Her görüntü için sınıf etiketi listesi
            target_size (tuple): Hedef görüntü boyutu
            
        Returns:
            tuple: (mosaic_image, mosaic_bboxes, mosaic_class_labels)
        """
        if len(images) < 4:
            # Yeterli görüntü yoksa, ilk görüntüyü döndür
            return images[0], bboxes_list[0], class_labels_list[0]
        
        # Rastgele 4 görüntü seç
        indices = random.sample(range(len(images)), 4)
        selected_images = [images[i] for i in indices]
        selected_bboxes = [bboxes_list[i] for i in indices]
        selected_class_labels = [class_labels_list[i] for i in indices]
        
        # Mozaik görüntüsü oluştur
        mosaic_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_class_labels = []
        
        # Görüntüleri yerleştir
        positions = [
            (0, 0, target_size[0]//2, target_size[1]//2),  # Sol üst
            (target_size[0]//2, 0, target_size[0], target_size[1]//2),  # Sağ üst
            (0, target_size[1]//2, target_size[0]//2, target_size[1]),  # Sol alt
            (target_size[0]//2, target_size[1]//2, target_size[0], target_size[1])  # Sağ alt
        ]
        
        for i, (img, bboxes, class_labels) in enumerate(zip(selected_images, selected_bboxes, selected_class_labels)):
            x1, y1, x2, y2 = positions[i]
            
            # Görüntüyü yeniden boyutlandır
            h, w = y2 - y1, x2 - x1
            resized_img = cv2.resize(img, (w, h))
            mosaic_image[y1:y2, x1:x2] = resized_img
            
            # Sınırlayıcı kutuları ayarla
            for bbox, cls in zip(bboxes, class_labels):
                orig_x1, orig_y1, orig_x2, orig_y2 = bbox
                orig_h, orig_w = img.shape[:2]
                
                # Yeni koordinatları hesapla
                new_x1 = int(orig_x1 * w / orig_w) + x1
                new_y1 = int(orig_y1 * h / orig_h) + y1
                new_x2 = int(orig_x2 * w / orig_w) + x1
                new_y2 = int(orig_y2 * h / orig_h) + y1
                
                # Sınırları kontrol et
                new_x1 = max(0, min(new_x1, target_size[0] - 1))
                new_y1 = max(0, min(new_y1, target_size[1] - 1))
                new_x2 = max(new_x1 + 1, min(new_x2, target_size[0]))
                new_y2 = max(new_y1 + 1, min(new_y2, target_size[1]))
                
                mosaic_bboxes.append([new_x1, new_y1, new_x2, new_y2])
                mosaic_class_labels.append(cls)
        
        return mosaic_image, mosaic_bboxes, mosaic_class_labels
    
    def apply_mixup(self, img1, bboxes1, class_labels1, img2, bboxes2, class_labels2, alpha=0.5):
        """
        Mixup veri artırma tekniğini uygular (iki görüntüyü karıştırır).
        
        Args:
            img1, img2: Karıştırılacak görüntüler
            bboxes1, bboxes2: Görüntülere ait sınırlayıcı kutular
            class_labels1, class_labels2: Görüntülere ait sınıf etiketleri
            alpha (float): Karıştırma oranı
            
        Returns:
            tuple: (mixed_image, mixed_bboxes, mixed_class_labels)
        """
        # İki görüntüyü aynı boyuta getir
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Görüntüleri karıştır
        mixed_image = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
        
        # Tüm sınırlayıcı kutuları ve etiketleri birleştir
        mixed_bboxes = bboxes1 + bboxes2
        mixed_class_labels = class_labels1 + class_labels2
        
        return mixed_image, mixed_bboxes, mixed_class_labels
    
    def apply_augmentation(self):
        """
        Veri artırma işlemini uygular.
        """
        # Klasörlerin varlığını kontrol et ve oluştur
        for dir_path in [self.config['input_image_dir'], self.config['input_label_dir'], 
                         self.config['output_image_dir'], self.config['output_label_dir']]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"Veri artırma işlemi başladı...")
        print(f"Giriş görüntü dizini: {self.config['input_image_dir']}")
        print(f"Çıkış görüntü dizini: {self.config['output_image_dir']}")
        
        # Tüm görüntüleri ve etiketleri yükle
        image_files = [f for f in os.listdir(self.config['input_image_dir']) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"Uyarı: {self.config['input_image_dir']} dizininde görüntü bulunamadı!")
            return
        
        print(f"Toplam {len(image_files)} görüntü bulundu.")
        
        # Orijinal görüntüleri kopyala (eğer isteniyorsa)
        if self.config['preserve_original']:
            print("Orijinal görüntüler korunuyor...")
            for img_name in tqdm(image_files, desc="Orijinal görüntüler kopyalanıyor"):
                img_path = os.path.join(self.config['input_image_dir'], img_name)
                label_path = os.path.join(self.config['input_label_dir'], 
                                         os.path.splitext(img_name)[0] + '.txt')
                
                # Görüntüyü kopyala
                shutil.copy2(img_path, os.path.join(self.config['output_image_dir'], img_name))
                
                # Etiket dosyası varsa kopyala
                if os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(self.config['output_label_dir'], 
                                                         os.path.splitext(img_name)[0] + '.txt'))
        
        # Tüm görüntüleri ve etiketleri yükle (mozaik ve mixup için)
        all_images = []
        all_bboxes = []
        all_class_labels = []
        
        for img_name in tqdm(image_files, desc="Görüntüler yükleniyor"):
            img_path = os.path.join(self.config['input_image_dir'], img_name)
            label_path = os.path.join(self.config['input_label_dir'], 
                                     os.path.splitext(img_name)[0] + '.txt')
            
            image = cv2.imread(img_path)
            if image is None:
                print(f"Uyarı: Görüntü yüklenemedi: {img_path}")
                continue
                
            img_height, img_width, _ = image.shape
            bboxes, class_labels = self.read_yolo_labels(label_path, img_width, img_height)
            
            all_images.append(image)
            all_bboxes.append(bboxes)
            all_class_labels.append(class_labels)
        
        # Standart veri artırma
        print("Standart veri artırma uygulanıyor...")
        for i, img_name in enumerate(tqdm(image_files, desc="Standart artırma")):
            image = all_images[i]
            bboxes = all_bboxes[i]
            class_labels = all_class_labels[i]
            
            img_height, img_width, _ = image.shape
            
            # Her görüntü için belirtilen sayıda varyasyon oluştur
            for j in range(self.config['variations_per_image']):
                # Veri artırma işlemi
                augmented = self.augmentation_pipeline(
                    image=image, 
                    bboxes=bboxes, 
                    class_labels=class_labels
                )
                augmented_image = augmented["image"]
                augmented_bboxes = augmented["bboxes"]
                augmented_class_labels = augmented["class_labels"]
                
                # Artırılmış görüntüyü kaydet
                output_img_name = f"aug_{j}_{img_name}"
                output_image_path = os.path.join(self.config['output_image_dir'], output_img_name)
                cv2.imwrite(output_image_path, augmented_image)
                
                # Artırılmış etiketleri kaydet
                output_label_path = os.path.join(
                    self.config['output_label_dir'], 
                    f"aug_{j}_{os.path.splitext(img_name)[0]}.txt"
                )
                self.save_yolo_labels(
                    output_label_path, 
                    augmented_bboxes, 
                    augmented_class_labels, 
                    augmented_image.shape[1], 
                    augmented_image.shape[0]
                )
        
        # Mozaik veri artırma
        if random.random() < self.config['mosaic_probability'] and len(all_images) >= 4:
            print("Mozaik veri artırma uygulanıyor...")
            num_mosaics = len(image_files) // 2  # Görüntü sayısının yarısı kadar mozaik oluştur
            
            for i in tqdm(range(num_mosaics), desc="Mozaik artırma"):
                mosaic_image, mosaic_bboxes, mosaic_class_labels = self.apply_mosaic(
                    all_images, all_bboxes, all_class_labels
                )
                
                # Mozaik görüntüyü kaydet
                output_img_name = f"mosaic_{i}.jpg"
                output_image_path = os.path.join(self.config['output_image_dir'], output_img_name)
                cv2.imwrite(output_image_path, mosaic_image)
                
                # Mozaik etiketleri kaydet
                output_label_path = os.path.join(self.config['output_label_dir'], f"mosaic_{i}.txt")
                self.save_yolo_labels(
                    output_label_path, 
                    mosaic_bboxes, 
                    mosaic_class_labels, 
                    mosaic_image.shape[1], 
                    mosaic_image.shape[0]
                )
        
        # Mixup veri artırma
        if random.random() < self.config['mixup_probability'] and len(all_images) >= 2:
            print("Mixup veri artırma uygulanıyor...")
            num_mixups = len(image_files) // 3  # Görüntü sayısının üçte biri kadar mixup oluştur
            
            for i in tqdm(range(num_mixups), desc="Mixup artırma"):
                # Rastgele iki görüntü seç
                idx1, idx2 = random.sample(range(len(all_images)), 2)
                
                mixed_image, mixed_bboxes, mixed_class_labels = self.apply_mixup(
                    all_images[idx1], all_bboxes[idx1], all_class_labels[idx1],
                    all_images[idx2], all_bboxes[idx2], all_class_labels[idx2],
                    alpha=random.uniform(0.3, 0.7)
                )
                
                # Mixup görüntüyü kaydet
                output_img_name = f"mixup_{i}.jpg"
                output_image_path = os.path.join(self.config['output_image_dir'], output_img_name)
                cv2.imwrite(output_image_path, mixed_image)
                
                # Mixup etiketleri kaydet
                output_label_path = os.path.join(self.config['output_label_dir'], f"mixup_{i}.txt")
                self.save_yolo_labels(
                    output_label_path, 
                    mixed_bboxes, 
                    mixed_class_labels, 
                    mixed_image.shape[1], 
                    mixed_image.shape[0]
                )
        
        print("Veri artırma işlemi tamamlandı!")
        print(f"Artırılmış görüntüler {self.config['output_image_dir']} dizinine kaydedildi.")
        print(f"Artırılmış etiketler {self.config['output_label_dir']} dizinine kaydedildi.")

def parse_args():
    """
    Komut satırı argümanlarını ayrıştırır.
    """
    parser = argparse.ArgumentParser(description='Gelişmiş veri artırma pipeline')
    parser.add_argument('--config', type=str, help='Yapılandırma dosyası yolu')
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
    
    # Yapılandırma dosyasını yükle (varsa)
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
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
    pipeline = AugmentationPipeline(config)
    pipeline.apply_augmentation()

if __name__ == "__main__":
    main() 