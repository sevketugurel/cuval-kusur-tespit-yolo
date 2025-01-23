import os
import cv2
import albumentations as A
from tqdm import tqdm

# Veri artırma pipeline'ı tanımlama
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.RandomSizedCrop(min_max_height=(300, 500), height=512, width=512, p=0.5),
    A.HueSaturationValue(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Klasör yollarını belirleme
input_image_dir = "train/images"  # Orijinal görüntülerin bulunduğu klasör
input_label_dir = "train/labels"  # Etiketlerin bulunduğu klasör
output_image_dir = "augmented/augmented_images"  # Artırılmış görüntülerin kaydedileceği klasör
output_label_dir = "augmented/augmented_labels"  # Artırılmış etiketlerin kaydedileceği klasör

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# YOLO formatındaki etiketleri okuma fonksiyonu
def read_yolo_labels(label_path, img_width, img_height):
    bboxes = []
    class_labels = []
    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            # YOLO formatını Albumentations formatına çevir
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)
    return bboxes, class_labels

# Veri artırma işlemi
def apply_augmentation():
    print("Veri artırma işlemi başladı...")
    for img_name in tqdm(os.listdir(input_image_dir)):
        img_path = os.path.join(input_image_dir, img_name)
        label_path = os.path.join(input_label_dir, img_name.replace(".jpg", ".txt"))

        # Görüntü ve etiketlerin yüklenmesi
        image = cv2.imread(img_path)
        if image is None:
            print(f"Görüntü yüklenemedi: {img_path}")
            continue

        img_height, img_width, _ = image.shape
        bboxes, class_labels = read_yolo_labels(label_path, img_width, img_height)

        # Her görüntüden 3 varyasyon oluştur
        for i in range(3):
            # Veri artırma işlemi
            augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            augmented_image = augmented["image"]
            augmented_bboxes = augmented["bboxes"]
            augmented_class_labels = augmented["class_labels"]

            # Artırılmış görüntüyü kaydetme
            output_image_path = os.path.join(output_image_dir, f"aug_{i}_{img_name}")
            cv2.imwrite(output_image_path, augmented_image)

            # Artırılmış etiketleri YOLO formatında kaydetme
            output_label_path = os.path.join(output_label_dir, f"aug_{i}_{img_name.replace('.jpg', '.txt')}")
            with open(output_label_path, "w") as file:
                for bbox, class_id in zip(augmented_bboxes, augmented_class_labels):
                    # YOLO formatına dönüştür
                    x_center, y_center, width, height = bbox
                    file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print("Veri artırma işlemi tamamlandı!")

# Pipeline'ı çalıştırma
if __name__ == "__main__":
    apply_augmentation()
