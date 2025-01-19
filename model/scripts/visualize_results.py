import pandas as pd
import matplotlib.pyplot as plt

# Eğitim sonuçlarını yükle
results_file = 'runs/train/exp/results.csv'  # Eğitim sonuçlarının kaydedildiği varsayılan dosya yolu
data = pd.read_csv(results_file)

# Epoch'lar ve metrikler
epochs = data.index + 1
train_loss = data['train/box_loss']
val_loss = data['metrics/val_box_loss']
mAP_50 = data['metrics/mAP_50']
mAP_50_95 = data['metrics/mAP_50-95']

# Eğitim kaybını görselleştir
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Eğitim Kayıp', color='blue')
plt.plot(epochs, val_loss, label='Doğrulama Kayıp', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.title('Eğitim ve Doğrulama Kayıp Grafiği')
plt.legend()
plt.grid()
plt.show()

# mAP İlerleyişi
plt.figure(figsize=(10, 6))
plt.plot(epochs, mAP_50, label='mAP@50', color='green')
plt.plot(epochs, mAP_50_95, label='mAP@50:95', color='red')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('mAP İlerleyişi')
plt.legend()
plt.grid()
plt.show()
