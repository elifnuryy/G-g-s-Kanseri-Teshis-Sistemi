import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

print("1. Veriler yükleniyor...")
# Hazır kanser verisini yüklüyoruz
data = load_breast_cancer()
X = data.data
y = data.target

print("2. Veriler CNN için hazırlanıyor...")
# Verileri standartlaştırıyoruz
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scaler'ı kaydediyoruz
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# CNN için boyutlandırma (Reshape)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Eğitim ve Test verisi ayırma
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

print("3. Yapay Zeka (CNN) inşa ediliyor...")
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(30, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("4. Eğitim başlıyor! (Biraz bekleyin)...")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

model.save('cnn_model.h5')
print("TEBRİKLER! 'cnn_model.h5' dosyası oluşturuldu.")