🎗️ Yapay Zeka Destekli Göğüs Kanseri Teşhis ve 3D Görselleştirme Sistemi

Bu proje, meme kanseri verilerini Derin Öğrenme (Deep Learning) yöntemleriyle analiz eden ve sonuçları Unity Oyun Motoru üzerinde 3 boyutlu olarak görselleştiren bir Karar Destek Sistemidir.

Proje Hakkında

Meme kanserinde erken teşhis hayati önem taşır. Bu sistem, doktorların sayısal verileri daha hızlı ve hatasız yorumlamasına yardımcı olmak amacıyla geliştirilmiştir. Proje iki ana modülden oluşur:
1. Yapay Zeka Modülü (Python): 1D-CNN (Convolutional Neural Network) kullanarak %98+ doğrulukla teşhis koyar.
2. Görselleştirme Modülü (Unity): Sonuçları anatomik bir model üzerinde görselleştirir.

🛠️ Kullanılan Teknolojiler

* Yapay Zeka & Backend: Python, TensorFlow (Keras), Flask, Scikit-learn, Pandas.
* Arayüz & Görselleştirme: Unity 3D, C#.
* Veri Seti: UCI Breast Cancer Wisconsin (Diagnostic).

⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Python Sunucusunu Başlatma
Önce yapay zeka sunucusunu ayağa kaldırmanız gerekir. Terminalde proje klasörüne gidip şu komutu çalıştırın:

```bash
python flaskserver.py
Sunucu http://127.0.0.1:5000 adresinde çalışmaya başlayacaktır.
<img width="605" height="332" alt="image" src="https://github.com/user-attachments/assets/15a83b4c-b8db-4896-9df2-284acaaaf924" />


Unity Simülasyonunu Açma
Unity Hub üzerinden projeyi açın.

Scenes klasöründen ana sahneyi başlatın.

Play tuşuna bastığınızda Unity, Python sunucusu ile haberleşmeye başlayacaktır.

<img width="605" height="325" alt="image" src="https://github.com/user-attachments/assets/77719bfe-2c42-4b18-9a26-f6b8d8baaa92" />


📊 Model Başarısı
Geliştirilen 1D-CNN modeli, test verileri üzerinde %98.68 doğruluk (Accuracy) oranına ulaşmıştır. Ayrıca yanlış negatifleri önlemek için Duyarlılık (Recall) değeri optimize edilmiştir.
<img width="605" height="319" alt="image" src="https://github.com/user-attachments/assets/504872fb-6d03-4b54-8e86-161b61cf9072" />

Hazırlayan: Elifnur YÜKSEL

