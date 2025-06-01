# 🎭 Çok Modaliteli Duygu Analizi - Derin Öğrenme Projesi

Bu proje, **Yapay Sinir Ağları (ANN) tabanlı** derin öğrenme teknikleri kullanarak görüntü ve metin verilerini birlikte analiz eden profesyonel bir duygu analizi sistemidir.

## 📋 **Proje Şartlarına Uygunluk**

### ✅ **VERİ SETİ ŞARTLARİ**
- **Veri Kaynağı**: Kaggle uyumlu çok modaliteli veri seti
- **Veri Boyutu**: 71,702+ örnek (>>1,000 şartı)
- **Usability Score**: Kaggle standartlarına uygun
- **Görüntü Boyutu**: 128x128 piksel (şart ≥128x128)
- **NLP Verisi**: 1,000+ kelime, temizlenmiş İngilizce metinler
- **Sınıf Dağılımı**: POSITIVE/NEGATIVE (dengeli dağılım)

### 🤖 **KULLANILAN MODEL TİPLERİ**
- **🔸 CNN** - Convolutional Neural Network (Görüntü analizi)
- **🔸 LSTM** - Long Short-Term Memory (Metin analizi)
- **🔸 Multimodal** - CNN + LSTM birleşimi (Çok modaliteli)

### 📊 **DEĞERLENDİRME METRİKLERİ**
- **Accuracy** - Genel doğruluk
- **Precision** - Kesinlik
- **Recall** - Duyarlılık
- **F1-Score** - Harmonic mean
- **ROC-AUC** - Receiver Operating Characteristic
- **Confusion Matrix** - Karışıklık matrisi

### 📈 **GÖRSEL ÇIKTILAR**
- **Training/Validation Curves** - Eğitim eğrileri (Loss, Accuracy, Precision, Recall)
- **Confusion Matrix** - Her model için karışıklık matrisi
- **ROC Curves** - AUC skorları ile ROC eğrileri
- **Model Comparison** - Model karşılaştırma grafikleri

## 🚀 **Nasıl Çalıştırılır**

### 1. **Gerekli Kütüphaneleri Kurun**
```bash
pip install -r requirements.txt
```

### 2. **Geleneksel ML Modelleri (Hızlı Test)**
```bash
python sentiment_analysis_project.py
```

### 3. **Derin Öğrenme Modelleri (Ana Proje)**
```bash
python deep_learning_main.py
```

### 4. **Web Uygulaması**
```bash
streamlit run sentiment_web_app.py
```

## 🏗️ **Proje Yapısı**

```
sentiment-analysis-project/
├── sentiment_analysis_project.py    # Geleneksel ML (Logistic Regression, Random Forest)
├── deep_learning_models.py          # Derin öğrenme sınıfları
├── deep_learning_main.py            # Ana derin öğrenme scripti
├── sentiment_web_app.py             # Streamlit web uygulaması
├── requirements.txt                 # Tüm bağımlılıklar (TensorFlow dahil)
├── README.md                        # Proje dokümantasyonu
└── .gitignore                       # Git yapılandırması
```

## 🔬 **Model Mimarileri**

### 🖼️ **CNN Modeli (Görüntü)**
```python
Sequential([
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### 📝 **LSTM Modeli (Metin)**
```python
Sequential([
    Embedding(vocab_size, 128),
    LSTM(256, dropout=0.3, return_sequences=True),
    LSTM(128, dropout=0.3),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### 🎭 **Multimodal Model**
- CNN branch (görüntü işleme)
- LSTM branch (metin işleme)
- Concatenation layer (birleştirme)
- Dense layers (sınıflandırma)

## 📊 **Beklenen Performans**

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| CNN | ~75-85% | ~0.80 | ~0.75 | ~0.77 | ~0.85 |
| LSTM | ~80-90% | ~0.85 | ~0.82 | ~0.83 | ~0.90 |
| Multimodal | **~85-95%** | **~0.90** | **~0.87** | **~0.88** | **~0.92** |

## 🔧 **Teknik Özellikler**

### **Derin Öğrenme Optimizasyonları**
- **Batch Normalization** - Eğitim kararlılığı
- **Dropout** - Overfitting önleme
- **Early Stopping** - Otomatik durma
- **Learning Rate Scheduling** - Adaptive öğrenme oranı
- **Data Augmentation** - Veri çeşitlendirme

### **Veri İşleme**
- **Image Preprocessing** - Normalizasyon ve resize
- **Text Tokenization** - Kelime vektörleştirme
- **Sequence Padding** - Eşit uzunluk garantisi
- **Label Encoding** - Kategorik kodlama

## 🎯 **Kullanım Alanları**

- **Sosyal Medya Analizi** - Post/comment duygu analizi
- **E-ticaret** - Ürün yorumu analizi
- **Pazarlama** - Marka duygu takibi
- **Müşteri Hizmetleri** - Otomatik kategorizasyon
- **Araştırma** - Akademik çalışmalar

## 📚 **Öğrenme Hedefleri**

### **Başlangıç Seviyesi**
- Derin öğrenme temelleri
- CNN ve LSTM mimarileri
- Çok modaliteli veri işleme

### **Orta Seviye**
- Model optimizasyonu
- Hyperparameter tuning
- Transfer learning

### **İleri Seviye**
- Custom loss functions
- Attention mechanisms
- BERT/Transformer models

## 🛠️ **Gereksinimler**

- **Python**: 3.8+
- **TensorFlow**: 2.10+
- **GPU**: Önerilen (CUDA uyumlu)
- **RAM**: 8GB+ (model boyutuna göre)
- **Disk**: 2GB+ (veri seti ve modeller için)

## 📈 **Oluşturulan Dosyalar**

Proje çalıştırıldığında şu dosyalar oluşturulur:

### **Geleneksel ML Çıktıları**
- `sentiment_distribution.png`
- `sentiment_analysis.png`

### **Derin Öğrenme Çıktıları**
- `CNN_training_curves.png`
- `CNN_confusion_matrix.png`
- `CNN_roc_curve.png`
- `LSTM_training_curves.png`
- `LSTM_confusion_matrix.png`
- `LSTM_roc_curve.png`
- `Multimodal_training_curves.png`
- `Multimodal_confusion_matrix.png`
- `Multimodal_roc_curve.png`
- `model_comparison.png`

## ⚠️ **Önemli Notlar**

- **Dataset**: Büyük dosya (.csv) GitHub'a yüklenmez (.gitignore)
- **Models**: Eğitilmiş modeller local'de saklanır
- **GPU**: CUDA yoksa CPU'da çalışır (daha yavaş)
- **Memory**: Büyük modeller için RAM kullanımına dikkat

## 🤝 **Katkıda Bulunma**

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📝 **Lisans**

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📞 **İletişim**

- **GitHub**: [ardanar](https://github.com/ardanar)
- **Proje Linki**: [sentiment-analysis-project](https://github.com/ardanar/sentiment-analysis-project)

---

**⭐ Bu projeyi beğendiyseniz star vermeyi unutmayın!** 