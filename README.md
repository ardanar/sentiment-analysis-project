# 🎭 Çok Modaliteli Duygu Analizi Projesi

## 📋 Proje Özeti

Bu proje, **akademik gereksinimlere %100 uyumlu** çok modaliteli duygu analizi sistemidir. Görüntü ve metin verilerini birlikte kullanarak gelişmiş sentiment analizi yapar.

## ✅ Akademik Şartlara Uygunluk

### 📊 **Temel Şartlar:**
- ✅ **Özellik sayısı**: 23 sayısal özellik (≥5)
- ✅ **Sınıf sayısı**: 3 sınıf - POSITIVE, NEGATIVE, NEUTRAL (≥3)
- ✅ **Veri seti**: 71,702+ örnek (≥1,000)
- ✅ **Görüntü boyutu**: 128x128 piksel (≥128x128)
- ✅ **NLP verisi**: 74,179+ kelime (≥1,000)

### 🧠 **ANN Tabanlı Modeller:**
- ✅ **CNN**: Görüntü analizi için
- ✅ **ANN**: Sayısal özellik analizi için
- ✅ **Multimodal**: CNN + Feature birleşimi
- ✅ **Geleneksel ML**: Random Forest, SVM, Gradient Boosting, MLP

### 📈 **Değerlendirme Metrikleri:**
- ✅ **Accuracy, F1-Score, Precision, Recall**
- ✅ **Confusion Matrix**
- ✅ **Model karşılaştırma grafikleri**
- ✅ **Özellik önem analizi**

## 🚀 Proje Versiyonları

### 1. **Temel Versiyon** (`sentiment_analysis_project.py`)
- Geleneksel ML modelleri
- Logistic Regression, Random Forest
- TF-IDF özellik çıkarımı

### 2. **Derin Öğrenme Versiyonu** (`deep_learning_main.py`)
- CNN, LSTM, Multimodal modeller
- 2 sınıflı sistem (POSITIVE/NEGATIVE)
- Sentetik görüntü verisi

### 3. **Gelişmiş Versiyon** ⭐ (`enhanced_deep_learning.py`)
- **23 sayısal özellik** çıkarımı
- **3 sınıflı** sistem (POSITIVE/NEGATIVE/NEUTRAL)
- **7 farklı model** karşılaştırması
- **Kapsamlı görselleştirmeler**

### 4. **Web Uygulaması** (`sentiment_web_app.py`)
- Streamlit tabanlı interaktif arayüz
- Gerçek zamanlı tahmin
- Görselleştirmeler

## 📊 Özellik Çıkarımı (23 Özellik)

### 📝 **Metin Özellikleri (13):**
1. Kelime sayısı
2. Karakter sayısı  
3. Cümle sayısı
4. Ortalama kelime uzunluğu
5. Sentiment polaritesi
6. Sentiment öznelliği
7. Okunabilirlik skoru
8. Eğitim seviyesi
9. Ünlem sayısı
10. Soru sayısı
11. Büyük harf oranı
12. Pozitif kelime sayısı
13. Negatif kelime sayısı

### 🖼️ **Görüntü Özellikleri (10):**
1. Parlaklık
2. Kontrast
3. Kırmızı kanal ortalaması
4. Yeşil kanal ortalaması
5. Mavi kanal ortalaması
6. Renk varyansı
7. Histogram ortalaması
8. Histogram standart sapması
9. Kenar yoğunluğu
10. Doku karmaşıklığı

## 🏆 Model Performansları

| Model | Accuracy | F1-Score | Açıklama |
|-------|----------|----------|----------|
| **SVM** | 61.0% | 0.543 | 🥇 En iyi geleneksel model |
| **Feature ANN** | 59.5% | 0.543 | 🥈 Sayısal özellik tabanlı |
| **Multimodal** | 59.0% | 0.531 | 🥉 CNN + Feature birleşimi |
| Random Forest | 58.5% | 0.519 | Ensemble yöntemi |
| Gradient Boosting | 52.0% | 0.489 | Boosting algoritması |
| MLP Neural Network | 51.5% | 0.514 | Çok katmanlı ANN |
| CNN | 43.0% | 0.348 | Görüntü tabanlı |

## 📁 Dosya Yapısı

```
sentiment-analysis-project/
├── 📄 enhanced_deep_learning.py      # Ana gelişmiş script (ÖNERİLEN)
├── 📄 feature_engineering.py         # Özellik çıkarım modülü
├── 📄 deep_learning_main.py          # Derin öğrenme versiyonu
├── 📄 deep_learning_models.py        # DL model sınıfları
├── 📄 sentiment_analysis_project.py  # Temel ML versiyonu
├── 📄 sentiment_web_app.py           # Streamlit web uygulaması
├── 📄 requirements.txt               # Gerekli kütüphaneler
├── 📄 README.md                      # Dokümantasyon
├── 📊 enhanced_model_comparison.png  # Model karşılaştırma grafiği
├── 📊 enhanced_analysis_details.png  # Detaylı analiz grafikleri
└── 📊 *.png                         # Diğer görselleştirmeler
```

## 🛠️ Kurulum ve Çalıştırma

### 1. **Gerekli Kütüphaneleri Kurun:**
```bash
pip install -r requirements.txt
```

### 2. **Veri Setini İndirin:**
- Kaggle'dan multimodal sentiment analysis veri setini indirin
- `/Users/ardanar/Downloads/dataset.csv` konumuna yerleştirin

### 3. **Gelişmiş Versiyonu Çalıştırın:**
```bash
python enhanced_deep_learning.py
```

### 4. **Web Uygulamasını Başlatın:**
```bash
streamlit run sentiment_web_app.py
```

## 📊 Oluşturulan Görselleştirmeler

### 🔍 **Model Karşılaştırma:**
- Accuracy ve F1-Score bar grafikleri
- Confusion matrix (en iyi model için)
- Radar chart (performans analizi)

### 📈 **Analiz Detayları:**
- Sınıf dağılımı (pie chart)
- Özellik önem analizi (Random Forest)
- Tahmin vs gerçek dağılım karşılaştırması

## 🎯 Kullanım Senaryoları

### 🎓 **Akademik Projeler:**
- ✅ Tüm şartları karşılar
- ✅ 7 farklı model karşılaştırması
- ✅ Kapsamlı görselleştirmeler
- ✅ Detaylı rapor ve metrikler

### 💼 **Endüstriyel Uygulamalar:**
- Sosyal medya sentiment analizi
- Ürün inceleme otomasyonu
- Müşteri geri bildirim analizi
- İçerik moderasyonu

### 📚 **Eğitim Amaçlı:**
- Machine Learning kavramları
- Deep Learning teknikleri
- Feature Engineering yöntemleri
- Model karşılaştırma metodları

## 🔧 Teknik Detaylar

### 📦 **Kullanılan Teknolojiler:**
- **Python 3.12**: Ana programlama dili
- **TensorFlow/Keras**: Derin öğrenme modelleri
- **Scikit-learn**: Geleneksel ML algoritmaları
- **Pandas/NumPy**: Veri işleme
- **Matplotlib/Seaborn**: Görselleştirme
- **Streamlit**: Web uygulaması
- **TextBlob**: Doğal dil işleme
- **OpenCV**: Görüntü işleme

### 🏗️ **Mimari Tasarım:**
- **Modüler yapı**: Her bileşen ayrı dosyada
- **Ölçeklenebilir**: Yeni modeller kolayca eklenebilir
- **Hata yönetimi**: Robust exception handling
- **Dokümantasyon**: Kapsamlı code comments

## 📈 Gelecek Geliştirmeler

- [ ] GPU optimizasyonu
- [ ] Gerçek görüntü verisi entegrasyonu
- [ ] API endpoint geliştirme
- [ ] Docker containerization
- [ ] Model deployment (MLOps)
- [ ] A/B testing framework

## 👨‍💻 Geliştirici

**Ardanar** 
- GitHub: [ardanar](https://github.com/ardanar/sentiment-analysis-project)
- Proje Türü: Akademik Çok Modaliteli Sentiment Analizi

## 📄 Lisans

Bu proje akademik amaçlar için geliştirilmiştir ve MIT lisansı altında dağıtılmaktadır.

---

## 🎉 Sonuç

Bu proje, **akademik gereksinimleri tam karşılayan** profesyonel bir çok modaliteli duygu analizi sistemidir. 

### ✅ **Başarılan Hedefler:**
- 📊 **23 sayısal özellik** (≥5)
- 🎯 **3 sınıflı sistem** (≥3)  
- 🧠 **7 farklı ANN/ML modeli**
- 📈 **Kapsamlı değerlendirme metrikleri**
- 🎨 **Profesyonel görselleştirmeler**

**🚀 Çalıştırmak için: `python enhanced_deep_learning.py`** 