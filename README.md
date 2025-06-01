# 🎭 Çok Modaliteli Duygu Analizi Projesi

Bu proje, görüntü ve metin verilerini birlikte kullanarak duygu analizi yapan bir makine öğrenmesi projesidir.

## 📊 Veri Seti Hakkında

- **Boyut**: ~71,702 satır
- **Sütunlar**:
  - `Image`: Görüntü verileri (numpy array formatında)
  - `Text`: Metin verileri (İngilizce cümleler)
  - `Sentiment`: Duygu etiketleri (POSITIVE, NEGATIVE, vb.)

## 🚀 Nasıl Çalıştırılır

### 1. Gerekli Kütüphaneleri Kurun
```bash
pip install -r requirements.txt
```

### 2. Python Scriptini Çalıştırın
```bash
python sentiment_analysis_project.py
```

## 🔍 Proje Özellikleri

### ✅ Yapılanlar:
- **Veri Keşfi**: Veri setinin genel yapısını anlama
- **Veri Görselleştirme**: Duygu dağılımları ve desenler
- **Metin Analizi**: TF-IDF kullanarak özellik çıkarma
- **Model Eğitimi**: Logistic Regression ve Random Forest
- **Performans Değerlendirme**: Doğruluk ve classification report

### 📈 Çıktılar:
- `sentiment_distribution.png`: Duygu dağılım grafiği
- `sentiment_analysis.png`: Detaylı analiz grafikleri
- Model performans raporları

## 🤖 Kullanılan Modeller

1. **Logistic Regression**: Hızlı ve etkili doğrusal model
2. **Random Forest**: Ensemble öğrenme yöntemi

## 📊 Analiz Edilen Desenler

- Duygu türlerine göre metin uzunluğu
- Kelime sayısı dağılımları
- En sık kullanılan kelimeler
- Duygu dağılım yüzdeleri

## 🔮 Gelecekteki Geliştirmeler

- **Görüntü Analizi**: CNN kullanarak görüntü özelliklerini çıkarma
- **Derin Öğrenme**: LSTM/BERT gibi gelişmiş modeller
- **Çok Modaliteli Füzyon**: Görüntü + metin birleştirme
- **Web Uygulaması**: Flask/Streamlit ile interaktif arayüz

## 📚 Önerilen Çalışma Yöntemleri

### Başlangıç Seviyesi:
1. Veri keşfi ve görselleştirme
2. Basit metin analizi
3. Temel makine öğrenmesi modelleri

### Orta Seviye:
1. Özellik mühendisliği
2. Model optimizasyonu
3. Cross-validation

### İleri Seviye:
1. Derin öğrenme modelleri
2. Transfer learning
3. Çok modaliteli yaklaşımlar

## 🛠️ Gerekli Araçlar

- Python 3.7+
- Jupyter Notebook (opsiyonel)
- Anaconda/Miniconda (önerilen)

## 📞 Yardım ve Destek

Bu proje ile ilgili sorularınız için:
- GitHub Issues kullanabilirsiniz
- Makine öğrenmesi toplulukları
- Türkiye AI topluluğu

## 🎯 Hedef Kitle

- Makine öğrenmesi öğrencileri
- Veri bilimi meraklıları
- Duygu analizi projesi yapmak isteyenler
- Çok modaliteli öğrenme ile ilgilenenler 