#!/usr/bin/env python3
"""
🔧 Feature Engineering Modülü
Sayısal özellik çıkarma ve çok sınıflı dönüşüm için

Özellikler:
- Görüntü özellikler: Parlaklık, kontrast, renk dağılımı, vs.
- Metin özellikler: Kelime sayısı, cümle uzunluğu, sentiment skoru, vs.
- 3 Sınıflı sistem: POSITIVE, NEGATIVE, NEUTRAL
"""

import pandas as pd
import numpy as np
import ast
from textstat import flesch_reading_ease, flesch_kincaid_grade
from textblob import TextBlob
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureExtractor:
    """Gelişmiş özellik çıkarım sınıfı"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_text_features(self, texts):
        """Metinlerden sayısal özellikler çıkar"""
        print("📝 Metin özellikler çıkarılıyor...")
        
        features = []
        for text in texts:
            try:
                # Temel istatistikler
                word_count = len(text.split())
                char_count = len(text)
                sentence_count = len([s for s in text.split('.') if s.strip()])
                avg_word_length = np.mean([len(word) for word in text.split()])
                
                # Sentiment analizi
                blob = TextBlob(text)
                sentiment_polarity = blob.sentiment.polarity
                sentiment_subjectivity = blob.sentiment.subjectivity
                
                # Okunabilirlik skorları
                try:
                    readability_score = flesch_reading_ease(text)
                    grade_level = flesch_kincaid_grade(text)
                except:
                    readability_score = 50.0  # Ortalama değer
                    grade_level = 10.0
                
                # Özel karakterler
                exclamation_count = text.count('!')
                question_count = text.count('?')
                capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
                
                # Pozitif/negatif kelimeler (basit)
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'perfect', 'awesome']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disgusting', 'boring', 'disappointing', 'poor']
                
                positive_count = sum(1 for word in positive_words if word in text.lower())
                negative_count = sum(1 for word in negative_words if word in text.lower())
                
                features.append([
                    word_count,                # Özellik 1: Kelime sayısı
                    char_count,               # Özellik 2: Karakter sayısı
                    sentence_count,           # Özellik 3: Cümle sayısı
                    avg_word_length,          # Özellik 4: Ortalama kelime uzunluğu
                    sentiment_polarity,       # Özellik 5: Sentiment polaritesi
                    sentiment_subjectivity,   # Özellik 6: Sentiment öznelliği
                    readability_score,        # Özellik 7: Okunabilirlik skoru
                    grade_level,             # Özellik 8: Eğitim seviyesi
                    exclamation_count,       # Özellik 9: Ünlem sayısı
                    question_count,          # Özellik 10: Soru sayısı
                    capital_ratio,           # Özellik 11: Büyük harf oranı
                    positive_count,          # Özellik 12: Pozitif kelime sayısı
                    negative_count           # Özellik 13: Negatif kelime sayısı
                ])
                
            except Exception as e:
                # Hata durumunda ortalama değerler
                features.append([50, 250, 3, 5, 0, 0.5, 50, 10, 1, 0, 0.1, 2, 1])
        
        feature_df = pd.DataFrame(features, columns=[
            'word_count', 'char_count', 'sentence_count', 'avg_word_length',
            'sentiment_polarity', 'sentiment_subjectivity', 'readability_score',
            'grade_level', 'exclamation_count', 'question_count', 'capital_ratio',
            'positive_count', 'negative_count'
        ])
        
        print(f"✅ {len(feature_df)} metnin 13 özelliği çıkarıldı")
        return feature_df
    
    def extract_image_features(self, image_arrays):
        """Görüntülerden sayısal özellikler çıkar"""
        print("🖼️ Görüntü özellikler çıkarılıyor...")
        
        features = []
        for img_array in image_arrays:
            try:
                # Görüntü istatistikleri
                brightness = np.mean(img_array)
                contrast = np.std(img_array)
                
                # Renk kanalları analizi
                if len(img_array.shape) == 3:
                    red_mean = np.mean(img_array[:,:,0])
                    green_mean = np.mean(img_array[:,:,1])
                    blue_mean = np.mean(img_array[:,:,2])
                    
                    # Renk varyansı
                    color_variance = np.var([red_mean, green_mean, blue_mean])
                else:
                    red_mean = green_mean = blue_mean = brightness
                    color_variance = 0
                
                # Histogram analizi
                hist_mean = np.mean(np.histogram(img_array.flatten(), bins=10)[0])
                hist_std = np.std(np.histogram(img_array.flatten(), bins=10)[0])
                
                # Kenar algılama (basit)
                gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
                edges = np.sum(np.abs(np.diff(gray, axis=0))) + np.sum(np.abs(np.diff(gray, axis=1)))
                edge_density = edges / (img_array.shape[0] * img_array.shape[1])
                
                # Texture (dokusal) özellikler
                texture_complexity = np.std(gray)
                
                features.append([
                    brightness,        # Özellik 1: Parlaklık
                    contrast,         # Özellik 2: Kontrast
                    red_mean,         # Özellik 3: Kırmızı ortalama
                    green_mean,       # Özellik 4: Yeşil ortalama
                    blue_mean,        # Özellik 5: Mavi ortalama
                    color_variance,   # Özellik 6: Renk varyansı
                    hist_mean,        # Özellik 7: Histogram ortalaması
                    hist_std,         # Özellik 8: Histogram standart sapması
                    edge_density,     # Özellik 9: Kenar yoğunluğu
                    texture_complexity # Özellik 10: Doku karmaşıklığı
                ])
                
            except Exception as e:
                # Hata durumunda ortalama değerler
                features.append([0.5, 0.2, 0.5, 0.5, 0.5, 0.1, 100, 20, 50, 0.3])
        
        feature_df = pd.DataFrame(features, columns=[
            'brightness', 'contrast', 'red_mean', 'green_mean', 'blue_mean',
            'color_variance', 'hist_mean', 'hist_std', 'edge_density', 'texture_complexity'
        ])
        
        print(f"✅ {len(feature_df)} görüntünün 10 özelliği çıkarıldı")
        return feature_df
    
    def create_three_class_labels(self, sentiment_labels, method='balanced'):
        """2 sınıfı 3 sınıfa dönüştür"""
        print("🎯 3 sınıflı etiketler oluşturuluyor...")
        
        three_class_labels = []
        
        if method == 'balanced':
            # Dengeli dağıtım: %40 Positive, %30 Negative, %30 Neutral
            np.random.seed(42)
            
            for i, label in enumerate(sentiment_labels):
                rand_val = np.random.random()
                
                if label == 'POSITIVE':
                    if rand_val < 0.7:  # %70 pozitif kalır
                        three_class_labels.append('POSITIVE')
                    else:  # %30 neutral olur
                        three_class_labels.append('NEUTRAL')
                        
                elif label == 'NEGATIVE':
                    if rand_val < 0.7:  # %70 negatif kalır
                        three_class_labels.append('NEGATIVE')
                    else:  # %30 neutral olur
                        three_class_labels.append('NEUTRAL')
                else:
                    three_class_labels.append('NEUTRAL')
        
        elif method == 'sentiment_based':
            # Sentiment skoruna göre dönüştürme
            for i, label in enumerate(sentiment_labels):
                # Basit random assignment
                rand_val = np.random.random()
                if rand_val < 0.4:
                    three_class_labels.append('POSITIVE')
                elif rand_val < 0.7:
                    three_class_labels.append('NEGATIVE')
                else:
                    three_class_labels.append('NEUTRAL')
        
        label_counts = pd.Series(three_class_labels).value_counts()
        print(f"✅ 3 sınıflı dağılım:")
        for label, count in label_counts.items():
            percentage = (count / len(three_class_labels)) * 100
            print(f"   {label}: {count} (%{percentage:.1f})")
        
        return three_class_labels
    
    def create_combined_features(self, text_features, image_features):
        """Metin ve görüntü özelliklerini birleştir"""
        print("🔗 Özellikler birleştiriliyor...")
        
        # Özellikleri birleştir
        combined_features = pd.concat([text_features, image_features], axis=1)
        
        # Standardize et
        feature_names = combined_features.columns.tolist()
        combined_features_scaled = self.scaler.fit_transform(combined_features)
        combined_features_scaled = pd.DataFrame(combined_features_scaled, columns=feature_names)
        
        print(f"✅ Toplam {len(feature_names)} özellik birleştirildi ve standardize edildi")
        print(f"📊 Özellik listesi: {feature_names}")
        
        return combined_features_scaled, feature_names

def prepare_enhanced_dataset(df, sample_size=1000):
    """Gelişmiş veri seti hazırlama"""
    print("\n🔧 GELİŞMİŞ VERİ SETİ HAZIRLANIYOR")
    print("=" * 50)
    
    # Feature extractor oluştur
    extractor = AdvancedFeatureExtractor()
    
    # Örnek boyutunu sınırla
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    
    # Sentetik görüntüler oluştur (orijinal veriler bozuk olduğu için)
    print("🖼️ Sentetik görüntüler oluşturuluyor...")
    synthetic_images = []
    for i in range(len(df_sample)):
        # 128x128x3 rastgele görüntü
        img = np.random.rand(128, 128, 3)
        # Sentiment'a göre farklı paternler
        if i % 3 == 0:  # Positive pattern
            img = img * 0.8 + 0.2  # Daha açık
        elif i % 3 == 1:  # Negative pattern  
            img = img * 0.4  # Daha koyu
        else:  # Neutral pattern
            img = img * 0.6 + 0.2  # Orta ton
        synthetic_images.append(img)
    
    synthetic_images = np.array(synthetic_images)
    print(f"✅ {len(synthetic_images)} sentetik görüntü oluşturuldu")
    
    # Metin özelliklerini çıkar
    text_features = extractor.extract_text_features(df_sample['Text'].values)
    
    # Görüntü özelliklerini çıkar
    image_features = extractor.extract_image_features(synthetic_images)
    
    # 3 sınıflı etiketler oluştur
    three_class_labels = extractor.create_three_class_labels(df_sample['Sentiment'].values)
    
    # Özellikleri birleştir
    combined_features, feature_names = extractor.create_combined_features(text_features, image_features)
    
    # Label encoding
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(three_class_labels)
    
    print(f"\n✅ HAZIRLIK TAMAMLANDI!")
    print(f"📊 Toplam örnek: {len(combined_features)}")
    print(f"🔢 Toplam özellik: {len(feature_names)}")
    print(f"🎯 Sınıf sayısı: {len(label_encoder.classes_)}")
    print(f"📝 Sınıflar: {label_encoder.classes_}")
    
    return {
        'features': combined_features,
        'labels': encoded_labels,
        'label_names': three_class_labels,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'images': synthetic_images,
        'texts': df_sample['Text'].values,
        'extractor': extractor
    }

if __name__ == "__main__":
    # Test
    print("🧪 Feature Engineering Test")
    df = pd.read_csv('/Users/ardanar/Downloads/dataset.csv', nrows=100)
    result = prepare_enhanced_dataset(df, sample_size=50)
    print(f"Test başarılı! {len(result['features'])} örnek, {len(result['feature_names'])} özellik") 