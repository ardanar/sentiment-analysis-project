#!/usr/bin/env python3
"""
🧪 Derin Öğrenme Test Scripti
Görüntü ve metin verilerini test etmek için basitleştirilmiş versiyon
"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Veri yükleme ve format kontrolü"""
    print("🔍 VERİ YÜKLEME TESTİ")
    print("=" * 40)
    
    # Veri yükle
    df = pd.read_csv('/Users/ardanar/Downloads/dataset.csv', nrows=100)
    print(f"✅ {len(df)} satır yüklendi")
    
    # İlk görüntüyü test et
    print("\n🖼️ GÖRÜNTÜ VERİSİ TESTİ:")
    for i in range(min(5, len(df))):
        try:
            img_str = df['Image'].iloc[i]
            # String'i değerlendirmeye çalış
            img_data = ast.literal_eval(img_str)
            img_array = np.array(img_data)
            
            print(f"Görüntü {i+1}:")
            print(f"  Shape: {img_array.shape}")
            print(f"  Min: {img_array.min():.3f}, Max: {img_array.max():.3f}")
            print(f"  Dtype: {img_array.dtype}")
            
            # İlk başarılı görüntüde dur
            if len(img_array.shape) >= 2:
                print(f"✅ Başarılı görüntü bulundu: {img_array.shape}")
                return df, img_array
                
        except Exception as e:
            print(f"Görüntü {i+1} hatası: {e}")
            continue
    
    print("❌ Hiç geçerli görüntü bulunamadı")
    return df, None

def test_simple_models():
    """Basit modelleri test et"""
    print("\n🤖 MODEL TESTİ")
    print("=" * 40)
    
    df, sample_image = test_data_loading()
    
    if sample_image is None:
        print("❌ Görüntü verisi olmadan model test edilemez")
        return
    
    # Labels hazırla
    le = LabelEncoder()
    y = le.fit_transform(df['Sentiment'])
    print(f"✅ Labels: {le.classes_}")
    print(f"✅ Dağılım: {np.bincount(y)}")
    
    # Metinleri hazırla
    texts = df['Text'].values
    print(f"✅ {len(texts)} metin hazırlandı")
    print(f"✅ Ortalama uzunluk: {np.mean([len(t) for t in texts]):.1f}")
    
    # Basit özellik çıkarma
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text = vectorizer.fit_transform(texts)
    print(f"✅ TF-IDF: {X_text.shape}")
    
    # Basit model test
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.3, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"✅ Random Forest Accuracy: {accuracy:.4f}")
    
    return True

def create_synthetic_data():
    """Derin öğrenme için sentetik veri oluştur"""
    print("\n🎯 SENTETİK VERİ OLUŞTURMA")
    print("=" * 40)
    
    # Gerçek metinleri al
    df = pd.read_csv('/Users/ardanar/Downloads/dataset.csv', nrows=500)
    
    # Sentetik görüntüler oluştur (128x128x3)
    n_samples = len(df)
    synthetic_images = np.random.rand(n_samples, 128, 128, 3)
    
    # Metinleri ve etiketleri al
    texts = df['Text'].values
    labels = df['Sentiment'].values
    
    print(f"✅ {n_samples} örnek oluşturuldu")
    print(f"✅ Görüntü boyutu: {synthetic_images.shape}")
    print(f"✅ Metin sayısı: {len(texts)}")
    
    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Metin tokenization (basit)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=50, padding='post')
    
    print(f"✅ Tokenized sequences: {padded_sequences.shape}")
    
    return synthetic_images, padded_sequences, y, le

def test_simple_cnn():
    """Basit CNN test et"""
    print("\n🔥 BASIT CNN TESTİ")
    print("=" * 40)
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        
        # Sentetik veri oluştur
        images, texts, labels, le = create_synthetic_data()
        
        # Basit CNN modeli
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ CNN modeli oluşturuldu")
        print(f"📊 Model parametreleri: {model.count_params():,}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        
        print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Kısa eğitim
        print("🚀 Model eğitiliyor (2 epoch)...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=2,
            batch_size=32,
            verbose=1
        )
        
        # Sonuçlar
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        print(f"✅ Final Training Accuracy: {final_acc:.4f}")
        print(f"✅ Final Validation Accuracy: {final_val_acc:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ CNN testi başarısız: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🧪 DERİN ÖĞRENME VERİ VE MODEL TESTLERİ")
    print("=" * 50)
    
    # Test 1: Veri yükleme
    success1 = test_simple_models()
    
    # Test 2: CNN
    if success1:
        success2 = test_simple_cnn()
        
        if success2:
            print("\n🎉 TÜM TESTLER BAŞARILI!")
            print("✅ Veri formatı doğru")
            print("✅ TensorFlow çalışıyor")
            print("✅ CNN modeli eğitilebilir")
            print("\n💡 Ana projeyi çalıştırmaya hazır!")
        else:
            print("\n⚠️ CNN testi başarısız - TensorFlow kurulumunu kontrol edin")
    else:
        print("\n❌ Veri testleri başarısız")

if __name__ == "__main__":
    main() 