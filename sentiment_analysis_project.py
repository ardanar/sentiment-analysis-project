import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import ast
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(file_path):
    """Veri setini yükle ve keşfet"""
    print("Veri seti yükleniyor...")
    
    # Büyük dosya için chunk'lar halinde okuma
    chunk_size = 10000
    chunks = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
        if len(chunks) * chunk_size > 50000:  # İlk 50k satırı al
            break
    
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"Veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    
    # Temel bilgiler
    print("\nVeri Seti Özeti:")
    print(df.info())
    
    # Duygu dağılımı
    print("\nDuygu Dağılımı:")
    sentiment_counts = df['Sentiment'].value_counts()
    print(sentiment_counts)
    
    # Duygu dağılımı görselleştirme
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar')
    plt.title('Duygu Dağılımı')
    plt.xlabel('Duygu')
    plt.ylabel('Frekans')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def preprocess_text_data(df):
    """Metin verilerini temizle ve hazırla"""
    print("\nMetin verileri temizleniyor...")
    
    # Null değerleri kontrol et
    print(f"Null metin sayısı: {df['Text'].isnull().sum()}")
    
    # Null değerleri temizle
    df = df.dropna(subset=['Text'])
    
    # Temel metin istatistikleri
    df['text_length'] = df['Text'].str.len()
    df['word_count'] = df['Text'].str.split().str.len()
    
    print(f"Ortalama metin uzunluğu: {df['text_length'].mean():.2f}")
    print(f"Ortalama kelime sayısı: {df['word_count'].mean():.2f}")
    
    return df

def extract_image_features(df, sample_size=1000):
    """Görüntü verilerinden basit özellikler çıkar"""
    print(f"\nGörüntü özellikler çıkarılıyor (örnek: {sample_size})...")
    
    # Örnek almak için (büyük veri seti olduğu için)
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    image_features = []
    valid_indices = []
    
    for idx, image_str in enumerate(df_sample['Image']):
        try:
            # String'i numpy array'e çevir
            image_array = ast.literal_eval(image_str)
            image_array = np.array(image_array)
            
            # Basit özellikler çıkar
            features = {
                'mean': np.mean(image_array),
                'std': np.std(image_array),
                'min': np.min(image_array),
                'max': np.max(image_array)
            }
            
            image_features.append(features)
            valid_indices.append(df_sample.index[idx])
            
        except:
            continue
    
    image_df = pd.DataFrame(image_features, index=valid_indices)
    print(f"{len(image_features)} görüntüden özellik çıkarıldı")
    
    return image_df

def train_text_sentiment_model(df):
    """Sadece metin kullanarak duygu analizi modeli eğit"""
    print("\nMetin tabanlı duygu analizi modeli eğitiliyor...")
    
    # Veriyi hazırla
    X_text = df['Text']
    y = df['Sentiment']
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Metin özelliklerini çıkar
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_text_features = vectorizer.fit_transform(X_text)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Model eğitimi
    print("Logistic Regression modeli eğitiliyor...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    print("Random Forest modeli eğitiliyor...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Değerlendirme
    models = {'Logistic Regression': lr_model, 'Random Forest': rf_model}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{name} Sonuçları:")
        print(f"Doğruluk: {accuracy:.4f}")
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return models, vectorizer, le

def analyze_sentiment_patterns(df):
    """Duygu desenlerini analiz et"""
    print("\nDuygu desenleri analiz ediliyor...")
    
    # Metin uzunluğu vs duygu
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='Sentiment', y='text_length')
    plt.title('Duygu vs Metin Uzunluğu')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='Sentiment', y='word_count')
    plt.title('Duygu vs Kelime Sayısı')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    df['Sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Duygu Dağılım Yüzdesi')
    
    plt.subplot(2, 2, 4)
    # En sık kullanılan kelimeler
    from collections import Counter
    import re
    
    all_text = ' '.join(df['Text'].astype(str))
    words = re.findall(r'\w+', all_text.lower())
    common_words = Counter(words).most_common(10)
    
    words_df = pd.DataFrame(common_words, columns=['Kelime', 'Frekans'])
    sns.barplot(data=words_df, y='Kelime', x='Frekans')
    plt.title('En Sık Kullanılan 10 Kelime')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Ana fonksiyon"""
    print("Çok Modaliteli Duygu Analizi Projesi Başlıyor!")
    print("=" * 50)
    
    # Veri setini yükle
    file_path = "/Users/ardanar/Downloads/dataset.csv"
    df = load_and_explore_data(file_path)
    
    # Veri ön işleme
    df = preprocess_text_data(df)
    
    # Duygu desenlerini analiz et
    analyze_sentiment_patterns(df)
    
    # Metin tabanlı model eğit
    models, vectorizer, label_encoder = train_text_sentiment_model(df)
    
    # Görüntü özelliklerini çıkar (opsiyonel)
    print("\nGörüntü özelliklerini de çıkarmak ister misiniz? (Bu uzun sürebilir)")
    print("Image features extraction can be added if needed")
    
    print("\nProje tamamlandı!")
    print("Oluşturulan dosyalar:")
    print("- sentiment_distribution.png")
    print("- sentiment_analysis.png")
    
    return df, models, vectorizer, label_encoder

if __name__ == "__main__":
    df, models, vectorizer, label_encoder = main() 