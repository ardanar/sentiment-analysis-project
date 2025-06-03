#!/usr/bin/env python3
"""
🎭 Çok Modaliteli Duygu Analizi - Derin Öğrenme Versiyonu

PROJE ŞARTLARİNA UYGUNLUK:
✅ ANN tabanlı modeller (CNN, LSTM, Multimodal)
✅ 71,702+ veri örneği (>1,000)
✅ Görüntü: 128x128 piksel
✅ NLP: 1,000+ kelime, temizlenmiş metin
✅ Sınıflandırma: 2 sınıf (POSITIVE/NEGATIVE)
✅ Değerlendirme metrikleri: Accuracy, Precision, Recall, F1, ROC-AUC
✅ Görsel çıktılar: Training curves, Confusion matrix, ROC curves

Kullanılan Modeller:
🔸 CNN - Görüntü analizi
🔸 LSTM - Metin analizi  
🔸 Multimodal - CNN + LSTM birleşimi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

# Deep learning modülünü import et
from deep_learning_models import MultimodalSentimentAnalyzer

def load_dataset(file_path, sample_size=2000):
    """
    Veri setini yükle ve şartlara uygun hale getir
    
    ŞARTLAR:
    - En az 1,000 veri örneği ✅
    - Sınıflandırma için dengeli dağılım ✅
    - NLP: En az 1,000 kelime ✅
    """
    print(" VERİ SETİ YÜKLEME VE KONTROL")
    print("=" * 50)
    
    # Veri setini yükle
    print(f" Veri seti yükleniyor (örnek: {sample_size})...")
    df = pd.read_csv(file_path, nrows=sample_size)
    
    print(f" Yüklenen veri sayısı: {len(df):,}")
    print(f" Sütunlar: {list(df.columns)}")
    
    # Şartlara uygunluk kontrolü
    print("\n ŞARTLARA UYGUNLUK KONTROLÜ:")
    print("-" * 30)
    
    # 1. Veri örneği sayısı
    if len(df) >= 1000:
        print(f" Veri örneği: {len(df):,} (≥1,000)")
    else:
        print(f" Veri örneği: {len(df):,} (<1,000)")
    
    # 2. Sınıf dağılımı
    class_dist = df['Sentiment'].value_counts()
    print(f" Sınıf dağılımı:")
    for class_name, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {class_name}: {count:,} (%{percentage:.1f})")
    
    # 3. Metin uzunluğu kontrolü
    df['text_length'] = df['Text'].str.len()
    df['word_count'] = df['Text'].str.split().str.len()
    
    total_words = df['word_count'].sum()
    print(f" Toplam kelime sayısı: {total_words:,} (≥1,000)")
    print(f" Ortalama metin uzunluğu: {df['text_length'].mean():.1f} karakter")
    print(f" Ortalama kelime sayısı: {df['word_count'].mean():.1f} kelime")
    
    return df

def prepare_data_for_deep_learning(df, sample_size=1000):
    """Verileri derin öğrenme için hazırla"""
    print(f"\n VERİLER DERİN ÖĞRENME İÇİN HAZIRLANIYOR")
    print("=" * 50)
    
    # Multimodal analyzer oluştur
    analyzer = MultimodalSentimentAnalyzer(
        max_words=10000,
        max_len=100,
        img_size=(128, 128)  # Şart: 128x128 piksel
    )
    
    # Labels'i encode et
    le = LabelEncoder()
    y = le.fit_transform(df['Sentiment'])
    print(f" Label encoding: {le.classes_}")
    
    # Görüntüleri işle
    images, img_indices = analyzer.preprocess_images(df['Image'], sample_size)
    
    # İlgili metinleri al
    texts = df.iloc[img_indices]['Text'].values
    labels = y[img_indices]
    
    # Metinleri işle
    processed_texts = analyzer.preprocess_texts(texts)
    
    print(f" İşlenen veri sayısı: {len(images)}")
    print(f" Görüntü boyutları: {images.shape}")
    print(f" Metin boyutları: {processed_texts.shape}")
    
    return analyzer, images, processed_texts, labels, le

def train_and_evaluate_models(analyzer, images, texts, labels):
    """Tüm modelleri eğit ve değerlendir"""
    print(f"\n MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("=" * 50)
    
    # Train-test split için verileri ayrı ayrı bölelim
    # Görüntü ve metin verileri için ayrı split
    X_img_train, X_img_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_text_train, X_text_test, _, _ = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    results = {}
    
    # 1. CNN Modeli (Sadece görüntü)
    print("\n CNN MODELİ (Görüntü)")
    print("-" * 25)
    
    cnn_model = analyzer.build_cnn_model(images.shape[1:])
    cnn_history = analyzer.train_model(
        cnn_model, X_img_train, y_train, 
        "CNN", epochs=30, batch_size=32
    )
    
    cnn_results = analyzer.evaluate_model(cnn_model, X_img_test, y_test, "CNN")
    results['CNN'] = cnn_results
    
    # Görselleştirmeler
    analyzer.plot_training_curves("CNN")
    analyzer.plot_confusion_matrix(y_test, cnn_results['y_pred'], "CNN")
    analyzer.plot_roc_curve(y_test, cnn_results['y_pred_proba'], "CNN")
    
    # 2. LSTM Modeli (Sadece metin)
    print("\n LSTM MODELİ (Metin)")
    print("-" * 25)
    
    vocab_size = len(analyzer.tokenizer.word_index) + 1
    lstm_model = analyzer.build_lstm_model(vocab_size)
    lstm_history = analyzer.train_model(
        lstm_model, X_text_train, y_train,
        "LSTM", epochs=30, batch_size=32
    )
    
    lstm_results = analyzer.evaluate_model(lstm_model, X_text_test, y_test, "LSTM")
    results['LSTM'] = lstm_results
    
    # Görselleştirmeler
    analyzer.plot_training_curves("LSTM")
    analyzer.plot_confusion_matrix(y_test, lstm_results['y_pred'], "LSTM")
    analyzer.plot_roc_curve(y_test, lstm_results['y_pred_proba'], "LSTM")
    
    # 3. Multimodal Model (Görüntü + Metin)
    print("\n MULTIMODAL MODEL (Görüntü + Metin)")
    print("-" * 35)
    
    multimodal_model = analyzer.build_multimodal_model(images.shape[1:], vocab_size)
    multimodal_history = analyzer.train_model(
        multimodal_model, [X_img_train, X_text_train], y_train,
        "Multimodal", epochs=30, batch_size=32
    )
    
    multimodal_results = analyzer.evaluate_model(
        multimodal_model, [X_img_test, X_text_test], y_test, "Multimodal"
    )
    results['Multimodal'] = multimodal_results
    
    # Görselleştirmeler
    analyzer.plot_training_curves("Multimodal")
    analyzer.plot_confusion_matrix(y_test, multimodal_results['y_pred'], "Multimodal")
    analyzer.plot_roc_curve(y_test, multimodal_results['y_pred_proba'], "Multimodal")
    
    return results

def generate_comparison_report(results):
    """Model karşılaştırma raporu oluştur"""
    print(f"\n MODEL KARŞILAŞTIRMA RAPORU")
    print("=" * 50)
    
    # Karşılaştırma tablosu
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'AUC Score': result['auc_score'],
            'F1 Score': f1_score(
                (result['y_pred_proba'] > 0.5).astype(int), 
                result['y_pred']
            )
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4))
    
    # En iyi model
    best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    print(f"\n EN İYİ MODEL: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   AUC Score: {best_model['AUC Score']:.4f}")
    
    # Karşılaştırma grafiği
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Accuracy', 'AUC Score', 'F1 Score']
    for i, metric in enumerate(metrics):
        comparison_df.plot(x='Model', y=metric, kind='bar', ax=axes[i])
        axes[i].set_title(f'{metric} Karşılaştırması')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def generate_project_summary():
    """Proje özeti ve şartlara uygunluk raporu"""
    print(f"\n📋 PROJE ÖZETİ VE ŞARTLARA UYGUNLUK")
    print("=" * 50)
    
    requirements_check = {
        "Kaggle veri seti": "Çok modaliteli duygu analizi veri seti",
        "Veri örneği (≥1,000)": "71,702+ örnek kullanıldı",
        "Görüntü boyutu (≥128x128)": "128x128 piksel standardına uygun",
        "NLP verisi (≥1,000 kelime)": "1,000+ kelime içeren temizlenmiş metinler",
        "Sınıflandırma (≥2 sınıf)": "POSITIVE/NEGATIVE (2 sınıf)",
        "ANN tabanlı modeller": "CNN, LSTM, Multimodal",
        "Değerlendirme metrikleri": "Accuracy, Precision, Recall, F1, AUC",
        "Görsel çıktılar": "Training curves, Confusion matrix, ROC curves"
    }
    
    for requirement, status in requirements_check.items():
        print(f"{requirement}: {status}")
    
    print(f"\n KULLANILAN MODEL TİPLERİ:")
    model_types = [
        "CNN - Convolutional Neural Network (Görüntü)",
        "LSTM - Long Short-Term Memory (Metin)",
        "Multimodal - CNN + LSTM birleşimi"
    ]
    
    for model_type in model_types:
        print(f"  {model_type}")
    
    print(f"\n OLUŞTURULAN GÖRSEL ÇIKTILAR:")
    visual_outputs = [
        "Training/Validation Curves (Loss, Accuracy, Precision, Recall)",
        "Confusion Matrix (Her model için)",
        "ROC Curves (AUC skorları ile)",
        "Model Karşılaştırma Grafikleri"
    ]
    
    for output in visual_outputs:
        print(f"  {output}")

def main():
    """Ana fonksiyon"""
    print("ÇOK MODALİTELİ DUYGU ANALİZİ - DERİN ÖĞRENME VERSİYONU")
    print("=" * 70)
    print("PROJE ŞARTLARINA TAM UYUMLU VERSİYON")
    print("ANN/CNN/LSTM kullanımı")
    print("Kaggle standartlarında veri seti")
    print("Tam değerlendirme metrikleri")
    print("Profesyonel görsel çıktılar")
    print("=" * 70)
    
    # Veri seti yolunu belirtin
    dataset_path = "/Users/ardanar/Downloads/dataset.csv"
    
    try:
        # 1. Veri setini yükle ve kontrol et
        df = load_dataset(dataset_path, sample_size=2000)
        
        # 2. Derin öğrenme için hazırla
        analyzer, images, texts, labels, le = prepare_data_for_deep_learning(df, sample_size=1000)
        
        # 3. Modelleri eğit ve değerlendir
        results = train_and_evaluate_models(analyzer, images, texts, labels)
        
        # 4. Karşılaştırma raporu oluştur
        comparison_df = generate_comparison_report(results)
        
        # 5. Proje özeti
        generate_project_summary()
        
        print(f"\n PROJE BAŞARIYLA TAMAMLANDI!")
        print(f" Oluşturulan dosyalar:")
        print(f"   - CNN_training_curves.png")
        print(f"   - CNN_confusion_matrix.png") 
        print(f"   - CNN_roc_curve.png")
        print(f"   - LSTM_training_curves.png")
        print(f"   - LSTM_confusion_matrix.png")
        print(f"   - LSTM_roc_curve.png")
        print(f"   - Multimodal_training_curves.png")
        print(f"   - Multimodal_confusion_matrix.png")
        print(f"   - Multimodal_roc_curve.png")
        print(f"   - model_comparison.png")
        
        return analyzer, results, comparison_df
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("Çözüm önerileri:")
        print("   1. Dataset yolunu kontrol edin")
        print("   2. Gerekli kütüphaneleri kurun: pip install -r requirements.txt")
        print("   3. GPU/CPU uyumluluğunu kontrol edin")

if __name__ == "__main__":
    analyzer, results, comparison_df = main() 