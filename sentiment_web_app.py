import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Duygu Analizi Uygulaması",
    layout="wide"
)

# Ana başlık
st.title("Çok Modaliteli Duygu Analizi")
st.markdown("---")

# Sidebar - Navigasyon
st.sidebar.title("Menü")
page = st.sidebar.selectbox(
    "Sayfa Seçin",
    ["Ana Sayfa", "Veri Analizi", "Model Testi", "Görselleştirmeler"]
)

@st.cache_data
def load_data():
    """Veri setini yükle"""
    try:
        df = pd.read_csv('/Users/ardanar/Downloads/dataset.csv', nrows=1000)
        df['text_length'] = df['Text'].str.len()
        df['word_count'] = df['Text'].str.split().str.len()
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        return None

@st.cache_resource
def train_model(df):
    """Model eğit ve cache'le"""
    try:
        # Veriyi hazırla
        le = LabelEncoder()
        y = le.fit_transform(df['Sentiment'])
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['Text'])
        
        # Model eğit
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, vectorizer, le
    except Exception as e:
        st.error(f"Model eğitilirken hata: {e}")
        return None, None, None

def predict_sentiment(text, model, vectorizer, label_encoder):
    """Duygu tahmini yap"""
    try:
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        sentiment = label_encoder.inverse_transform([prediction])[0]
        confidence = max(probability)
        
        return sentiment, confidence
    except Exception as e:
        return "HATA", 0.0

# Ana sayfa
if page == "Ana Sayfa":
    st.header("Proje Hakkında")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Veri Seti Özellikleri")
        st.write("""
        - **71,702 satır** veri
        - **3 sütun**: Image, Text, Sentiment
        - **Çok modaliteli** yapı
        - **Duygu etiketleri**: POSITIVE, NEGATIVE
        """)
        
        st.subheader("Kullanılan Teknolojiler")
        st.write("""
        - **Python**: Ana programlama dili
        - **Scikit-learn**: Machine Learning
        - **Pandas**: Veri manipülasyonu
        - **Streamlit**: Web arayüzü
        - **Matplotlib/Seaborn**: Görselleştirme
        """)
    
    with col2:
        st.subheader("Proje Hedefleri")
        st.write("""
        1. **Veri Keşfi**: Dataset'i anlama
        2. **Model Geliştirme**: ML algoritmaları
        3. **Görselleştirme**: Grafik ve analizler
        4. **Web Uygulaması**: Kullanıcı arayüzü
        """)
        
        st.subheader("Model Performansı")
        st.write("""
        - **Logistic Regression**: %85.7 doğruluk
        - **Random Forest**: %86.7 doğruluk
        - **F1-Score**: 0.87 (weighted avg)
        """)

# Veri analizi sayfası
elif page == "Veri Analizi":
    st.header("Veri Seti Analizi")
    
    df = load_data()
    if df is not None:
        # Temel istatistikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Kayıt", len(df))
        with col2:
            st.metric("Pozitif", len(df[df['Sentiment'] == 'POSITIVE']))
        with col3:
            st.metric("Negatif", len(df[df['Sentiment'] == 'NEGATIVE']))
        with col4:
            st.metric("Ort. Kelime", f"{df['word_count'].mean():.1f}")
        
        # Veri önizleme
        st.subheader("Veri Önizleme")
        st.dataframe(df[['Text', 'Sentiment', 'text_length', 'word_count']].head(10))
        
        # İstatistiksel özet
        st.subheader("İstatistiksel Özet")
        st.write(df[['text_length', 'word_count']].describe())

# Model test sayfası
elif page == "Model Testi":
    st.header("Canlı Duygu Analizi Testi")
    
    df = load_data()
    if df is not None:
        model, vectorizer, le = train_model(df)
        
        if model is not None:
            st.success("Model başarıyla yüklendi!")
            
            # Metin girişi
            user_text = st.text_area(
                "Analiz etmek istediğiniz metni girin:",
                height=100,
                placeholder="Örnek: I love this movie! It's amazing!"
            )
            
            if st.button("Duygu Analizi Yap"):
                if user_text.strip():
                    sentiment, confidence = predict_sentiment(user_text, model, vectorizer, le)
                    
                    # Sonuçları göster
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if sentiment == "POSITIVE":
                            st.success(f"**{sentiment}**")
                        else:
                            st.error(f"**{sentiment}**")
                    
                    with col2:
                        st.info(f"Güven: **{confidence:.2%}**")
                    
                    # Güven seviyesi göstergesi
                    st.progress(confidence)
                    
                else:
                    st.warning("Lütfen bir metin girin!")
            
            # Örnek metinler
            st.subheader("Örnek Metinler")
            examples = [
                "I absolutely love this product! It's fantastic!",
                "This is the worst experience I've ever had.",
                "The weather is okay today, nothing special.",
                "Amazing service! Highly recommended!",
                "Terrible quality, very disappointed."
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"Test {i+1}: {example[:50]}...", key=f"example_{i}"):
                    sentiment, confidence = predict_sentiment(example, model, vectorizer, le)
                    st.write(f"**Metin:** {example}")
                    st.write(f"**Tahmin:** {sentiment} (Güven: {confidence:.2%})")

# Görselleştirmeler sayfası
elif page == "Görselleştirmeler":
    st.header("Veri Görselleştirmeleri")
    
    df = load_data()
    if df is not None:
        # Duygu dağılımı
        st.subheader("Duygu Dağılımı")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            sentiment_counts = df['Sentiment'].value_counts()
            ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
            ax.set_title('Duygu Dağılımı')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax)
            ax.set_title('Duygu Frekansları')
            ax.set_xlabel('Duygu')
            ax.set_ylabel('Sayı')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Metin uzunluğu analizi
        st.subheader("Metin Uzunluğu Analizi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Sentiment', y='text_length', ax=ax)
            ax.set_title('Duygu vs Metin Uzunluğu')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='text_length', hue='Sentiment', alpha=0.7, ax=ax)
            ax.set_title('Metin Uzunluğu Dağılımı')
            st.pyplot(fig)
        
        # Kelime sayısı analizi
        st.subheader("Kelime Sayısı Analizi")
        
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Sentiment', y='word_count', ax=ax)
        ax.set_title('Duygu vs Kelime Sayısı')
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Çok Modaliteli Duygu Analizi Projesi</p>
    </div>
    """, 
    unsafe_allow_html=True
) 