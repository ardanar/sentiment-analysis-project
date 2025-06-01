# 🚀 GitHub'a Yükleme - Adım Adım Rehber

## 1️⃣ GitHub'da Repository Oluşturun

1. https://github.com adresine gidin
2. "+" → "New repository" tıklayın
3. **Repository name**: `sentiment-analysis-project`
4. **Description**: `🎭 Çok modaliteli duygu analizi projesi - Machine Learning`
5. **Public** seçin
6. **"Create repository"** tıklayın

## 2️⃣ Terminal'de Şu Komutları Çalıştırın

### GitHub Username'inizi değiştirin:
```bash
# GITHUB_USERNAME'i kendi kullanıcı adınızla değiştirin
git remote add origin https://github.com/GITHUB_USERNAME/sentiment-analysis-project.git

# Ana branch'i ayarlayın
git branch -M main

# Kodları GitHub'a yükleyin
git push -u origin main
```

## 3️⃣ Örnek Komutlar (Username: örnek_kullanici)

```bash
git remote add origin https://github.com/örnek_kullanici/sentiment-analysis-project.git
git branch -M main
git push -u origin main
```

## 🔐 Token Gerekirse:

Eğer şifre sorunu yaşarsanız:

1. GitHub Settings → Developer Settings → Personal Access Tokens
2. "Generate new token (classic)" tıklayın
3. "repo" yetkisini verin
4. Token'ı kopyalayın
5. Şifre yerine token'ı kullanın

## ✅ Başarı Kontrol:

Yükleme başarılı olduğunda:
- https://github.com/KULLANICI_ADINIZ/sentiment-analysis-project adresinde projenizi görebilirsiniz
- README.md otomatik görüntülenir
- Kodlarınız GitHub'da paylaşıma hazır olur

## 📱 Paylaşım:

Repository linkini şuralardan paylaşabilirsiniz:
- LinkedIn profili
- CV/Resume
- İş başvuruları
- Sosyal medya

## ⚠️ Önemli:

- Dataset dosyası (.csv) .gitignore ile hariç tutuldu
- Bu sayede GitHub'a 2.5MB'lık veri yüklenmez
- Sadece kod ve dokümantasyon paylaşılır 