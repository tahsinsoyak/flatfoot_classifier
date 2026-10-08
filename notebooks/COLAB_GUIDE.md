# 🚀 Google Colab & FootArchNet Quickstart Guide / Hızlı Başlangıç Kılavuzu

Bu kılavuz, yerel bilgisayardaki donanım kısıtlamalarına takılmadan **Google Colab'in ücretsiz/Pro GPU'ları (NVIDIA T4 / V100 / A100)** üzerinde yeni özgün modelimiz **FootArchNet**'i eğitmeniz, test etmeniz ve Grad-CAM ısı haritalarını çıkarmanız için hazırlanmıştır.

---

## 🌟 1. Özgün Modelimiz: FootArchNet Nedir? (Neden Standart Modellerden Farklı?)

Geleneksel derin öğrenme modelleri (ResNet-50, ConvNeXt, EfficientNet vb.) kare ve doğal nesne fotoğrafları (ImageNet) için tasarlanmıştır. Ayak röntgenlerinde ise:
1. **Anizotropik Geometri (Genişlik vs. Yükseklik):** Medial longitudinal ark, posterior kalkaneustan anterior 1. metatarsa kadar uzanan yatay bir hat boyunca uzanır. Standart kare konvolüsyonlar bu uzunlamasına biyomekaniği tek başına yakalamakta zorlanır.
2. **Çok Ölçekli Anatomik İhtiyaç:**
   - **Mikro/Mezo Düzey:** Talonaviküler eklem aralığı, trabeküler kemik deseni ve kortikal konturlar.
   - **Makro Düzey:** Genel kavis yüksekliği (Meary hattı açılanması ve kalkaneal eğim).
3. **FootArchNet Mimarisi:**
   - **Biyomekanik Şerit Havuzlama (Biomechanical Strip Pooling - BSAM):** Yatay ve dikey şerit konvolüsyonları ile boylamsal ark boyunca uzun menzilli ilişkileri modeller.
   - **Çok Ölçekli Özellik Füzyonu (Multi-Scale Fusion):** Ara katman eklem detayları ile üst katman kavis morfolojisini rezonanse eder.
   - **Biyomekanik Dikkat Geçidi (Biomechanical Attention):** Grad-CAM'in de odaklandığı kritik yük aktarım merkezlerini (naviküler-kuneiform çöküş bölgesi) dinamik olarak öne çıkarır.
   - **İkili Havuzlama Başlığı (Dual-Pooling):** Global AvgPool + Global MaxPool birleşimiyle 768 boyutlu sağlam gömme (embedding) vektörü üretir.
   - **Hafif & Hızlı:** Yalnızca **8.35 Milyon parametre** (ResNet-50'nin 25.6M parametresine kıyasla 3 kat daha kompakt, 512×512 çözünürlükte çok hızlı çalışır).

---

## 📦 2. Hazırlık: Veri Seti Zip Dosyası (Sadece 90 MB!)

Tüm 1.529 adet 512×512 kanonik ayak görseli ve veri bölünme manifestoları (`train.csv`, `val.csv`, `test.csv`) tek bir sıkıştırılmış zip dosyası haline getirilmiştir:
- **Yerel Dosya Yolu:** `data/flatfoot_processed_dataset.zip` (~90 MB)

### Adım 1: Google Drive'a Yükleme (1 Dakika)
1. Tarayıcınızda [Google Drive](https://drive.google.com)'ı açın.
2. Bilgisayarınızdaki `C:\Users\tahsinsoyak\Desktop\proje_github_clone\flatfoot_classifier\data\flatfoot_processed_dataset.zip` dosyasını doğrudan Google Drive ana dizininize (`MyDrive/`) sürükleyip bırakın.

---

## 💻 3. Google Colab'de Çalıştırma (Adım Adım)

### Adım 2: Notebook'u Colab'de Açma
1. Tarayıcınızda [Google Colaboratory (colab.research.google.com)](https://colab.research.google.com) adresine gidin.
2. **Yükle (Upload)** sekmesine tıklayın ve depomuzdaki:
   `notebooks/Flatfoot_Colab_Training.ipynb` dosyasını seçip yükleyin.
3. Üst menüden **Çalışma Zamanı (Runtime) > Çalışma zamanı türünü değiştir (Change runtime type)** seçeneğine gidin.
4. Donanım Hızlandırıcı (Hardware Accelerator) olarak **GPU (T4 veya üzeri)** seçin ve Kaydet'e basın.

### Adım 3: Hücreleri Sırayla Çalıştırma
Notebook içerisindeki hücreler tamamen anahtar teslim ve açıklamalıdır:
- **Hücre 1 (Donanım Kontrolü):** `!nvidia-smi` ile GPU'nuzu (T4 / V100 / A100) doğrular.
- **Hücre 2 (Kod Tabanı):** Depo dosyalarını yükler ve Python arama yoluna ekler.
- **Hücre 3 (Kütüphaneler):** `albumentations`, `timm` gibi gerekli paketleri kurar.
- **Hücre 4 (Veri Seti Açma):** Google Drive'a yüklediğiniz `flatfoot_processed_dataset.zip` dosyasını otomatik olarak bulur ve saniyeler içinde Colab çalışma alanına açar (1.070 eğitim, 229 doğrulama, 230 test vakası).
- **Hücre 5 (FootArchNet Mimarisi):** 8.35M parametreli özgün mimariyi oluşturur ve tensör boyutlarını test eder.
- **Hücre 6 (Eğitim):** AMP (FP16), AdamW, sınıf ağırlıklı Cross-Entropy ve Cosine Annealing takvimi ile 20 epok boyunca eğitimi başlatır.
- **Hücre 7 (Bağımsız Test Değerlendirmesi):** Modelin eğitimde hiç görmediği **230 bağımsız klinik test vakası** üzerindeki Doğruluk, Hassasiyet (Sensitivity), Özgüllük (Specificity), F1 ve ROC-AUC metriklerini hesaplar; ResNet-50 ve EfficientNet-B2 ile karşılaştırma tablosunu ve ROC eğrisini çizer.
- **Hücre 8 (Grad-CAM Görselleştirme):** Test hastaları üzerinde Grad-CAM ısı haritalarını çıkararak anatomik ark çöküşüne odaklandığını doğrular.
- **Hücre 9 (Google Drive'a Yedekleme):** En iyi model ağırlıklarını (`best_model.pt`) ve oluşturulan tüm yüksek çözünürlüklü grafikleri doğrudan Google Drive'ınızda `flatfoot_classifier_results/` klasörüne kopyalar.

---

## 📊 4. Hedeflenen Başarım ve Kıyaslama Tablosu

| Model Mimarisi | Parametre | Test Doğruluğu (Acc) | Klinik Duyarlılık (Recall) | Özgüllük (Spec) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | 25.6 M | %84.35 | **%94.89** | %68.82 | 0.9134 |
| **ConvNeXt-Tiny** | 28.6 M | %84.35 | %94.16 | %69.89 | 0.9152 |
| **EfficientNet-B2** | 9.2 M | **%86.09** | %91.24 | **%78.49** | **0.9222** |
| **FootArchNet (Özgün Modelimiz)** | **8.35 M** | *Hedef: > %87* | *Hedef: > %94* | *Hedef: > %80* | *Hedef: > 0.93* |

Colab üzerinde eğitimi tamamladıktan sonra ortaya çıkan sonuçları ve grafikleri doğrudan makalemize Table 2 ve Figure 4 olarak dahil edeceğiz!
