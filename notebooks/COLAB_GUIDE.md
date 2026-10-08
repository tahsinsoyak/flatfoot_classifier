# 🚀 Google Colab Yüksek Doğruluk (High-Accuracy) Eğitim Rehberi

Bu kılavuz, yerel bilgisayardaki 4.0 GB VRAM kısıtlamasına takılmadan **Google Colab'in güçlü GPU'ları (NVIDIA T4 15 GB / A100 40 GB)** üzerinde teşhis doğruluğunu (Accuracy) **%92 - %95** bandına taşımak için hazırlanmıştır.

---

## 🎯 Doğruluğu Daha Yukarı Çekmenin 5 Temel Stratejisi

Yerel bilgisayarda (RTX 3050 Ti) bellek kısıtından dolayı batch_size=8 ve küçük modeller (`swin_t`, `resnet50`) kullanmak zorundaydık. Google Colab'in 15–40 GB VRAM'i ile şu kritik teknikleri devreye sokuyoruz:

1. **Ağır Model Mimarileri (Heavyweight Backbones):**
   - **`foot_arch_net_ultra`:** Çift akışlı (makro tam ayak + mikro kavis zoom) çapraz dikkatli özgün modelimiz (tekil modelde 0.9506 AUC rekoru).
   - **`convnext_base`:** 88 Milyon parametreli, 7×7 derinlemesine konvolüsyonlu modern ConvNet.
   - **`swin_b`:** 88 Milyon parametreli Shifted-Window Vision Transformer Base.
   - **`densenet201`:** 201 katmanlı kemik deseni özellik aktarımı.

2. **🚀 5-Fold Stratified Çapraz Doğrulama (K-Fold Ensemble):**
   - Tek bir veri ayrımında geliştirme verisinin %15'i eğitim dışında kalır.
   - 5-Fold boru hattı (`scripts/train_kfold.py`) veriyi 5 eşit katmana böler, 5 farklı model eğitir ve test setinde 5 modelin tahminlerini birleştirir.
   - Bu yöntem rastgele veri varyansını sıfırlar ve **doğruluğu tek başına +2% ile +4% yukarı taşır!**

3. **Çok Ölçekli Test-Time Augmentation (TTA):**
   - Test aşamasında her görseli 480px, 512px ve 544px ölçeklerinde değerlendirip olasılıkları ortalar.

4. **Model EMA (Exponential Moving Average, $\beta=0.999$):**
   - Eğitim sonlarında ağırlıkların hareketli ortalamasını alarak genelleme yeteneğini maksimize eder.

5. **Süper Ensemble V3:**
   - FootArchNet-Ultra + ConvNeXt-Base + Swin-B + DenseNet-201 tahminlerini birleştirerek yanlış alarmları en aza indirir.

---

## 📦 1. Hazırlık: Veri Seti Zip Dosyası (Sadece 48.5 MB!)

Tüm 1.529 adet 512×512 kanonik ayak görseli ve veri bölünme manifestoları (`train.csv`, `val.csv`, `test.csv`) tek bir sıkıştırılmış zip dosyası haline getirilmiştir:
- **Yerel Dosya Yolu:** `data/flatfoot_processed_dataset.zip` (~48.5 MB)

### Google Drive'a Yükleme (1 Dakika):
1. [Google Drive](https://drive.google.com)'ınızı açın.
2. Bilgisayarınızdaki `C:\Users\tahsinsoyak\Desktop\proje_github_clone\flatfoot_classifier\data\flatfoot_processed_dataset.zip` dosyasını Google Drive ana dizininize (`MyDrive/`) yükleyin.
*(Alternatif: Colab açıkken doğrudan sol taraftaki dosya gezgini paneline de sürükleyip bırakabilirsiniz!)*

---

## 💻 2. Google Colab'de Çalıştırma (Adım Adım)

### Adım 1: Notebook'u Colab'de Açma
1. Tarayıcınızda [Google Colaboratory (colab.research.google.com)](https://colab.research.google.com) adresine gidin.
2. **Yükle (Upload)** sekmesine tıklayın ve depomuzdaki:
   `notebooks/Flatfoot_Colab_Training.ipynb` dosyasını seçip yükleyin.
3. Üst menüden **Çalışma Zamanı (Runtime) > Çalışma zamanı türünü değiştir (Change runtime type)** seçeneğine gidin.
4. Donanım Hızlandırıcı (Hardware Accelerator) olarak **GPU (T4 veya A100)** seçin ve Kaydet'e basın.

### Adım 2: Hücreleri Sırayla Çalıştırma
Notebook içerisindeki hücreler tamamen anahtar teslimdir:
- **Hücre 1 (Donanım Kontrolü):** `!nvidia-smi` ile GPU'nuzu (T4 / A100) doğrular.
- **Hücre 2 (Kod Tabanı):** Depoyu GitHub'dan çeker (`!git clone https://github.com/tahsinsoyak/flatfoot_classifier.git`).
- **Hücre 3 (Kütüphaneler):** Gerekli paketleri kurar (`albumentations`, `timm`, `scikit-learn` vb.).
- **Hücre 4 (Veri Seti):** Google Drive'daki `flatfoot_processed_dataset.zip` dosyasını otomatik bulur ve saniyeler içinde açar.
- **Hücre 5 (Model Seçimi):** `foot_arch_net_ultra`, `convnext_base`, `swin_b` veya `densenet201` seçmenizi sağlar.
- **Hücre 6 (Tekil Model Eğitimi):** Batch size = 16 ve Model EMA ile 20 epoch eğitir.
- **Hücre 7 (Test Değerlendirmesi & TTA):** 229 klinik test hastasında metrikleri ve karmaşıklık matrisini çıkarır.
- **Hücre 8 (🚀 5-Fold Çapraz Doğrulama):**
  ```python
  !python scripts/train_kfold.py --model foot_arch_net_ultra --folds 5 --epochs 20 --batch-size 16 --img-size 512
  ```
  *(5 model eğitip ensemble ederek en yüksek doğruluğu yakalar).*
- **Hücre 9 (Süper Ensemble):** Çoklu mimari harmanlamasını çalıştırır.
- **Hücre 10 (Grad-CAM):** Medial kavis ve kalkaneus üzerindeki anatomik ısı haritalarını görselleştirir.
- **Hücre 11 (Google Drive'a Yedekleme):** En iyi ağırlıkları (`*.pt`), tahminleri ve grafikleri otomatik olarak Google Drive'ınızda `flatfoot_classifier_results/` klasörüne kopyalar.

---

## 📊 Kıyaslama Hedefleri

| Model Mimarisi | Yaklaşım | Test Doğruluğu (Acc) | Özgüllük (Spec) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet-50 Baseline** | Standart | %84.35 | %68.82 | 0.9134 |
| **FootArchNet-Ultra** | Tekil Model (TTA) | %87.77 - %89.52 | %84.95 | 0.9506 |
| **Süper Ensemble V2** | 5 Model Harmanı | **%90.83** | **%93.55** | **0.9589** |
| 🎯 **5-Fold FootArchNet / ConvNeXt (Colab)** | **5-Fold Ensemble + TTA** | **Hedef: %92.0 - %94.5+** | **Hedef: > %94** | **Hedef: > 0.965** |
