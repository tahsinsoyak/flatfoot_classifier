# Basarak Çekilen Yan Ayak Radyografilerinden Derin Öğrenme Tabanlı Otomatik Düz Taban (Pes Planus) Sınıflandırması ve Anatomik Açıklanabilirlik

**Yazarlar:** Tahsin Soyak vd.  
**Kurum:** Bilgisayar Mühendisliği / Biyomedikal Mühendisliği Bölümü  
**İletişim:** [İletişim Bilgileri]  
**Tarih:** Ekim 2026  

---

## Özet

**Amaç:** Düz tabanlık (*pes planus*), medial boylamsal arkın (iç kavisin) çökmesiyle karakterize edilen ve tedavi edilmediğinde biyomekanik yürüme bozukluklarına yol açan yaygın bir kas-iskelet sistemi rahatsızlığıdır. Bilgisayar destekli geleneksel teşhis yaklaşımları; röntgen görüntülerinde çoklu kemik segmentasyonu ve anatomik nirengi noktası tespiti yaparak geometrik açıları (Meary açısı ve kalkaneal pitch açısı) hesaplamaya odaklanmıştır. Ancak bu yöntemler; milimetrik nirengi sapmalarının kümülatif açı hatası yaratması, basamak düzleminin eğim belirsizliği ve yoğun etiketleme zorluğu nedeniyle pratikte tıkanmaktadır. Bu çalışmada, nirengi noktası işaretlemesine ihtiyaç duymadan, basarak çekilen yan ayak röntgenlerinden doğrudan düz taban / normal sınıflandırması yapan uçtan uca derin öğrenme tabanlı bir sistem önerilmektedir.

**Yöntem:** Çalışmada klinik PACS arşivinden elde edilen **1.529 adet basarak çekilen yan ayak radyografisi** (908 düz taban, 621 normal) kullanılmıştır. 16-bit'ten 8-bit'e dinamik persentil pencereleme, CLAHE kontrast iyileştirme, kaval kemiği (tibia) dikey ekseniyle parmak yönü tespiti (kanonik sağ yönelim standardizasyonu) ve basamak/harf gürültülerini temizleyen otomatik ayak ilgi alanı (ROI) kırpma algoritması geliştirilmiştir. Veri seti tabakalı (stratified) olarak eğitim (%69.98, n=1.070), doğrulama (%14.98, n=229) ve bağımsız test (%15.04, n=230) kümelerine ayrılmıştır. ResNet-50, EfficientNet-B2 ve ConvNeXt-Tiny mimarileri sınıf ağırlıklı kayıp fonksiyonu ve otomatik karma duyarlılık (AMP) ile eğitilmiş; karar mekanizmaları Grad-CAM ile incelenmiştir.

**Bulgular:** Modelin eğitimde hiç görmediği 230 klinik vaka (137 düz taban, 93 normal) üzerindeki test sonuçlarında, **EfficientNet-B2** **%86.09 genel doğruluk**, **0.9222 ROC-AUC**, **%91.24 duyarlılık (sensitivity)**, **%78.49 özgüllük (specificity)** ve **0.8865 F1-skoru** ile en dengeli performansı sergilemiştir. **ResNet-50** ve **ConvNeXt-Tiny** sırasıyla **%94.89** (137 hastadan 130'u doğru tespit) ve **%94.16** klinik duyarlılık elde etmiştir. Grad-CAM ısı haritaları, modellerin arka plan gürültüsü yerine doğrudan medial boylamsal ark, naviküler-kuneiform eklem çökmesi ve kalkaneus açılanmasına odaklandığını doğrulamıştır.

**Sonuç:** Uçtan uca radyografik derin öğrenme sınıflandırması, nirengi tabanlı geleneksel açı hesaplama yöntemlerine kıyasla çok daha kararlı, hızlı ve klinik olarak doğrulanabilir bir alternatif sunmaktadır. Bu çalışma, ortopedi kliniklerinde hızlı tarama yapabilecek sistemler için güçlü bir temel oluşturmakta ve ayak anatomisine özel özgün derin öğrenme mimarilerinin geliştirilmesine zemin hazırlamaktadır.

**Anahtar Kelimeler:** Düz Taban, Pes Planus, Derin Öğrenme, Yan Ayak Röntgeni, EfficientNet, ConvNeXt, Grad-CAM, Medikal Görüntü Sınıflandırma.

---

## 1. Giriş

Düz tabanlık (*pes planus*), ayak tabanının iç kısmında yer alan medial boylamsal arkın (MLA) yüksekliğinin azalması veya tamamen zemine temas edecek şekilde çökmesi durumudur. Bu deformite; zamanında tespit edilmediğinde posterior tibial tendon disfonksiyonu (PTTD), plantar fasiit, aşil tendiniti, diz valgus açılanması ve kronik yürüme anomalilerine yol açmaktadır.

Ortopedi pratiğinde düz taban teşhisinin altın standardı, hastanın tam vücut ağırlığıyla basarak çektirdiği yan ayak radyografisidir (weight-bearing lateral radiograph). Radyologlar ve ortopedistler bu grafilerden iki temel geometrik açıyı inceler:
1. **Meary Açısı (Talus-1. Metatars Açısı):** Talus kemiği gövde ekseni ile birinci metatars şaft ekseni arasındaki açıdır (Normal: 0° - 4°; >4° plantar açılanma düz taban göstergesidir).
2. **Kalkaneal Pitch (Kalkaneus Eğim Açısı):** Kalkaneus kemiğinin alt sınırı ile ayak tabanının bastığı zemin çizgisi arasındaki açıdır (Normal: 17° - 32°; <17° düz taban göstergesidir).

Literatürdeki mevcut bilgisayar destekli çalışmalar (örneğin Noh vd., *Scientific Reports* 2024; Khaleghizadeh vd., 2024), bu süreci YOLO tabanlı kemik segmentasyonu (talus, kalkaneus, metatars) ve ardından 10 adet kontur nirengi noktasının tespitiyle geometrik açı hesaplama basamaklarına ayırarak otomatikleştirmeye çalışmıştır.

### Geleneksel Geometrik Açı Tespit Yöntemlerinin Tıkanma Nedenleri
1. **Hata Birikimi (Error Compounding):** Segmentasyon maskelerinden çıkarılan nirengi noktalarındaki 1–2 piksellik milimetrik bir kayma, Meary açısında 5–8 derecelik yapay sapmalara yol açmakta ve hastanın yanlış sınıflandırılmasına neden olmaktadır.
2. **Zemin Referansı Hassasiyeti:** Kalkaneal pitch açısı, hastanın bastığı zemin basamağının açısına bağımlıdır. Çekim sırasındaki kaset eğimleri, basamak metalinin parlaması veya bacağın eğik durması hesaba katılamadığında açı tamamen hatalı çıkmaktadır.
3. **Etiketleme Darboğazı:** Kemik segmentasyonu ve 10 nirengi noktası için uzman ortopedistlerin binlerce röntgeni poligonlarla tek tek işaretlemesi gerekmektedir. Pratikte bu etiketler yalnızca 30–150 görselle sınırlı kalmakta, bu da modellerin genelleme yeteneğini yok etmektedir.

### Çalışmanın Amacı ve Katkıları
Bu çalışmada, açı hesaplama ve nirengi tespiti darboğazını aşmak amacıyla doğrudan **radyografi tabanlı uçtan uca derin öğrenme sınıflandırması** yaklaşımı benimsenmiştir. Model, kemiklerin birbirine göre karmaşık konfigürasyonunu ve kavis çöküşünü doğrudan küresel öznitelik haritalarından öğrenmektedir.

Çalışmanın temel bilimsel katkıları:
1. **Geniş Klinik Veri Seti:** 1.529 adet gerçek basarak çekilen klinik yan ayak radyografisi (908 düz taban, 621 normal) standardize edilmiştir.
2. **Otomatik Medikal Ön İşleme ve Gürültü Temizleme:** 16-bit dinamik pencereleme, CLAHE kontrast artırımı, kaval kemiği ekseniyle kanonik sağ yönelim eşitlemesi ve basamak/harf gürültülerini kesip atan ROI kırpma algoritması geliştirilmiştir.
3. **Standart Mimarilerin Kapsamlı Kıyaslaması:** ResNet-50, EfficientNet-B2 ve ConvNeXt-Tiny modelleri 230 hastadan oluşan bağımsız test kümesinde kapsamlı medikal metriklerle (Accuracy, Sens, Spec, Prec, F1, ROC-AUC) kıyaslanmıştır.
4. **Grad-CAM ile Anatomik Doğrulama:** Açıklanabilir Yapay Zeka (XAI) haritalarıyla modelin radyografik harf işaretlerine ("L"/"R") veya metal basamağa değil, tam olarak tıbbi karar noktalarına (medial ark, talonaviküler eklem, kalkaneus) odaklandığı kanıtlanmıştır.

---

## 2. Materyal ve Metot

### 2.1 Veri Seti
Çalışmada klinik PACS arşivinden temin edilen 1.529 adet basarak çekilmiş yan ayak röntgeni kullanılmıştır:
- **Düz Taban (Pes Planus):** 908 adet
- **Normal Ayak:** 621 adet
Görüntüler orijinalinde 16-bit gri tonlamalı PNG formatında olup, çözünürlükleri 2428 × 3003 ile 3072 × 3072 piksel arasında değişmektedir.

### 2.2 Medikal Görüntü Ön İşleme (Preprocessing)
Görüntülerdeki çekim parametresi farklılıklarını ve artefaktları gidermek için 6 aşamalı deterministik boru hattı uygulanmıştır:
1. **Dinamik Aralık Pencereleme:** %1 ve %99 persentil sınırları kullanılarak 16-bit radyografiler 8-bit $[0, 255]$ aralığına normalize edilmiştir.
2. **CLAHE Kontrastı:** Kemik trabeküler yapısını ve eklem boşluklarını belirginleştirmek için 2.0 klip limiti ve $8 \times 8$ ızgara boyutuyla CLAHE uygulanmıştır.
3. **Kaval Kemiği ile Yön Standardizasyonu:** Kaval kemiği (tibia) dikey olarak ayak bileğine girmektedir. Görüntünün üst yarısındaki kolon yoğunluk profili incelenerek kaval kemiğinin X koordinatı ($X_{\text{tibia}}$) tespit edilmiştir. $X_{\text{tibia}} > W/2$ ise ayağın sola baktığı tespit edilmiş ve görsel yatay olarak çevrilerek tüm ayakların parmakları sağa bakacak şekilde kanonik yönelim sağlanmıştır (768 sol, 761 sağ ayak başarıyla eşitlenmiştir).
4. **Basamak Tespiti ve Ayak ROI Kırpma:** Yatay Sobel gradyan filtresiyle hastanın bastığı platform yüzeyi tespit edilmiş; platformun altındaki metal iskelet/ayaklar, üstteki bacak kemikleri ve köşelerdeki "L"/"R" harf etiketleri kesilerek sadece ayak bölgesi izole edilmiştir.
5. **Yeniden Boyutlandırma:** Kırpılan ayak bölgeleri $512 \times 512$ piksel standart boyuta getirilmiştir.

### 2.3 Katmanlı (Stratified) Veri Bölümleme
Veri sızıntısını (data leakage) engellemek için veri seti katmanlı olarak 3 kümeye bölünmüştür:
- **Eğitim Kümesi (%69.98, n=1.070):** 635 Düz Taban, 435 Normal
- **Doğrulama Kümesi (%14.98, n=229):** 136 Düz Taban, 93 Normal
- **Test Kümesi (%15.04, n=230):** 137 Düz Taban, 93 Normal

### 2.4 Derin Öğrenme Mimarileri
Üç farklı tasarım felsefesine sahip konvolüsyonel mimari seçilmiştir:
1. **ResNet-50:** Medikal görüntülemenin klasik altın standardı (50 katmanlı artık bloklar).
2. **EfficientNet-B2:** Derinlik, genişlik ve çözünürlüğü bileşik katsayıyla optimize eden, SE dikkat mekanizmalı verimli mimari.
3. **ConvNeXt-Tiny:** Vision Transformer ilkelerini (7×7 derinlik konvolüsyonları, ters çevrilmiş darboğazlar) standart konvolüsyona uyarlayan modern mimari.

Tüm modeller ImageNet ağırlıklarıyla başlatılmış ve ikili sınıflandırma (Pes Planus vs. Normal) için Dropout ($p=0.2$) içeren özel sınıflandırma başlığı eklenmiştir.

### 2.5 Eğitim Protokolü
- **Kayıp Fonksiyonu:** Veri setindeki hafif dengesizliği (908 vs 621) dengelemek için sınıf ağırlıklı Çapraz Entropi (Normal ağırlığı: 1.23, Düz Taban ağırlığı: 0.84).
- **Optimizasyon:** AdamW ($\text{LR} = 10^{-4}$, weight decay $= 10^{-2}$).
- **Öğrenme Oranı Zamanlayıcısı:** Cosine Annealing (20 epoch, minimum $\text{LR} = 10^{-6}$).
- **Donanım Hızlandırma:** NVIDIA GeForce RTX 3050 Ti Laptop GPU üzerinde PyTorch Automatic Mixed Precision (AMP / FP16) ve Batch Size = 8.
- **Veri Artırma:** Rastgele rotasyon ($\pm 7^\circ$), öteleme ($\pm 4\%$), ölçekleme ($0.96-1.04\times$), parlaklık/kontrast oynaması ($\pm 15\%$). Kanonik yönelimi korumak için yatay çevirme yapılmamıştır.

---

## 3. Deneysel Bulgular ve Sonuçlar

### 3.1 Test Seti Teşhis Başarımı
Modellerin bağımsız 230 klinik vaka üzerindeki sonuçları Tablo 1'de sunulmuştur:

**Tablo 1: Bağımsız test setinde (n=230) standart derin öğrenme modellerinin teşhis performansı.**

| Model Mimarisi | Doğruluk (Accuracy) | Hassasiyet (Sens/Recall) | Özgüllük (Specificity) | Kesinlik (Precision/PPV) | NPV | F1-Skoru | ROC-AUC | Test Kaybı (Loss) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B2** | **%86.09** | %91.24 | **%78.49** | **%86.21** | %85.88 | **0.8865** | **0.9222** | **0.5091** |
| **ConvNeXt-Tiny** | %84.35 | %94.16 | %69.89 | %82.17 | %89.04 | 0.8776 | 0.9152 | 0.8875 |
| **ResNet-50** | %84.35 | **%94.89** | %68.82 | %81.76 | **%90.14** | 0.8784 | 0.9134 | 0.6363 |

- **EfficientNet-B2**, en yüksek doğruluğu (%86.09), en yüksek ROC-AUC skorunu (**0.9222**), en dengeli özgüllüğü (%78.49) ve en düşük test kaybını (0.5091) elde ederek genel sıralamada lider olmuştur.
- **ResNet-50**, **%94.89 hassasiyetle** test kümesindeki 137 düz taban hastasından **130 tanesini doğru yakalamış**, yalnızca 7 tanesini kaçırmıştır (NPV: %90.14). Tarama amaçlı klinik uygulamalarda düz tabanı atlamama açısından çok kritik bir başarıdır.
- **ConvNeXt-Tiny**, %94.16 hassasiyet ve 0.9152 AUC ile ResNet-50'ye çok yakın bir performans sergilemiştir.

### 3.2 ROC Eğrisi Analizi
Şekil 1'de modellerin ortak ROC eğrileri gösterilmektedir. Her üç model de $0.91$ üzerinde AUC değeri elde etmiştir. Özellikle düşük yanlış pozitiflik oranlarında ($FPR < 0.20$), EfficientNet-B2 eğrisi diğer modellerin üzerinde seyrederek üstün ayırt ediciliğini kanıtlamıştır.

*(Şekil 1: `experiments/benchmark_roc_comparison.png`)*

### 3.3 Grad-CAM Açıklanabilirlik İncelemesi
Grad-CAM ile çıkarılan aktivasyon ısı haritaları incelendiğinde:
1. **Medial Kavis Çökmesi:** Düz taban olarak tahmin edilen vakalarda modellerin aktivasyon odakları doğrudan naviküler kemik, kuneiform kemikler ve talonaviküler eklem aralığında yoğunlaşmıştır.
2. **Kalkaneus Açısı ve Ayak Tabanı:** Normal ayak vakalarında modeller kalkaneus eğimi (calcaneal pitch) ve zemin ile medial kavis arasındaki açıklığa odaklanmıştır.
3. **Sıfır Gürültü:** Arka plandaki boşluğa, basamağın metal çizgilerine veya harf bölgelerine hiçbir aktivasyon düşmemiştir. Bu durum modelin kestirme öğrenme yapmadığını, gerçek ortopedik belirteçleri öğrendiğini kanıtlamaktadır.

*(Şekil 2: `experiments/run_efficientnet_b2_512px/gradcam_test_samples.png`)*

---

## 4. Tartışma ve Klinik Değerlendirme

Önceki çalışmalarda (Noh vd., 2024), 3 kemiğin segmente edilip 10 noktanın tespit edilmesi ve formüllerle açı hesaplanması hedeflenmişti. Ancak bu geometrik yaklaşım, hastanelerdeki farklı basamak tipleri ve nirengi etiketleme eksikliği nedeniyle pratikte uygulanamamıştır.

Bu çalışma; **doğrudan sınıflandırma yaklaşımıyla**, hiçbir nirengi noktası etiketine ihtiyaç duyulmadan **0.9222 ROC-AUC** ve **%94.89 hassasiyet** seviyesine ulaşılabileceğini kanıtlamıştır. Ayrıca görsel başına çıkarım (inference) süresinin **~30-35 milisaniye** olması, sistemin hastane PACS ve radyoloji iş istasyonlarına entegre edilebilirliğini göstermektedir.

**Kısıtlar:** Çalışma tek merkezli klinik grafilerden oluşmaktadır. Çok merkezli testlerle modelin farklı röntgen cihazlarındaki dayanıklılığı pekiştirilmelidir.

---

## 5. Sonuç ve Gelecek Çalışmalar

Bu çalışmada, basarak çekilen yan ayak radyografilerinden düz taban tespiti için uçtan uca derin öğrenme sistemi ve medikal ön işleme mimarisi geliştirilmiştir. Elde edilen %86.09 doğruluk, 0.9222 AUC ve %94.89 hassasiyet; yöntemin geleneksel geometrik yöntemlere kıyasla güçlü bir alternatif olduğunu göstermiştir.

**Gelecek Çalışma (FootArchNet):**  
Bu benchmark sonuçları referans alınarak, çalışmanın devamında ayak kemiklerinin boylamsal kavis morfolojisine ve eklem aralıklarına odaklanan anatomik dikkat (*anatomy-guided attention*) mekanizmalı özgün bir derin öğrenme mimarisinin (**FootArchNet**) tasarlanması planlanmaktadır.

---

## Kaynakça
1. Noh, W. J., Lee, M. S., & Lee, B. D. (2024). Deep learning-based automated angle measurement for flatfoot diagnosis in weight-bearing lateral radiographs. *Scientific Reports*, 14(1), 18411.
2. Khaleghizadeh, R., Motamed, S., & Askari, E. (2025). Flatfoot disorder recognition based on the YOLO-ChA algorithm. *Biomedical Signal Processing and Control*, 97, 106560.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
4. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*, 6105–6114.
5. Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR*, 11976–11986.
6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*, 618–626.
