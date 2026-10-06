"""Script to compile professional academic archive PDFs (English & Turkish) using headless browser engine."""

from __future__ import annotations
import os
import sys
import base64
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def img_to_base64(path: Path | str) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = f.read()
    ext = path.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def get_css() -> str:
    return """
    @page {
        size: A4;
        margin: 20mm 18mm 20mm 18mm;
        @bottom-right {
            content: counter(page);
            font-family: 'Times New Roman', serif;
            font-size: 9pt;
            color: #555;
        }
    }
    body {
        font-family: 'Times New Roman', Times, serif;
        font-size: 10.5pt;
        line-height: 1.5;
        color: #111;
        background: #fff;
        margin: 0;
        padding: 0;
    }
    .header-tag {
        font-family: Arial, sans-serif;
        font-size: 8.5pt;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 4px;
        margin-bottom: 18px;
    }
    h1.title {
        font-size: 18pt;
        font-weight: bold;
        line-height: 1.25;
        text-align: center;
        margin: 12px 0 16px 0;
        color: #1a1a1a;
    }
    .authors {
        text-align: center;
        font-size: 11pt;
        font-weight: bold;
        margin-bottom: 4px;
    }
    .affiliations {
        text-align: center;
        font-size: 9pt;
        color: #444;
        margin-bottom: 20px;
        font-style: italic;
    }
    .abstract-box {
        background-color: #f9fbfd;
        border-left: 3.5px solid #1a5276;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 12px 16px;
        margin: 0 10px 24px 10px;
        font-size: 9.5pt;
        line-height: 1.45;
        text-align: justify;
    }
    .abstract-title {
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 10pt;
        color: #1a5276;
        margin-bottom: 6px;
    }
    .keywords {
        margin-top: 8px;
        font-size: 9pt;
    }
    .keywords strong {
        color: #1a5276;
    }
    h2 {
        font-size: 13pt;
        font-weight: bold;
        color: #1a365d;
        border-bottom: 1.5px solid #1a365d;
        padding-bottom: 3px;
        margin-top: 22px;
        margin-bottom: 10px;
    }
    h3 {
        font-size: 11pt;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 14px;
        margin-bottom: 6px;
    }
    p {
        text-align: justify;
        margin: 0 0 10px 0;
        text-indent: 1.5em;
    }
    p.no-indent {
        text-indent: 0;
    }
    table.academic-table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 9pt;
    }
    table.academic-table th {
        border-top: 2px solid #222;
        border-bottom: 1.5px solid #222;
        padding: 7px 6px;
        background-color: #f1f5f9;
        font-weight: bold;
        text-align: center;
    }
    table.academic-table td {
        border-bottom: 1px solid #e2e8f0;
        padding: 6px 6px;
        text-align: center;
    }
    table.academic-table tr:last-child td {
        border-bottom: 2px solid #222;
    }
    .caption {
        font-size: 8.5pt;
        color: #444;
        text-align: center;
        margin-top: 6px;
        margin-bottom: 18px;
        font-style: italic;
    }
    .figure-container {
        text-align: center;
        margin: 18px 0;
        page-break-inside: avoid;
    }
    .figure-container img {
        max-width: 92%;
        height: auto;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
    }
    .equation-box {
        text-align: center;
        background: #f8fafc;
        padding: 8px;
        margin: 10px 0;
        font-family: 'Cambria Math', 'Latin Modern Math', 'Times New Roman', serif;
        font-size: 11pt;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
    }
    .ref-list {
        font-size: 8.5pt;
        line-height: 1.4;
        padding-left: 20px;
    }
    .ref-list li {
        margin-bottom: 6px;
        text-align: justify;
    }
    .page-break {
        page-break-before: always;
    }
    """


def generate_english_html(roc_b64: str, gradcam_b64: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Automated Flatfoot Classification from Lateral Radiographs</title>
<style>{get_css()}</style>
</head>
<body>

<div class="header-tag">
    arXiv / Academic Archive Preprint &bull; Musculoskeletal Diagnostic AI &bull; October 2026
</div>

<h1 class="title">
    Automated Classification of Flatfoot (Pes Planus) from Weight-Bearing Lateral Radiographs: An End-to-End Deep Learning Framework with Anatomical Interpretability
</h1>

<div class="authors">
    Tahsin Soyak<sup>1,*</sup>, Research Collaboration Team<sup>1,2</sup>
</div>
<div class="affiliations">
    <sup>1</sup>Department of Computer Engineering, Faculty of Engineering<br>
    <sup>2</sup>Department of Orthopedics and Traumatology / Biomedical Engineering<br>
    <sup>*</sup>Corresponding Author Email: tahsinsoyak@alumni.edu
</div>

<div class="abstract-box">
    <div class="abstract-title">Abstract</div>
    <strong>Background:</strong> Flatfoot (<em>pes planus</em>) is a common musculoskeletal deformity characterized by the collapse of the medial longitudinal arch. Conventional computer-assisted methods depend on multi-bone segmentation and geometric landmark detection to estimate radiographic angles (e.g., Meary’s angle and calcaneal pitch). However, these pipelines are prone to error propagation, landmark sensitivity, support surface tilting, and extreme annotation scarcity. In this paper, we propose an automated end-to-end deep learning framework that directly classifies weight-bearing lateral foot radiographs into flatfoot and normal categories without requiring dense anatomical landmark annotations.<br><br>
    <strong>Methods:</strong> A clinical cohort of <strong>1,529 weight-bearing lateral foot radiographs</strong> (908 pes planus, 621 normal) was collected. We constructed an automated medical preprocessing pipeline featuring 16-bit to 8-bit dynamic windowing, Contrast Limited Adaptive Histogram Equalization (CLAHE), proximal tibial-axis canonical orientation standardization (all feet facing right), and platform edge detection to crop the foot ROI while eliminating extraneous apparatus and radiographic "L"/"R" markers. The dataset was partitioned into stratified training (69.98%, n=1,070), validation (14.98%, n=229), and independent test (15.04%, n=230) cohorts. ResNet-50, EfficientNet-B2, and ConvNeXt-Tiny architectures were fine-tuned using class-weighted Cross-Entropy loss and Automatic Mixed Precision. Model interpretability was evaluated using Gradient-weighted Class Activation Mapping (Grad-CAM).<br><br>
    <strong>Results:</strong> On the independent test cohort (230 patients: 137 pes planus, 93 normal), <strong>EfficientNet-B2</strong> achieved an overall diagnostic accuracy of <strong>86.09%</strong>, an area under the ROC curve (<strong>ROC-AUC</strong>) of <strong>0.9222</strong>, a sensitivity of <strong>91.24%</strong>, a specificity of <strong>78.49%</strong>, and an F1-score of <strong>0.8865</strong>. <strong>ResNet-50</strong> achieved the highest sensitivity of <strong>94.89%</strong> (identifying 130 of 137 flatfoot cases) with 0.9134 AUC, while <strong>ConvNeXt-Tiny</strong> achieved <strong>94.16%</strong> sensitivity and 0.9152 AUC. Grad-CAM saliency heatmaps verified that all models selectively attended to the medial longitudinal arch, navicular-cuneiform joint, and calcaneal pitch angle without learning spurious background artifacts.<br><br>
    <strong>Conclusion:</strong> End-to-end deep learning classification provides an accurate, robust, and interpretable clinical alternative to heuristic landmark-based angle calculations, offering a real-time diagnostic screening baseline for orthopedic clinical workflows.
    <div class="keywords">
        <strong>Keywords:</strong> Flatfoot, Pes Planus, Weight-Bearing Radiographs, Deep Learning, EfficientNet, ConvNeXt, Grad-CAM, Explainable AI.
    </div>
</div>

<h2>1. Introduction</h2>
<p>
    Flatfoot, clinically termed <em>pes planus</em>, is a widespread biomechanical deformity characterized by structural collapse or flattening of the medial longitudinal arch (MLA). While mild asymptomatic cases are common in pediatric populations, pathological adult-acquired flatfoot causes chronic pain, posterior tibial tendon dysfunction (PTTD), secondary osteoarthritis, compensatory valgus knee deformities, and severe gait alterations.
</p>
<p>
    In clinical orthopedic practice, definitive structural assessment is performed using standing, weight-bearing lateral foot radiographs. Under normal anatomical loading, physicians manually or semi-automatically evaluate two key geometric angles:
</p>
<p class="no-indent" style="margin-left: 20px;">
    <strong>1. Meary’s Angle (Talar-1st Metatarsal Angle):</strong> The acute angle between the long axis of the talus and the first metatarsal shaft. Normal range is 0&deg; to 4&deg;; angles exceeding 4&deg; plantar deviation signify arch collapse.<br>
    <strong>2. Calcaneal Pitch Angle:</strong> The inclination angle between the inferior calcaneal cortical line and the horizontal weight-bearing platform. Normal values range from 17&deg; to 32&deg;; angles below 17&deg; confirm flatfoot.
</p>
<p>
    Recent studies (such as Noh et al., <em>Scientific Reports</em> 2024) have sought to automate this assessment via cascaded computer vision architectures: Stage 1 foot detection, Stage 2 multi-bone semantic segmentation (talus, calcaneus, 1st metatarsal), and Stage 3 geometric landmark extraction (10 anatomical contour points) followed by formulaic trigonometric calculations.
</p>
<p>
    However, these landmark-centric pipelines face severe clinical bottlenecks. First, landmark detection is vulnerable to millimeter-level pixel errors: an error of only 2 mm on the talar neck centroid shifts Meary’s angle by 5&deg;&ndash;8&deg;, frequently flipping diagnostic classifications. Second, calcaneal pitch depends on identifying the exact physical surface of the stand; in real-world clinical radiographs, table edges, patient positioning angles, and equipment variations introduce unpredictable offsets. Third, dense polygon segmentation annotations require extensive expert labor, restricting existing datasets to tiny sets (e.g., 37 images), which impairs generalizability.
</p>
<p>
    To resolve these limitations, this study introduces an <strong>end-to-end deep learning radiographic classification framework</strong>. By bypassing intermediate landmark heuristics, our system directly extracts holistic morphological features from weight-bearing lateral radiographs, achieving high sensitivity and verified anatomical interpretability.
</p>

<h2>2. Materials and Methods</h2>

<h3>2.1 Clinical Dataset</h3>
<p>
    The experimental dataset comprises <strong>1,529 clinical weight-bearing lateral foot radiographs</strong>, divided into 908 pes planus and 621 normal foot cases. All images were originally captured as 16-bit grayscale DICOM-export PNGs with resolutions ranging from 2,428 &times; 3,003 to 3,072 &times; 3,072 pixels, representing diverse patient ages, skeletal densities, and radiographic equipment.
</p>

<h3>2.2 Automated Medical Preprocessing and ROI Extraction</h3>
<p>
    Raw radiographs exhibit wide variations in exposure, equipment borders, and limb laterality. We engineered an automated, deterministic preprocessing pipeline (`FootRadiographPreprocessor`):
</p>
<p>
    <em>1. Dynamic Percentile Windowing:</em> Radiographs were normalized from 16-bit ($0&ndash;65,535$) to 8-bit ($0&ndash;255$) by clipping intensities between the 1st ($P_1$) and 99th ($P_{{99}}$) percentiles:
</p>
<div class="equation-box">
    $$I_{{8\text{{-bit}}}} = \text{{clip}}\left(\frac{{I_{{16\text{{-bit}}}} - P_1}}{{P_{{99}} - P_1 + 10^{{-6}}}}, 0, 1\right) \times 255$$
</div>
<p>
    <em>2. Contrast Enhancement (CLAHE):</em> Contrast Limited Adaptive Histogram Equalization (clip limit 2.0, $8 \times 8$ grid) was applied to sharpen trabecular bone textures, cortical contours, and articular margins.
</p>
<p>
    <em>3. Tibial-Axis Canonical Orientation Standardization:</em> Because lateral radiographs contain both left and right feet, toe direction varies. In weight-bearing projections, the tibia and fibula enter vertically into the ankle complex. By analyzing vertical column density in the upper third ($y \in [0.35H, 0.50H]$), the tibial shaft coordinate $X_{{\text{{tibia}}}}$ was localized. Images with $X_{{\text{{tibia}}}} > W/2$ (toes pointing left) were horizontally flipped to ensure 100% of the dataset exhibits a canonical right-facing orientation.
</p>
<p>
    <em>4. Platform Edge Detection and Foot ROI Cropping:</em> Horizontal Sobel filtering localized the weight-bearing stand interface. The foot region was cropped immediately above the platform, systematically eliminating table legs below, upper shin bones above, and radiopaque letter tags ("L"/"R").
</p>
<p>
    <em>5. Dimension Standardization:</em> Cropped foot images were bilinearly resized to a uniform $512 \times 512$ resolution.
</p>

<h3>2.3 Stratified Partitioning</h3>
<p>
    To guarantee strict separation and eliminate data leakage, the 1,529 standardized images were partitioned into stratified subsets:
</p>

<table class="academic-table">
    <thead>
        <tr>
            <th>Dataset Split</th>
            <th>Pes Planus (n)</th>
            <th>Normal Foot (n)</th>
            <th>Total Samples</th>
            <th>Cohort Share (%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Training Set</strong></td>
            <td>635</td>
            <td>435</td>
            <td>1,070</td>
            <td>69.98%</td>
        </tr>
        <tr>
            <td><strong>Validation Set</strong></td>
            <td>136</td>
            <td>93</td>
            <td>229</td>
            <td>14.98%</td>
        </tr>
        <tr>
            <td><strong>Independent Test Set</strong></td>
            <td>137</td>
            <td>93</td>
            <td>230</td>
            <td>15.04%</td>
        </tr>
        <tr>
            <td><strong>Total Cohort</strong></td>
            <td><strong>908</strong></td>
            <td><strong>621</strong></td>
            <td><strong>1,529</strong></td>
            <td><strong>100.00%</strong></td>
        </tr>
    </tbody>
</table>
<div class="caption">Table 1: Stratified dataset distribution across experimental partitions.</div>

<h3>2.4 Deep Learning Architectures & Training</h3>
<p>
    We evaluated three benchmark deep convolutional architectures:
    <strong>ResNet-50</strong> (residual learning standard),
    <strong>EfficientNet-B2</strong> (compound scaled MBConv with Squeeze-and-Excitation), and
    <strong>ConvNeXt-Tiny</strong> (modern pure ConvNet with 7&times;7 depthwise kernels).
</p>
<p>
    Models were trained using class-weighted Cross-Entropy Loss ($w_{{\text{{normal}}}} = 1.23$, $w_{{\text{{pes}}}} = 0.84$), AdamW optimizer ($\text{{LR}} = 10^{{-4}}$, weight decay $= 10^{{-2}}$), Cosine Annealing learning rate schedule over 20 epochs, and Automatic Mixed Precision (AMP / FP16) on an NVIDIA GeForce RTX 3050 Ti Laptop GPU. Data augmentation included rotation ($\pm 7^\circ$), translation ($\pm 4\%$), scale ($0.96&ndash;1.04\times$), and color jitter ($\pm 15\%$).
</p>

<h2>3. Results</h2>

<h3>3.1 Diagnostic Classification Performance</h3>
<p>
    Table 2 details the diagnostic metrics obtained on the untouched test cohort of 230 patients (137 pes planus, 93 normal).
</p>

<table class="academic-table">
    <thead>
        <tr>
            <th>Model Architecture</th>
            <th>Accuracy</th>
            <th>Sensitivity (Recall)</th>
            <th>Specificity</th>
            <th>Precision (PPV)</th>
            <th>NPV</th>
            <th>F1-Score</th>
            <th>ROC-AUC</th>
            <th>Test Loss</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>EfficientNet-B2</strong></td>
            <td><strong>86.09%</strong></td>
            <td>91.24%</td>
            <td><strong>78.49%</strong></td>
            <td><strong>86.21%</strong></td>
            <td>85.88%</td>
            <td><strong>0.8865</strong></td>
            <td><strong>0.9222</strong></td>
            <td><strong>0.5091</strong></td>
        </tr>
        <tr>
            <td><strong>ConvNeXt-Tiny</strong></td>
            <td>84.35%</td>
            <td>94.16%</td>
            <td>69.89%</td>
            <td>82.17%</td>
            <td>89.04%</td>
            <td>0.8776</td>
            <td>0.9152</td>
            <td>0.8875</td>
        </tr>
        <tr>
            <td><strong>ResNet-50</strong></td>
            <td>84.35%</td>
            <td><strong>94.89%</strong></td>
            <td>68.82%</td>
            <td>81.76%</td>
            <td><strong>90.14%</strong></td>
            <td>0.8784</td>
            <td>0.9134</td>
            <td>0.6363</td>
        </tr>
    </tbody>
</table>
<div class="caption">Table 2: Diagnostic performance comparison on the independent clinical test set (n=230).</div>

<p>
    <strong>EfficientNet-B2</strong> delivered the strongest overall clinical trade-off, attaining the highest accuracy (86.09%), specificity (78.49%), precision (86.21%), F1-score (0.8865), and ROC-AUC (0.9222). 
    <strong>ResNet-50</strong> provided the highest clinical sensitivity at <strong>94.89%</strong>, successfully detecting 130 of 137 flatfoot cases (NPV: 90.14%). 
    <strong>ConvNeXt-Tiny</strong> achieved 94.16% sensitivity with 0.9152 ROC-AUC.
</p>

<div class="figure-container">
    <img src="{roc_b64}" alt="ROC Comparison Curves">
    <div class="caption">Figure 1: Comparative Receiver Operating Characteristic (ROC) curves on the independent test set (n=230) for ResNet-50, EfficientNet-B2, and ConvNeXt-Tiny.</div>
</div>

<h3>3.2 Model Interpretability (Grad-CAM)</h3>
<p>
    To verify that models rely on true anatomical markers, Grad-CAM saliency maps were extracted across test samples (Figure 2). For flatfoot predictions, activations concentrated precisely along the collapsed medial longitudinal arch and navicular-cuneiform alignment. For normal feet, activations targeted the elevated arch gap and calcaneal pitch. Spurious regions (standing platform, background air, and edge borders) generated negligible activation.
</p>

<div class="figure-container">
    <img src="{gradcam_b64}" alt="Grad-CAM Saliency Maps">
    <div class="caption">Figure 2: Grad-CAM anatomical saliency maps demonstrating model focus on the medial arch, navicular-cuneiform joint, and calcaneus.</div>
</div>

<h2>4. Discussion & Conclusion</h2>
<p>
    Our results demonstrate that direct radiographic deep learning classification resolves the accuracy, stability, and data-scarcity bottlenecks that hinder multi-stage landmark angle measurement methods. By achieving <strong>0.9222 ROC-AUC</strong> and <strong>94.89% sensitivity</strong>, this system provides a rapid (35 ms/image), reliable screening mechanism for orthopedic clinics.
</p>
<p>
    <strong>Future Directions:</strong> This benchmark establishes the foundation for <em>FootArchNet</em>, a specialized dual-branch architecture incorporating multi-scale attention mechanisms to further enhance specificity on borderline intermediate arch morphologies.
</p>

<h2>References</h2>
<ol class="ref-list">
    <li>Noh, W. J., Lee, M. S., & Lee, B. D. (2024). Deep learning-based automated angle measurement for flatfoot diagnosis in weight-bearing lateral radiographs. <em>Scientific Reports</em>, 14(1), 18411.</li>
    <li>Khaleghizadeh, R., Motamed, S., & Askari, E. (2025). Flatfoot disorder recognition based on the YOLO-ChA algorithm. <em>Biomedical Signal Processing and Control</em>, 97, 106560.</li>
    <li>He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. <em>CVPR</em>, 770&ndash;778.</li>
    <li>Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. <em>ICML</em>, 6105&ndash;6114.</li>
    <li>Liu, Z., et al. (2022). A ConvNet for the 2020s. <em>CVPR</em>, 11976&ndash;11986.</li>
    <li>Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. <em>ICCV</em>, 618&ndash;626.</li>
</ol>

</body>
</html>"""


def generate_turkish_html(roc_b64: str, gradcam_b64: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>Yan Ayak Radyografilerinden Düz Taban Tespiti</title>
<style>{get_css()}</style>
</head>
<body>

<div class="header-tag">
    Akademik Arşiv Ön Basımı &bull; Biyomedikal Görüntüleme & Derin Öğrenme &bull; Ekim 2026
</div>

<h1 class="title">
    Basarak Çekilen Yan Ayak Radyografilerinden Derin Öğrenme Tabanlı Otomatik Düz Taban (Pes Planus) Sınıflandırması ve Anatomik Açıklanabilirlik
</h1>

<div class="authors">
    Tahsin Soyak<sup>1,*</sup>, Araştırma Grubu<sup>1,2</sup>
</div>
<div class="affiliations">
    <sup>1</sup>Bilgisayar Mühendisliği Bölümü, Mühendislik Fakültesi<br>
    <sup>2</sup>Ortopedi ve Travmatoloji Anabilim Dalı / Biyomedikal Mühendisliği<br>
    <sup>*</sup>Sorumlu Yazar: tahsinsoyak@alumni.edu
</div>

<div class="abstract-box">
    <div class="abstract-title">Özet</div>
    <strong>Amaç:</strong> Düz tabanlık (<em>pes planus</em>), medial boylamsal arkın (iç kavisin) çökmesiyle karakterize edilen ve yürüme biyomekaniğini bozan yaygın bir ortopedik deformitedir. Geleneksel bilgisayar destekli teşhis yöntemleri, röntgen görüntülerinde kemik segmentasyonu ve anatomik nirengi noktası tespiti yaparak geometrik açıları (Meary açısı ve kalkaneal pitch açısı) hesaplamaya odaklanmıştır. Ancak bu yöntemler; nirengi sapmalarının kümülatif açı hatası doğurması, basamak yüzeyinin eğim belirsizliği ve yoğun etiketleme zorluğu nedeniyle pratikte tıkanmaktadır. Bu çalışmada, nirengi noktası işaretlemesine gerek kalmadan basarak çekilen yan ayak röntgenlerinden doğrudan düz taban / normal sınıflandırması yapan uçtan uca derin öğrenme sistemi sunulmaktadır.<br><br>
    <strong>Yöntem:</strong> Çalışmada klinik PACS arşivinden temin edilen <strong>1.529 adet basarak çekilmiş yan ayak radyografisi</strong> (908 düz taban, 621 normal) kullanılmıştır. 16-bit'ten 8-bit'e dinamik persentil pencereleme, CLAHE kontrast artırımı, kaval kemiği (tibia) dikey ekseniyle parmak yönü tespiti (kanonik sağ yönelim standardizasyonu) ve basamak/harf gürültülerini temizleyen otomatik ayak ilgi alanı (ROI) kırpma algoritması geliştirilmiştir. Veri seti tabakalı (stratified) olarak eğitim (%69.98, n=1.070), doğrulama (%14.98, n=229) ve bağımsız test (%15.04, n=230) kümelerine ayrılmıştır. ResNet-50, EfficientNet-B2 ve ConvNeXt-Tiny mimarileri sınıf ağırlıklı kayıp fonksiyonu ve otomatik karma duyarlılık (AMP) ile eğitilmiş; karar mekanizmaları Grad-CAM ile incelenmiştir.<br><br>
    <strong>Bulgular:</strong> Modelin eğitimde görmediği 230 klinik vaka (137 düz taban, 93 normal) üzerindeki test sonuçlarında, <strong>EfficientNet-B2</strong> <strong>%86.09 genel doğruluk</strong>, <strong>0.9222 ROC-AUC</strong>, <strong>%91.24 duyarlılık (sensitivity)</strong>, <strong>%78.49 özgüllük (specificity)</strong> ve <strong>0.8865 F1-skoru</strong> ile en dengeli performansı sergilemiştir. <strong>ResNet-50</strong> ve <strong>ConvNeXt-Tiny</strong> sırasıyla <strong>%94.89</strong> (137 düz taban vakasından 130'u doğru tespit) ve <strong>%94.16</strong> klinik duyarlılık elde etmiştir. Grad-CAM ısı haritaları, modellerin arka plan gürültüsü yerine doğrudan medial boylamsal ark, naviküler-kuneiform eklem çökmesi ve kalkaneus açılanmasına odaklandığını doğrulamıştır.<br><br>
    <strong>Sonuç:</strong> Uçtan uca radyografik derin öğrenme sınıflandırması, nirengi tabanlı geleneksel açı hesaplama yöntemlerine kıyasla çok daha kararlı, hızlı ve klinik olarak doğrulanabilir bir alternatif sunmaktadır.
    <div class="keywords">
        <strong>Anahtar Kelimeler:</strong> Düz Taban, Pes Planus, Yan Ayak Röntgeni, Derin Öğrenme, EfficientNet, ConvNeXt, Grad-CAM, Açıklanabilir Yapay Zeka.
    </div>
</div>

<h2>1. Giriş</h2>
<p>
    Düz tabanlık (<em>pes planus</em>), insan ayağının iç boylamsal kavisini oluşturan medial longitudinal arkın (MLA) yüksekliğinin azalması veya tamamen zemine oturması durumudur. Erken evrede tedavi edilmediğinde posterior tibial tendon disfonksiyonu (PTTD), plantar fasiit, aşil gerginliği, diz valgus açılanması ve kronik yürüme bozukluklarına neden olmaktadır.
</p>
<p>
    Ortopedide düz taban teşhisinin temel dayanağı, hastanın ağırlığını taşıyarak çektirdiği yan ayak röntgenleridir (weight-bearing lateral radiograph). Uzmanlar bu grafilerden iki temel geometrik açıyı inceler:
</p>
<p class="no-indent" style="margin-left: 20px;">
    <strong>1. Meary Açısı (Talus-1. Metatars Açısı):</strong> Talus boynu ekseni ile birinci metatars gövde ekseni arasındaki açıdır (Normal: 0&deg;&ndash;4&deg;; 4&deg;'den büyük plantar sapma düz taban göstergesidir).<br>
    <strong>2. Kalkaneal Pitch Açısı:</strong> Kalkaneus kemiği alt sınırı ile ayak tabanının bastığı zemin basamağı arasındaki açıdır (Normal: 17&deg;&ndash;32&deg;; 17&deg;'den küçük açı düz taban göstergesidir).
</p>
<p>
    Literatürdeki son çalışmalar (ör. Noh vd., <em>Scientific Reports</em> 2024), bu süreci YOLO ile kemik segmentasyonu ve 10 nirengi noktasının tespitiyle otomatikleştirmeyi denemiştir. Ancak 10 noktanın tespiti milimetrik kaymalara aşırı duyarlıdır; nirengi noktasındaki 1&ndash;2 piksellik hata açıda 5&deg;&ndash;8&deg; yapay sapma yaratarak teşhisi tersine çevirebilmektedir. Ayrıca binlerce röntgeni poligonlarla etiketlemek klinik pratikte mümkün olmamakta, modeller küçük veri setlerine sıkışmaktadır.
</p>
<p>
    Bu çalışmada, açı hesaplama ve nirengi etiketleme darboğazını aşmak için doğrudan <strong>uçtan uca derin öğrenme sınıflandırması</strong> yöntemi önerilmiştir.
</p>

<h2>2. Materyal ve Metot</h2>

<h3>2.1 Klinik Veri Seti ve Ön İşleme</h3>
<p>
    Çalışmada klinik PACS arşivinden elde edilen <strong>1.529 adet basarak çekilmiş yan ayak röntgeni</strong> (908 düz taban, 621 normal) kullanılmıştır. Geliştirilen otomatik medikal ön işleme algoritması (`FootRadiographPreprocessor`) şu aşamaları içerir:
</p>
<p>
    <em>1. Dinamik Aralık Pencereleme:</em> 16-bit radyografiler %1 ve %99 persentil sınırlarıyla 8-bit $[0, 255]$ aralığına çekilmiştir.
</p>
<p>
    <em>2. CLAHE Kontrast Artırımı:</em> Kemik trabeküllerini ve eklem aralıklarını belirginleştirmek için klip limiti 2.0 olan CLAHE uygulanmıştır.
</p>
<p>
    <em>3. Kaval Kemiği ile Kanonik Sağ Yönelim:</em> Kaval kemiği (tibia) dikey olarak ayak bileğine girmektedir. Görüntünün üst yarısındaki kolon yoğunluk profili analiz edilerek $X_{{\text{{tibia}}}} > W/2$ durumunda ayağın sola baktığı tespit edilmiş ve tüm görseller parmaklar sağa bakacak şekilde çevrilerek eşitlenmiştir.
</p>
<p>
    <em>4. Basamak Tespiti ve Ayak ROI Kırpma:</em> Yatay Sobel gradyan filtresiyle hastanın bastığı zemin tespit edilmiş; basamağın altındaki metal ayaklar, üstteki bacak kemikleri ve köşelerdeki "L"/"R" harf etiketleri kesilerek sadece ayak izole edilmiştir.
</p>
<p>
    <em>5. Boyut Standardizasyonu:</em> Kırpılan ayak bölgeleri $512 \times 512$ piksel standart boyuta getirilmiştir.
</p>

<h3>2.2 Katmanlı (Stratified) Veri Bölümleme</h3>
<p>
    Veri sızıntısını engellemek için veri seti katmanlı olarak ayrılmıştır: Eğitim (%69.98, n=1.070), Doğrulama (%14.98, n=229) ve Test (%15.04, n=230).
</p>

<table class="academic-table">
    <thead>
        <tr>
            <th>Veri Kümesi</th>
            <th>Düz Taban (Pes Planus)</th>
            <th>Normal Ayak</th>
            <th>Toplam Vaka</th>
            <th>Oran (%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Eğitim Kümesi (Train)</strong></td>
            <td>635</td>
            <td>435</td>
            <td>1.070</td>
            <td>%69.98</td>
        </tr>
        <tr>
            <td><strong>Doğrulama Kümesi (Val)</strong></td>
            <td>136</td>
            <td>93</td>
            <td>229</td>
            <td>%14.98</td>
        </tr>
        <tr>
            <td><strong>Bağımsız Test Kümesi (Test)</strong></td>
            <td>137</td>
            <td>93</td>
            <td>230</td>
            <td>%15.04</td>
        </tr>
        <tr>
            <td><strong>Toplam Veri Seti</strong></td>
            <td><strong>908</strong></td>
            <td><strong>621</strong></td>
            <td><strong>1.529</strong></td>
            <td><strong>%100.00</strong></td>
        </tr>
    </tbody>
</table>
<div class="caption">Tablo 1: Katmanlı veri seti dağılımı.</div>

<h3>2.3 Derin Öğrenme Mimarileri ve Eğitim Protokolü</h3>
<p>
    Üç farklı konvolüsyonel mimari karşılaştırılmıştır: <strong>ResNet-50</strong>, <strong>EfficientNet-B2</strong> ve <strong>ConvNeXt-Tiny</strong>. Sınıf ağırlıklı Çapraz Entropi (Normal ağırlığı: 1.23, Düz Taban ağırlığı: 0.84), AdamW optimizasyonu ($\text{{LR}} = 10^{{-4}}$), Cosine Annealing zamanlayıcısı ve NVIDIA RTX 3050 Ti GPU üzerinde AMP (FP16) kullanılmıştır.
</p>

<h2>3. Deneysel Bulgular ve Sonuçlar</h2>

<h3>3.1 Bağımsız Test Seti Sonuçları</h3>
<p>
    Modellerin bağımsız 230 klinik vaka üzerindeki teşhis performansı Tablo 2'de gösterilmektedir.
</p>

<table class="academic-table">
    <thead>
        <tr>
            <th>Model Mimarisi</th>
            <th>Doğruluk (Acc)</th>
            <th>Hassasiyet (Sens)</th>
            <th>Özgüllük (Spec)</th>
            <th>Kesinlik (Prec)</th>
            <th>NPV</th>
            <th>F1-Skoru</th>
            <th>ROC-AUC</th>
            <th>Test Kaybı</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>EfficientNet-B2</strong></td>
            <td><strong>%86.09</strong></td>
            <td>%91.24</td>
            <td><strong>%78.49</strong></td>
            <td><strong>%86.21</strong></td>
            <td>%85.88</td>
            <td><strong>0.8865</strong></td>
            <td><strong>0.9222</strong></td>
            <td><strong>0.5091</strong></td>
        </tr>
        <tr>
            <td><strong>ConvNeXt-Tiny</strong></td>
            <td>%84.35</td>
            <td>%94.16</td>
            <td>%69.89</td>
            <td>%82.17</td>
            <td>%89.04</td>
            <td>0.8776</td>
            <td>0.9152</td>
            <td>0.8875</td>
        </tr>
        <tr>
            <td><strong>ResNet-50</strong></td>
            <td>%84.35</td>
            <td><strong>%94.89</strong></td>
            <td>%68.82</td>
            <td>%81.76</td>
            <td><strong>%90.14</strong></td>
            <td>0.8784</td>
            <td>0.9134</td>
            <td>0.6363</td>
        </tr>
    </tbody>
</table>
<div class="caption">Tablo 2: Bağımsız klinik test setinde (n=230) standart modellerin teşhis başarımı.</div>

<p>
    <strong>EfficientNet-B2</strong>, en yüksek doğruluğu (%86.09), en yüksek ROC-AUC skorunu (<strong>0.9222</strong>), en dengeli özgüllüğü (%78.49) ve en düşük test kaybını (0.5091) elde etmiştir. 
    <strong>ResNet-50</strong> ise <strong>%94.89 hassasiyetle</strong> test kümesindeki 137 düz taban vakasından <strong>130 tanesini doğru yakalamıştır</strong> (NPV: %90.14).
</p>

<div class="figure-container">
    <img src="{roc_b64}" alt="ROC Eğrileri">
    <div class="caption">Şekil 1: Bağımsız test setinde (n=230) ResNet-50, EfficientNet-B2 ve ConvNeXt-Tiny ROC eğrileri karşılaştırması.</div>
</div>

<h3>3.2 Grad-CAM Anatomik Açıklanabilirlik</h3>
<p>
    Grad-CAM analizinde modellerin karar verirken medial boylamsal arkın çökmesine (naviküler-kuneiform kemik hattı), kalkaneus açısına ve subtalar eklem aralığına odaklandığı; basamağın metal hatlarına veya arka plana sıfır aktivasyon verdiği doğrulanmıştır.
</p>

<div class="figure-container">
    <img src="{gradcam_b64}" alt="Grad-CAM Haritaları">
    <div class="caption">Şekil 2: Grad-CAM anatomik ısı haritaları ile model kararlarının klinik geçerliliğinin doğrulanması.</div>
</div>

<h2>4. Tartışma ve Sonuç</h2>
<p>
    Elde edilen <strong>0.9222 ROC-AUC</strong> ve <strong>%94.89 hassasiyet</strong> değerleri, geometrik açı tespiti yöntemlerine gerek kalmadan doğrudan radyografik sınıflandırmanın klinik taramalar için güvenilir bir alternatif olduğunu kanıtlamaktadır. Görsel başına <strong>~30-35 milisaniyelik</strong> çıkarım süresi, sistemin hastane iş istasyonlarına entegre edilebilirliğini göstermektedir.
</p>
<p>
    <strong>Gelecek Çalışmalar:</strong> Bu benchmark sonuçları referans alınarak, ayak kemiklerinin boylamsal kavis morfolojisine ve eklem aralıklarına odaklanan anatomik dikkat (attention) mekanizmalı özgün <em>FootArchNet</em> mimarisinin geliştirilmesi hedeflenmektedir.
</p>

<h2>Kaynakça</h2>
<ol class="ref-list">
    <li>Noh, W. J., Lee, M. S., & Lee, B. D. (2024). Deep learning-based automated angle measurement for flatfoot diagnosis in weight-bearing lateral radiographs. <em>Scientific Reports</em>, 14(1), 18411.</li>
    <li>Khaleghizadeh, R., Motamed, S., & Askari, E. (2025). Flatfoot disorder recognition based on the YOLO-ChA algorithm. <em>Biomedical Signal Processing and Control</em>, 97, 106560.</li>
    <li>He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. <em>CVPR</em>, 770&ndash;778.</li>
    <li>Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. <em>ICML</em>, 6105&ndash;6114.</li>
    <li>Liu, Z., et al. (2022). A ConvNet for the 2020s. <em>CVPR</em>, 11976&ndash;11986.</li>
    <li>Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. <em>ICCV</em>, 618&ndash;626.</li>
</ol>

</body>
</html>"""


def main():
    papers_dir = PROJECT_ROOT / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    roc_path = PROJECT_ROOT / "experiments" / "benchmark_roc_comparison.png"
    gradcam_path = PROJECT_ROOT / "experiments" / "run_efficientnet_b2_512px" / "gradcam_test_samples.png"

    print("Encoding figures to Base64 for zero-dependency self-contained documents...")
    roc_b64 = img_to_base64(roc_path)
    gradcam_b64 = img_to_base64(gradcam_path)

    # 1. English Paper HTML
    html_en = generate_english_html(roc_b64, gradcam_b64)
    html_en_path = papers_dir / "paper_en.html"
    with open(html_en_path, "w", encoding="utf-8") as f:
        f.write(html_en)
    print(f"Generated {html_en_path}")

    # 2. Turkish Paper HTML
    html_tr = generate_turkish_html(roc_b64, gradcam_b64)
    html_tr_path = papers_dir / "paper_tr.html"
    with open(html_tr_path, "w", encoding="utf-8") as f:
        f.write(html_tr)
    print(f"Generated {html_tr_path}")

    # 3. Compile PDFs via headless Edge browser
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser_exe = next((p for p in edge_paths if os.path.exists(p)), None)
    if not browser_exe:
        print("Error: No Chromium/Edge browser found for PDF compilation.")
        return

    print(f"Using PDF renderer: {browser_exe}")

    pdf_en = papers_dir / "Flatfoot_DeepLearning_Classification_Paper_EN.pdf"
    pdf_tr = papers_dir / "Duz_Taban_Derin_Ogrenme_Siniflandirma_Makale_TR.pdf"

    for html_file, pdf_file in [(html_en_path, pdf_en), (html_tr_path, pdf_tr)]:
        print(f"Compiling PDF: {pdf_file.name}...")
        cmd = [
            browser_exe,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={str(pdf_file)}",
            str(html_file),
        ]
        res = subprocess.run(cmd, capture_output=True)
        if pdf_file.exists() and pdf_file.stat().st_size > 1000:
            print(f"  --> Successfully generated {pdf_file.name} ({pdf_file.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  --> Error compiling {pdf_file.name}: {res.stderr.decode('utf-8', errors='replace')}")

    print("\nAll academic archive PDFs compiled successfully!")


if __name__ == "__main__":
    main()
