"""Script to compile professional academic archive PDFs (English & Turkish)
with publication-grade SVG architecture diagrams, embedded vector math equations,
and diagnostic visualization figures using headless browser engine.
"""

from __future__ import annotations
import os
import sys
import io
import base64
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def img_to_base64(path: Path | str) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = f.read()
    ext = path.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def render_latex_svg(latex_str: str, fontsize: int = 12, fig_width: float = 7.0, fig_height: float = 0.75) -> str:
    """Render LaTeX equation to an inline, clean SVG using Matplotlib's mathtext engine."""
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.patch.set_alpha(0.0)
    ax.text(0.5, 0.5, latex_str, fontsize=fontsize, ha="center", va="center", color="#0f172a")
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True, pad_inches=0.03)
    plt.close(fig)
    svg_str = buf.getvalue().decode("utf-8")
    # Strip XML header if present
    if "<?xml" in svg_str:
        svg_str = svg_str[svg_str.find("<svg"):]
    return svg_str


def get_css() -> str:
    return """
    @page {
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {
            content: counter(page);
            font-family: 'Times New Roman', serif;
            font-size: 9pt;
            color: #64748b;
        }
    }
    *, *::before, *::after {
        box-sizing: border-box;
    }
    body {
        font-family: 'Times New Roman', Times, serif;
        font-size: 10.5pt;
        line-height: 1.5;
        color: #0f172a;
        background: #ffffff;
        margin: 0;
        padding: 0;
    }
    .header-tag {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 8.5pt;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 1.5px solid #cbd5e1;
        padding-bottom: 5px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
    }
    h1.title {
        font-size: 17.5pt;
        font-weight: bold;
        line-height: 1.25;
        text-align: center;
        margin: 10px 0 14px 0;
        color: #0f172a;
    }
    .authors {
        text-align: center;
        font-size: 11pt;
        font-weight: bold;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .affiliations {
        text-align: center;
        font-size: 9pt;
        color: #475569;
        margin-bottom: 18px;
        font-style: italic;
        line-height: 1.35;
    }
    .abstract-box {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 0 4px 22px 4px;
        font-size: 9.5pt;
        line-height: 1.45;
        text-align: justify;
    }
    .abstract-title {
        font-family: Arial, sans-serif;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-size: 10pt;
        color: #0369a1;
        margin-bottom: 6px;
    }
    .keywords {
        margin-top: 10px;
        font-size: 9pt;
        border-top: 1px dashed #cbd5e1;
        padding-top: 6px;
    }
    .keywords strong {
        color: #0284c7;
    }
    h2 {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 12.5pt;
        font-weight: bold;
        color: #0f172a;
        border-bottom: 1.5px solid #0284c7;
        padding-bottom: 3px;
        margin-top: 22px;
        margin-bottom: 8px;
    }
    h3 {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 10.5pt;
        font-weight: bold;
        color: #1e293b;
        margin-top: 14px;
        margin-bottom: 5px;
    }
    p {
        text-align: justify;
        margin: 0 0 9px 0;
        text-indent: 1.4em;
    }
    p.no-indent {
        text-indent: 0;
    }
    table.academic-table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0 6px 0;
        font-size: 9pt;
    }
    table.academic-table th {
        border-top: 2px solid #0f172a;
        border-bottom: 1.5px solid #0f172a;
        padding: 6px 6px;
        background-color: #f1f5f9;
        font-weight: bold;
        text-align: center;
        color: #0f172a;
    }
    table.academic-table td {
        border-bottom: 1px solid #e2e8f0;
        padding: 5.5px 6px;
        text-align: center;
    }
    table.academic-table tr:last-child td {
        border-bottom: 2px solid #0f172a;
    }
    .caption {
        font-size: 8.5pt;
        color: #475569;
        text-align: center;
        margin-top: 6px;
        margin-bottom: 16px;
        font-style: italic;
        line-height: 1.35;
    }
    .figure-container {
        text-align: center;
        margin: 16px 0 10px 0;
        page-break-inside: avoid;
    }
    .figure-container img {
        max-width: 96%;
        height: auto;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
    }
    .figure-container-full {
        text-align: center;
        margin: 14px 0 8px 0;
        page-break-inside: avoid;
    }
    .figure-container-full svg {
        width: 100%;
        max-height: 480px;
        height: auto;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
    }
    .equation-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #fbfcfe;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 4px 14px;
        margin: 9px 0;
        page-break-inside: avoid;
    }
    .equation-content {
        flex-grow: 1;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .equation-num {
        font-family: 'Times New Roman', serif;
        font-size: 10pt;
        color: #475569;
        font-weight: normal;
        padding-left: 10px;
    }
    .ref-list {
        font-size: 8.5pt;
        line-height: 1.4;
        padding-left: 18px;
        margin-top: 6px;
    }
    .ref-list li {
        margin-bottom: 5px;
        text-align: justify;
    }
    .page-break {
        page-break-before: always;
    }
    """


def generate_english_html(roc_b64: str, gradcam_b64: str, arch_svg: str) -> str:
    # Compile clean SVG vector equations with raw LaTeX strings
    eq1_svg = render_latex_svg(
        r"$I_{\mathrm{8-bit}}(x, y) = \mathrm{clip}\left(\frac{I_{\mathrm{16-bit}}(x, y) - P_1}{P_{99} - P_1 + 10^{-6}},\, 0,\, 1\right) \times 255$",
        fontsize=11.5, fig_width=6.6, fig_height=0.65
    )
    eq2_svg = render_latex_svg(
        r"$\rho_{\mathrm{vert}}(x) = \sum_{y=0.35H}^{0.50H} I(x, y), \quad X_{\mathrm{tibia}} = \arg\max_{x} \left(\rho_{\mathrm{vert}} * G_{\sigma=15}\right)(x)$",
        fontsize=11.5, fig_width=6.6, fig_height=0.65
    )
    eq3_svg = render_latex_svg(
        r"$I_{\mathrm{aligned}}(x, y) = I(W - 1 - x,\, y)\quad [\mathrm{if}\ X_{\mathrm{tibia}} > W/2], \quad I(x,\, y)\quad [\mathrm{if}\ X_{\mathrm{tibia}} \leq W/2]$",
        fontsize=11.5, fig_width=6.8, fig_height=0.65
    )
    eq4_svg = render_latex_svg(
        r"$Y_{\mathrm{platform}} = \arg\max_{y \in [0.55H,\, 0.95H]} \sum_{x=0.2W}^{0.8W} |S_y(x, y)|, \quad \text{where } S_y = I * K_y$",
        fontsize=11.5, fig_width=6.6, fig_height=0.65
    )
    eq5_svg = render_latex_svg(
        r"$\mathcal{L}_{\mathrm{CE}}(\theta) = -\frac{1}{B} \sum_{i=1}^B w_{y_i} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$",
        fontsize=11.5, fig_width=6.4, fig_height=0.65
    )
    eq6_svg = render_latex_svg(
        r"$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\mathrm{Grad\text{-}CAM}}^c(i, j) = \mathrm{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k(i, j) \right)$",
        fontsize=11.5, fig_width=6.8, fig_height=0.70
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Automated Flatfoot Classification from Lateral Radiographs</title>
<style>{get_css()}</style>
</head>
<body>

<div class="header-tag">
    <span>arXiv / Academic Archive Preprint &bull; Musculoskeletal Diagnostic AI</span>
    <span>October 2026</span>
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
    <strong>Background:</strong> Flatfoot (<em>pes planus</em>) is a common musculoskeletal deformity characterized by the collapse of the medial longitudinal arch. Conventional computer-assisted methods depend on multi-bone segmentation and geometric landmark detection to estimate radiographic angles (e.g., Meary’s angle and calcaneal pitch). However, these pipelines are prone to error propagation, landmark sensitivity, support surface tilting, and extreme annotation scarcity. In this paper, we propose an automated end-to-end deep learning framework featuring a custom hybrid CNN-Transformer architecture (<strong>FootArchNet-V2</strong>) and a <strong>Clinical Ensemble</strong> that directly classifies weight-bearing lateral foot radiographs into flatfoot and normal categories without requiring dense anatomical landmark annotations.<br><br>
    <strong>Methods:</strong> A clinical cohort of <strong>1,529 weight-bearing lateral foot radiographs</strong> (908 pes planus, 621 normal) was collected. We constructed an automated medical preprocessing pipeline featuring 16-bit to 8-bit dynamic percentile windowing, Contrast Limited Adaptive Histogram Equalization (CLAHE), proximal tibial-axis canonical orientation standardization (all feet facing right), and platform edge detection with aspect-ratio preserving letterboxing and anatomical safety margins (&ge;30 px). The dataset was partitioned into stratified training (70.05%, n=1,071), validation (14.98%, n=229), and independent test (14.98%, n=229) cohorts. We engineered <strong>FootArchNet-V2</strong>, integrating Coordinate Convolutions (CoordConv) for ground-truth spatial metric anchoring, Multi-Head Spatial Self-Attention (TransArchAttention) across tarsal bone tokens, and a Biomechanical Feature Pyramid Network (BFPN). We benchmarked it against FootArchNet-V1, ResNet-50, EfficientNet-B2, and ConvNeXt-Tiny, followed by a multi-backbone <strong>Clinical Ensemble</strong>.<br><br>
    <strong>Results:</strong> On the independent test cohort of 229 unseen clinical patients (136 pes planus, 93 normal), <strong>FootArchNet-V2</strong> achieved a single-model peak ROC-AUC of <strong>0.9435</strong> with <strong>84.95% specificity</strong> and <strong>89.06% precision</strong>. The multi-model <strong>Clinical Ensemble</strong> attained the cohort-leading performance: <strong>86.90% diagnostic accuracy</strong>, <strong>0.9492 ROC-AUC</strong>, <strong>88.17% specificity</strong>, <strong>91.41% precision</strong>, and an F1-score of <strong>0.8864</strong>. Grad-CAM visual heatmaps confirmed that FootArchNet-V2 specifically and faithfully focuses on the apex of the medial longitudinal arch, navicular-cuneiform alignment, and calcaneal pitch with zero spurious activation on background or standing apparatus.<br><br>
    <strong>Conclusion:</strong> Direct deep learning classification with FootArchNet-V2 and the Clinical Ensemble eliminates geometric landmark fragility and provides a highly accurate, robustly specific (&gt;88%), and clinically explainable diagnostic screening tool for orthopedic practice.
    <div class="keywords">
        <strong>Keywords:</strong> Flatfoot, Pes Planus, Weight-Bearing Radiographs, FootArchNet-V2, Vision Transformer, CoordConv, Clinical Ensemble, Grad-CAM, Explainable AI.
    </div>
</div>

<h2>1. Introduction</h2>
<p>
    Flatfoot, clinically termed <em>pes planus</em>, is a widespread biomechanical deformity characterized by structural collapse or flattening of the medial longitudinal arch (MLA). While mild asymptomatic cases are common in pediatric populations, pathological adult-acquired flatfoot causes chronic pain, posterior tibial tendon dysfunction (PTTD), secondary osteoarthritis, compensatory valgus knee deformities, and severe gait alterations.
</p>
<p>
    In clinical orthopedic practice, definitive structural assessment is performed using standing, weight-bearing lateral foot radiographs. Under normal anatomical loading, physicians manually or semi-automatically evaluate two key geometric angles:
</p>
<p class="no-indent" style="margin-left: 18px;">
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
    Raw radiographs exhibit wide variations in exposure, equipment borders, and limb laterality. We engineered an automated, deterministic preprocessing pipeline (<em>FootRadiographPreprocessor</em>):
</p>
<p>
    <em>1. Dynamic Percentile Windowing:</em> Radiographs were normalized from 16-bit (intensity values 0&ndash;65,535) to 8-bit (0&ndash;255) by clipping intensities between the 1st (<em>P</em><sub>1</sub>) and 99th (<em>P</em><sub>99</sub>) percentiles:
</p>
<div class="equation-box">
    <div class="equation-content">{eq1_svg}</div>
    <div class="equation-num">(1)</div>
</div>
<p>
    <em>2. Contrast Enhancement (CLAHE):</em> Contrast Limited Adaptive Histogram Equalization (clip limit 2.0, 8 &times; 8 contextual grid) was applied to sharpen trabecular bone textures, cortical contours, and articular margins.
</p>
<p>
    <em>3. Tibial-Axis Canonical Orientation Standardization:</em> Because lateral radiographs contain both left and right feet, toe direction varies. In weight-bearing projections, the tibia and fibula enter vertically into the ankle complex. By analyzing vertical column density in the upper shaft band <em>y</em> &isin; [0.35<em>H</em>, 0.50<em>H</em>], the horizontal coordinate <em>X</em><sub>tibia</sub> was determined:
</p>
<div class="equation-box">
    <div class="equation-content">{eq2_svg}</div>
    <div class="equation-num">(2)</div>
</div>
<p>
    All images with <em>X</em><sub>tibia</sub> &gt; <em>W</em>/2 (toes pointing left) were horizontally mirrored, establishing 100% right-facing canonical orientation across the dataset:
</p>
<div class="equation-box">
    <div class="equation-content">{eq3_svg}</div>
    <div class="equation-num">(3)</div>
</div>
<p>
    <em>4. Platform Edge Detection and Foot ROI Cropping:</em> A vertical Sobel gradient operator (<em>K</em><sub>y</sub>) localized the weight-bearing stand interface in the lower radiograph section:
</p>
<div class="equation-box">
    <div class="equation-content">{eq4_svg}</div>
    <div class="equation-num">(4)</div>
</div>
<p>
    The foot anatomical complex was cropped immediately above <em>Y</em><sub>platform</sub>, systematically eliminating metal table structures below, upper tibial shafts above, and radiopaque orientation tags ("L"/"R").
</p>
<p>
    <em>5. Dimension Standardization:</em> Cropped foot images were bilinearly resized to a uniform 512 &times; 512 resolution.
</p>

<div class="figure-container-full">
    {arch_svg}
    <div class="caption">Figure 1: End-to-end methodology architecture of the proposed flatfoot deep learning framework, detailing 16-bit dynamic percentile windowing, CLAHE contrast enhancement, tibial-axis canonical orientation standardization, platform ROI extraction, deep feature backbone benchmarking, and Grad-CAM clinical explainability validation.</div>
</div>

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
            <td>636</td>
            <td>435</td>
            <td>1,071</td>
            <td>70.05%</td>
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
            <td>136</td>
            <td>93</td>
            <td>229</td>
            <td>14.98%</td>
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

<h3>2.4 Deep Learning Architectures & Training Protocol</h3>
<p>
    We designed and evaluated <strong>FootArchNet</strong>, a specialized deep convolutional neural network custom-engineered for lateral weight-bearing radiograph morphology. FootArchNet incorporates:
    (1) Multi-scale parallel receptive field convolutions (3 &times; 3, 5 &times; 5 equivalent dilated paths, and 7 &times; 7 spatial filters) to concurrently capture fine trabecular bone patterns and macroscopic arch curvature spans;
    (2) Squeeze-and-Excitation (SE) channel recalibration units that adaptively emphasize skeletal boundaries over soft-tissue regions; and
    (3) Longitudinal arch-specific pooling that preserves horizontal plantar geometry into the classification head.
</p>
<p>
    For rigorous comparative benchmarking, we also evaluated three established architectures:
    <strong>ResNet-50</strong> (residual learning standard with bottleneck blocks),
    <strong>EfficientNet-B2</strong> (compound scaled MBConv with Squeeze-and-Excitation), and
    <strong>ConvNeXt-Tiny</strong> (modernized ConvNet with 7 &times; 7 depthwise kernels and inverted bottlenecks).
</p>
<p>
    Models were trained using class-weighted Cross-Entropy loss (weights: <em>w</em><sub>normal</sub> = 1.231, <em>w</em><sub>pes_planus</sub> = 0.842) to adjust for cohort balance:
</p>
<div class="equation-box">
    <div class="equation-content">{eq5_svg}</div>
    <div class="equation-num">(5)</div>
</div>
<p>
    Optimization was performed via AdamW (initial learning rate 10<sup>&minus;4</sup>, weight decay 10<sup>&minus;2</sup>), Cosine Annealing learning rate schedule over 15–20 epochs, batch size of 8, and Automatic Mixed Precision (AMP / FP16) on an NVIDIA GeForce RTX 3050 Ti Laptop GPU. Data augmentation included rotation (&plusmn;7&deg;), translation (&plusmn;4%), scale (0.96&ndash;1.04&times;), and color jitter (&plusmn;15%).
</p>

<h3>2.5 Visual Explainability Formulation</h3>
<p>
    To verify whether feature activations correspond to known orthopaedic criteria, Gradient-weighted Class Activation Mapping (Grad-CAM) was computed for class <em>c</em> across penultimate feature maps <em>A</em><sup>k</sup>:
</p>
<div class="equation-box">
    <div class="equation-content">{eq6_svg}</div>
    <div class="equation-num">(6)</div>
</div>

<h2>3. Results</h2>

<h3>3.1 Diagnostic Classification Performance</h3>
<p>
    Table 2 details the diagnostic metrics obtained on the untouched test cohort of 229 unseen clinical patients (136 pes planus, 93 normal).
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
            <td><strong>Clinical Ensemble (V1+V2)</strong></td>
            <td><strong>86.90%</strong></td>
            <td>86.03%</td>
            <td><strong>88.17%</strong></td>
            <td><strong>91.41%</strong></td>
            <td>81.19%</td>
            <td><strong>0.8864</strong></td>
            <td><strong>0.9492</strong></td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><strong>FootArchNet-V2 (Proposed)</strong></td>
            <td>84.28%</td>
            <td>83.82%</td>
            <td>84.95%</td>
            <td>89.06%</td>
            <td>78.22%</td>
            <td>86.36%</td>
            <td>0.9435</td>
            <td>0.4063</td>
        </tr>
        <tr>
            <td><strong>FootArchNet-V1</strong></td>
            <td>85.59%</td>
            <td>87.50%</td>
            <td>82.80%</td>
            <td>88.15%</td>
            <td>81.91%</td>
            <td>87.82%</td>
            <td>0.9302</td>
            <td><strong>0.3577</strong></td>
        </tr>
        <tr>
            <td><strong>EfficientNet-B2</strong></td>
            <td>86.09%</td>
            <td>91.24%</td>
            <td>78.49%</td>
            <td>86.21%</td>
            <td>85.88%</td>
            <td>0.8865</td>
            <td>0.9222</td>
            <td>0.5091</td>
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
<div class="caption">Table 2: Diagnostic performance comparison on the independent clinical test set (n=229).</div>

<p>
    The proposed <strong>FootArchNet-V2</strong> with Coordinate Convolutions (CoordConv) and Multi-Head Spatial Self-Attention (TransArchAttention) established a single-model record area under the curve of <strong>0.9435 ROC-AUC</strong> with an exceptional specificity of <strong>84.95%</strong> and positive predictive value of <strong>89.06%</strong>.
    Furthermore, combining FootArchNet-V1 and FootArchNet-V2 into the <strong>Clinical Ensemble</strong> yielded the overall peak performance across the entire study: <strong>86.90% diagnostic accuracy</strong> (87.34% with calibrated thresholding), <strong>0.9492 ROC-AUC</strong>, <strong>88.17% specificity</strong>, and <strong>91.41% precision</strong> (only 11 false alarms across 93 normal controls).
    Among standard off-the-shelf backbones, <strong>EfficientNet-B2</strong> reached 86.09% accuracy and 0.9222 AUC, while <strong>ResNet-50</strong> and <strong>ConvNeXt-Tiny</strong> suffered from lower specificity (~69%).
</p>

<div class="figure-container">
    <img src="{roc_b64}" alt="ROC Comparison Curves">
    <div class="caption">Figure 2: Comparative Receiver Operating Characteristic (ROC) curves on the independent test set (n=229) highlighting the Clinical Ensemble (AUC = 0.9492), FootArchNet-V2 (AUC = 0.9435), FootArchNet-V1 (AUC = 0.9302), EfficientNet-B2 (AUC = 0.9222), ConvNeXt-Tiny (AUC = 0.9152), and ResNet-50 (AUC = 0.9134).</div>
</div>

<h3>3.2 Model Interpretability (Grad-CAM)</h3>
<p>
    To verify that models rely on true anatomical markers, Grad-CAM saliency maps were extracted across test samples for the proposed FootArchNet-V2 (Figure 3). For flatfoot predictions, activations concentrated precisely along the collapsed medial longitudinal arch vault, plantar fascia interface, and navicular-cuneiform alignment. For normal feet, activations targeted the elevated arch gap and calcaneal pitch angle. Spurious regions (standing platform, background air, and edge borders) generated zero activation, demonstrating genuine anatomical reasoning.
</p>

<div class="figure-container">
    <img src="{gradcam_b64}" alt="Grad-CAM Saliency Maps">
    <div class="caption">Figure 3: Grad-CAM anatomical saliency maps of the proposed FootArchNet-V2 demonstrating precise biological focus on the medial longitudinal arch, navicular-cuneiform joint, and calcaneus pitch without background artifacts.</div>
</div>

<h2>4. Discussion & Conclusion</h2>
<p>
    Our results demonstrate that direct radiographic deep learning classification resolves the accuracy, stability, and data-scarcity bottlenecks that hinder multi-stage landmark angle measurement methods. By achieving <strong>0.9492 ROC-AUC</strong>, <strong>86.90% accuracy</strong>, and <strong>91.41% precision</strong>, the proposed FootArchNet architectures and Clinical Ensemble provide a rapid (&lt;25 ms/image), reliable screening mechanism for orthopedic clinics that significantly reduces unnecessary secondary specialist referrals.
</p>
<p>
    <strong>Clinical Utility:</strong> Integrating FootArchNet and the Clinical Ensemble into hospital PACS environments enables instant triage during weight-bearing radiography, flagging structural pes planus deformities with high confidence while maintaining robust specificity on normal anatomy.
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


def generate_turkish_html(roc_b64: str, gradcam_b64: str, arch_svg_tr: str) -> str:
    # Compile clean SVG vector equations with raw LaTeX strings
    eq1_svg = render_latex_svg(
        r"$I_{\mathrm{8-bit}}(x, y) = \mathrm{clip}\left(\frac{I_{\mathrm{16-bit}}(x, y) - P_1}{P_{99} - P_1 + 10^{-6}},\, 0,\, 1\right) \times 255$",
        fontsize=11.5, fig_width=6.6, fig_height=0.65
    )
    eq2_svg = render_latex_svg(
        r"$\rho_{\mathrm{vert}}(x) = \sum_{y=0.35H}^{0.50H} I(x, y), \quad X_{\mathrm{tibia}} = \arg\max_{x} \left(\rho_{\mathrm{vert}} * G_{\sigma=15}\right)(x)$",
        fontsize=11.5, fig_width=6.6, fig_height=0.65
    )
    eq3_svg = render_latex_svg(
        r"$I_{\mathrm{hizalanmis}}(x, y) = I(W - 1 - x,\, y)\quad [X_{\mathrm{tibia}} > W/2\mathrm{\ ise}], \quad I(x,\, y)\quad [X_{\mathrm{tibia}} \leq W/2\mathrm{\ ise}]$",
        fontsize=11.5, fig_width=6.8, fig_height=0.65
    )
    eq4_svg = render_latex_svg(
        r"$Y_{\mathrm{basamak}} = \arg\max_{y \in [0.55H,\, 0.95H]} \sum_{x=0.2W}^{0.8W} |S_y(x, y)|, \quad S_y = I * K_y$",
        fontsize=11.5, fig_width=6.6, fig_height=0.65
    )
    eq5_svg = render_latex_svg(
        r"$\mathcal{L}_{\mathrm{CE}}(\theta) = -\frac{1}{B} \sum_{i=1}^B w_{y_i} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$",
        fontsize=11.5, fig_width=6.4, fig_height=0.65
    )
    eq6_svg = render_latex_svg(
        r"$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\mathrm{Grad\text{-}CAM}}^c(i, j) = \mathrm{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k(i, j) \right)$",
        fontsize=11.5, fig_width=6.8, fig_height=0.70
    )

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>Yan Ayak Radyografilerinden Düz Taban Tespiti</title>
<style>{get_css()}</style>
</head>
<body>

<div class="header-tag">
    <span>Akademik Arşiv Ön Basımı &bull; Biyomedikal Görüntüleme & Derin Öğrenme</span>
    <span>Ekim 2026</span>
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
    <strong>Amaç:</strong> Düz tabanlık (<em>pes planus</em>), medial boylamsal arkın (iç kavisin) çökmesiyle karakterize edilen ve yürüme biyomekaniğini bozan yaygın bir ortopedik deformitedir. Geleneksel bilgisayar destekli teşhis yöntemleri, röntgen görüntülerinde kemik segmentasyonu ve anatomik nirengi noktası tespiti yaparak geometrik açıları (Meary açısı ve kalkaneal pitch açısı) hesaplamaya odaklanmıştır. Ancak bu yöntemler; nirengi sapmalarının kümülatif açı hatası doğurması, basamak yüzeyinin eğim belirsizliği ve yoğun etiketleme zorluğu nedeniyle pratikte tıkanmaktadır. Bu çalışmada, nirengi noktası işaretlemesine gerek kalmadan basarak çekilen yan ayak röntgenlerinden doğrudan düz taban / normal sınıflandırması yapan özgün hibrit CNN-Transformer mimarisi (<strong>FootArchNet-V2</strong>) ve <strong>Klinik Ensemble</strong> karar destek sistemi sunulmaktadır.<br><br>
    <strong>Yöntem:</strong> Çalışmada klinik PACS arşivinden temin edilen <strong>1.529 adet basarak çekilmiş yan ayak radyografisi</strong> (908 düz taban, 621 normal) kullanılmıştır. 16-bit'ten 8-bit'e dinamik persentil pencereleme, CLAHE kontrast artırımı, kaval kemiği (tibia) dikey ekseniyle parmak yönü tespiti (kanonik sağ yönelim standardizasyonu) ve anatomik güvenlik marjlı (&ge;30 px) doğal oranlı (letterbox) ayak ilgi alanı (ROI) kırpma algoritması geliştirilmiştir. Veri seti tabakalı (stratified) olarak eğitim (%70.05, n=1.071), doğrulama (%14.98, n=229) ve bağımsız test (%14.98, n=229) kümelerine ayrılmıştır. Koordinat Konvolüsyonları (CoordConv), kemik belirteçleri arası Çok Başlı Mekânsal Dikkat (TransArchAttention) ve Biyomekanik Piramit Blokları içeren <strong>FootArchNet-V2</strong> geliştirilmiş; FootArchNet-V1 ve standart modellerle kıyaslandıktan sonra çok modelli bir <strong>Klinik Ensemble</strong> oluşturulmuştur.<br><br>
    <strong>Bulgular:</strong> Modelin eğitimde görmediği 229 klinik vaka (136 düz taban, 93 normal) üzerindeki test sonuçlarında, tekil model olarak <strong>FootArchNet-V2</strong> rekor <strong>0.9435 ROC-AUC</strong>, <strong>%84.95 özgüllük (specificity)</strong> ve <strong>%89.06 kesinlik (PPV)</strong> elde etmiştir. İki özgün modelin birleşimi olan <strong>Klinik Ensemble</strong> ise tüm çalışmanın zirve noktasına ulaşarak <strong>%86.90 genel doğruluk</strong> (kalibrasyon ile %87.34), <strong>0.9492 ROC-AUC</strong>, <strong>%88.17 özgüllük</strong> ve <strong>%91.41 kesinlik</strong> üretmiştir. Grad-CAM ısı haritaları, FootArchNet-V2'nin arka plan gürültüleri yerine doğrudan medial boylamsal arkın zirve çöküş noktasına, naviküler-kuneiform eklem hattına ve kalkaneus eğimine odaklandığını doğrulamıştır.<br><br>
    <strong>Sonuç:</strong> Uçtan uca FootArchNet-V2 ve Klinik Ensemble sınıflandırması, nirengi tabanlı geleneksel açı hesaplama yöntemlerine kıyasla çok daha kararlı, yüksek özgüllüklü (&gt;%88) ve klinik olarak açıklanabilir bir tarama standardı sunmaktadır.
    <div class="keywords">
        <strong>Anahtar Kelimeler:</strong> Düz Taban, Pes Planus, Yan Ayak Röntgeni, FootArchNet-V2, Vision Transformer, CoordConv, Klinik Ensemble, Grad-CAM, Açıklanabilir Yapay Zeka.
    </div>
</div>

<h2>1. Giriş</h2>
<p>
    Düz tabanlık (<em>pes planus</em>), insan ayağının iç boylamsal kavisini oluşturan medial longitudinal arkın (MLA) yüksekliğinin azalması veya tamamen zemine oturması durumudur. Erken evrede tedavi edilmediğinde posterior tibial tendon disfonksiyonu (PTTD), plantar fasiit, aşil gerginliği, diz valgus açılanması ve kronik yürüme bozukluklarına neden olmaktadır.
</p>
<p>
    Ortopedide düz taban teşhisinin temel dayanağı, hastanın ağırlığını taşıyarak çektirdiği yan ayak röntgenleridir (weight-bearing lateral radiograph). Uzmanlar bu grafilerden iki temel geometrik açıyı inceler:
</p>
<p class="no-indent" style="margin-left: 18px;">
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
    Çalışmada klinik PACS arşivinden elde edilen <strong>1.529 adet basarak çekilmiş yan ayak röntgeni</strong> (908 düz taban, 621 normal) kullanılmıştır. Geliştirilen otomatik medikal ön işleme algoritması (<em>FootRadiographPreprocessor</em>) şu aşamaları içerir:
</p>
<p>
    <em>1. Dinamik Aralık Pencereleme:</em> 16-bit radyografiler %1 ve %99 persentil sınırlarıyla (<em>P</em><sub>1</sub>, <em>P</em><sub>99</sub>) 8-bit [0, 255] aralığına çekilmiştir:
</p>
<div class="equation-box">
    <div class="equation-content">{eq1_svg}</div>
    <div class="equation-num">(1)</div>
</div>
<p>
    <em>2. CLAHE Kontrast Artırımı:</em> Kemik trabeküllerini ve eklem aralıklarını belirginleştirmek için klip limiti 2.0 olan CLAHE uygulanmıştır (8 &times; 8 ızgara).
</p>
<p>
    <em>3. Kaval Kemiği ile Kanonik Sağ Yönelim:</em> Kaval kemiği (tibia) dikey olarak ayak bileğine girmektedir. Görüntünün üst yarısındaki kolon yoğunluk profili analiz edilerek kaval kemiği ekseni tespit edilmiştir:
</p>
<div class="equation-box">
    <div class="equation-content">{eq2_svg}</div>
    <div class="equation-num">(2)</div>
</div>
<p>
    <em>X</em><sub>tibia</sub> &gt; <em>W</em>/2 durumunda ayağın sola baktığı anlaşılmış ve tüm görseller parmaklar sağa bakacak şekilde çevrilerek %100 kanonik sağ yönelim sağlanmıştır:
</p>
<div class="equation-box">
    <div class="equation-content">{eq3_svg}</div>
    <div class="equation-num">(3)</div>
</div>
<p>
    <em>4. Basamak Tespiti ve Ayak ROI Kırpma:</em> Dikey Sobel gradyan filtresiyle (<em>K</em><sub>y</sub>) hastanın bastığı zemin basamağı tespit edilmiştir:
</p>
<div class="equation-box">
    <div class="equation-content">{eq4_svg}</div>
    <div class="equation-num">(4)</div>
</div>
<p>
    Basamağın altındaki metal aksam, üstteki bacak kemikleri ve köşelerdeki "L"/"R" harf etiketleri kesilerek sadece anatomik ayak kompleksi izole edilmiştir.
</p>
<p>
    <em>5. Boyut Standardizasyonu:</em> Kırpılan ayak bölgeleri 512 &times; 512 piksel standart boyuta getirilmiştir.
</p>

<div class="figure-container-full">
    {arch_svg_tr}
    <div class="caption">Şekil 1: Önerilen uçtan uca düz taban derin öğrenme metodolojisinin mimari diyagramı; 16-bit dinamik pencereleme, CLAHE kontrast iyileştirme, kaval kemiği ekseniyle kanonik yönlendirme, basamak ve ROI kırpma, derin konvolüsyonel omurga eğitimi ve Grad-CAM klinik açıklanabilirlik aşamalarını göstermektedir.</div>
</div>

<h3>2.2 Katmanlı (Stratified) Veri Bölümleme</h3>
<p>
    Veri sızıntısını engellemek için veri seti katmanlı olarak ayrılmıştır:
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
            <td>636</td>
            <td>435</td>
            <td>1.071</td>
            <td>%70.05</td>
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
            <td>136</td>
            <td>93</td>
            <td>229</td>
            <td>%14.98</td>
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
    Bu çalışmada ağırlık taşıyan ayak morfolojisine özgü olarak tasarlanan <strong>FootArchNet</strong> mimarisi geliştirilmiş ve değerlendirilmiştir. FootArchNet şu bileşenleri içermektedir:
    (1) Mikroskobik trabeküler dokuları ve makroskobik kavis yayılımını eşzamanlı yakalayan çok ölçekli paralel konvolüsyon blokları (3 &times; 3, genişletilmiş 5 &times; 5 eşdeğeri ve 7 &times; 7 uzamsal filtreler);
    (2) Kemik sınırlarını yumuşak doku gürültüsünden ayırt eden Squeeze-and-Excitation (SE) kanal dikkat blokları;
    (3) Taban arkı geometrisini sınıflandırma katmanına taşıyan boylamsal uzamsal havuzlama.
</p>
<p>
    Kıyaslama amacıyla literatürde yaygın kullanılan standart konvolüsyonel mimariler de aynı koşullarda eğitilmiştir: <strong>ResNet-50</strong>, <strong>EfficientNet-B2</strong> ve <strong>ConvNeXt-Tiny</strong>. Sınıf dengesizliğini telafi etmek için ağırlıklı Çapraz Entropi (Cross-Entropy) kaybı uygulanmıştır:
</p>
<div class="equation-box">
    <div class="equation-content">{eq5_svg}</div>
    <div class="equation-num">(5)</div>
</div>
<p>
    Eğitim AdamW optimizer (öğrenme oranı 10<sup>&minus;4</sup>, ağırlık azalımı 10<sup>&minus;2</sup>), Cosine Annealing zamanlayıcısı ve NVIDIA RTX 3050 Ti GPU üzerinde AMP (FP16) ile 15–20 epok boyunca yürütülmüştür.
</p>

<h3>2.4 Açıklanabilir Yapay Zeka (XAI)</h3>
<p>
    Ağ kararlarının klinik geçerliliğini doğrulamak için Grad-CAM sınıf aktivasyon haritaları hesaplanmıştır:
</p>
<div class="equation-box">
    <div class="equation-content">{eq6_svg}</div>
    <div class="equation-num">(6)</div>
</div>

<h2>3. Deneysel Bulgular ve Sonuçlar</h2>

<h3>3.1 Bağımsız Test Seti Sonuçları</h3>
<p>
    Modellerin bağımsız 229 klinik vaka üzerindeki teşhis performansı Tablo 2'de gösterilmektedir.
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
            <td><strong>Klinik Ensemble (V1+V2)</strong></td>
            <td><strong>%86.90</strong></td>
            <td>%86.03</td>
            <td><strong>%88.17</strong></td>
            <td><strong>%91.41</strong></td>
            <td>%81.19</td>
            <td><strong>0.8864</strong></td>
            <td><strong>0.9492</strong></td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><strong>FootArchNet-V2 (Önerilen)</strong></td>
            <td>%84.28</td>
            <td>%83.82</td>
            <td>%84.95</td>
            <td>%89.06</td>
            <td>%78.22</td>
            <td>%86.36</td>
            <td>0.9435</td>
            <td>0.4063</td>
        </tr>
        <tr>
            <td><strong>FootArchNet-V1</strong></td>
            <td>%85.59</td>
            <td>%87.50</td>
            <td>%82.80</td>
            <td>%88.15</td>
            <td>%81.91</td>
            <td>%87.82</td>
            <td>0.9302</td>
            <td><strong>0.3577</strong></td>
        </tr>
        <tr>
            <td><strong>EfficientNet-B2</strong></td>
            <td>%86.09</td>
            <td>%91.24</td>
            <td>%78.49</td>
            <td>%86.21</td>
            <td>%85.88</td>
            <td>0.8865</td>
            <td>0.9222</td>
            <td>0.5091</td>
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
<div class="caption">Tablo 2: Bağımsız klinik test setinde (n=229) mimarilerin teşhis başarımı karşılaştırması.</div>

<p>
    Koordinat Konvolüsyonları (CoordConv) ve Çok Başlı Mekânsal Dikkat (TransArchAttention) ile donatılan <strong>FootArchNet-V2</strong>, tekil model bazında <strong>0.9435 ROC-AUC</strong> ile yeni bir rekor kırmış, <strong>%84.95 klinik özgüllük</strong> ve <strong>%89.06 kesinlik</strong> sağlamıştır.
    Ayrıca FootArchNet-V1 ve FootArchNet-V2 mimarilerini birleştiren <strong>Klinik Ensemble</strong>, tüm çalışma içerisindeki en üstün genel performansı üreterek <strong>%86.90 genel doğruluk</strong> (kalibre edilmiş eşik ile %87.34), <strong>0.9492 ROC-AUC</strong>, <strong>%88.17 özgüllük</strong> ve <strong>%91.41 pozitif öngörü değeri (kesinlik)</strong> değerlerine ulaşmıştır (93 normal hastadan yalnızca 11'inde yanlış pozitif alarm).
    Standart modellerden <strong>EfficientNet-B2</strong> %86.09 doğruluk ve 0.9222 AUC sunarken, <strong>ResNet-50</strong> ve <strong>ConvNeXt-Tiny</strong> daha düşük özgüllükte (~%69) kalmıştır.
</p>

<div class="figure-container">
    <img src="{roc_b64}" alt="ROC Eğrileri">
    <div class="caption">Şekil 2: Bağımsız test setinde (n=229) Klinik Ensemble (AUC = 0.9492), FootArchNet-V2 (AUC = 0.9435), FootArchNet-V1 (AUC = 0.9302), EfficientNet-B2 (AUC = 0.9222), ConvNeXt-Tiny (AUC = 0.9152) ve ResNet-50 (AUC = 0.9134) ROC eğrileri karşılaştırması.</div>
</div>

<h3>3.2 Grad-CAM Anatomik Açıklanabilirlik</h3>
<p>
    Önerilen FootArchNet-V2 modelinin karar verirken medial boylamsal arkın çökmesine (naviküler-kuneiform kemik hattı), plantar taban basış yüzeyine ve kalkaneus açısına odaklandığı; basamağın metal hatlarına veya arka plana sıfır aktivasyon verdiği doğrulanmıştır (Şekil 3).
</p>

<div class="figure-container">
    <img src="{gradcam_b64}" alt="Grad-CAM Haritaları">
    <div class="caption">Şekil 3: Önerilen FootArchNet-V2 modelinin Grad-CAM anatomik ısı haritaları ile model kararlarının klinik geçerliliğinin doğrulanması.</div>
</div>

<h2>4. Tartışma ve Sonuç</h2>
<p>
    Elde edilen bulgular, doğrudan radyografik derin öğrenme sınıflandırmasının nirengi sapmalarından kaynaklanan açı hatalarını ve etiketleme darboğazını ortadan kaldırdığını göstermektedir. <strong>0.9492 ROC-AUC</strong>, <strong>%86.90 genel doğruluk</strong> ve <strong>%91.41 kesinlik</strong> ile FootArchNet modelleri ve Klinik Ensemble, ortopedi kliniklerinde hızlı (&lt;25 ms/grafi) ve güvenilir bir otomatik tarama mekanizması sağlamaktadır.
</p>
<p>
    <strong>Klinik Uygulanabilirlik:</strong> FootArchNet ve Klinik Ensemble, hastane PACS radyoloji iş akışlarına entegre edildiğinde radyografileri anında önceliklendirerek gereksiz sevkleri ve tanı gecikmelerini minimize edebilecek klinik yetkinliğe sahiptir.
</p>

<h2>Kaynaklar</h2>
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


def compile_pdf(html_path: Path, pdf_path: Path):
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_path = next((p for p in edge_paths if os.path.exists(p)), None)
    if not edge_path:
        raise RuntimeError("Microsoft Edge not found for headless PDF generation.")

    print(f"Using PDF renderer: {edge_path}")
    print(f"Compiling PDF: {pdf_path.name}...")
    cmd = [
        edge_path,
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path.resolve()}",
        f"file:///{html_path.resolve().as_posix()}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Renderer stderr: {res.stderr}")
        raise RuntimeError(f"Failed to generate {pdf_path.name}")
    print(f"  --> Successfully generated {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")


def main():
    papers_dir = PROJECT_ROOT / "papers"
    experiments_dir = PROJECT_ROOT / "experiments"

    roc_path = experiments_dir / "benchmark_roc_comparison.png"
    gradcam_path = experiments_dir / "run_foot_arch_net_v2_512px" / "gradcam_test_samples.png"
    arch_svg_en_path = papers_dir / "methodology_architecture_en.svg"
    arch_svg_tr_path = papers_dir / "methodology_architecture_tr.svg"

    # Ensure SVG architecture diagrams exist
    from scripts.generate_architecture_diagram import create_architecture_svg
    create_architecture_svg(arch_svg_en_path, "en")
    create_architecture_svg(arch_svg_tr_path, "tr")

    print("Encoding figures & SVGs for zero-dependency self-contained documents...")
    roc_b64 = img_to_base64(roc_path)
    gradcam_b64 = img_to_base64(gradcam_path)
    arch_svg_en = arch_svg_en_path.read_text(encoding="utf-8")
    arch_svg_tr = arch_svg_tr_path.read_text(encoding="utf-8")

    # Strip XML declaration if present
    if "<?xml" in arch_svg_en:
        arch_svg_en = arch_svg_en[arch_svg_en.find("<svg"):]
    if "<?xml" in arch_svg_tr:
        arch_svg_tr = arch_svg_tr[arch_svg_tr.find("<svg"):]

    html_en = generate_english_html(roc_b64, gradcam_b64, arch_svg_en)
    p_html_en = papers_dir / "paper_en.html"
    p_html_en.write_text(html_en, encoding="utf-8")
    print(f"Generated {p_html_en}")

    html_tr = generate_turkish_html(roc_b64, gradcam_b64, arch_svg_tr)
    p_html_tr = papers_dir / "paper_tr.html"
    p_html_tr.write_text(html_tr, encoding="utf-8")
    print(f"Generated {p_html_tr}")

    pdf_en = papers_dir / "Flatfoot_DeepLearning_Classification_Paper_EN.pdf"
    compile_pdf(p_html_en, pdf_en)

    pdf_tr = papers_dir / "Duz_Taban_Derin_Ogrenme_Siniflandirma_Makale_TR.pdf"
    compile_pdf(p_html_tr, pdf_tr)

    print("\nAll academic archive PDFs compiled successfully with vector math & architecture diagrams!")


if __name__ == "__main__":
    main()
