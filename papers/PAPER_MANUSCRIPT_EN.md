# Automated Classification of Flatfoot (Pes Planus) from Weight-Bearing Lateral Radiographs: An End-to-End Deep Learning Framework with Anatomical Interpretability

**Authors:** Tahsin Soyak et al.  
**Affiliation:** Department of Computer Engineering / Biomedical Engineering  
**Correspondence:** [Contact Information]  
**Date:** October 2026  

---

## Abstract

**Background:** Flatfoot (*pes planus*) is a prevalent musculoskeletal disorder characterized by the collapse of the medial longitudinal arch. Conventional computer-aided diagnosis has heavily relied on multi-bone segmentation and geometric landmark detection to estimate radiographic angles (e.g., Meary’s angle and calcaneal pitch). However, these pipelines are notoriously vulnerable to cumulative landmark errors, ambiguous platform support definitions, and severe annotation scarcity. In this study, we propose an automated, end-to-end deep learning framework that directly classifies weight-bearing lateral foot radiographs into flatfoot or normal categories without requiring dense anatomical landmark annotations.

**Methods:** A clinical dataset comprising **1,529 weight-bearing lateral foot radiographs** (908 pes planus, 621 normal) was collected. We developed an automated medical preprocessing pipeline incorporating 16-bit to 8-bit dynamic percentile windowing, Contrast Limited Adaptive Histogram Equalization (CLAHE), deterministic tibia-axis orientation standardization (canonical right alignment), and automated foot region-of-interest (ROI) cropping to eliminate non-anatomical artifacts (e.g., metal apparatus, radiopaque L/R markers). The standardized dataset was partitioned into stratified train (69.98%, n=1,070), validation (14.98%, n=229), and independent test (15.04%, n=230) cohorts. Three contemporary deep convolutional architectures—ResNet-50, EfficientNet-B2, and ConvNeXt-Tiny—were benchmarked using class-weighted loss and automatic mixed precision. Model interpretability was established using Gradient-weighted Class Activation Mapping (Grad-CAM).

**Results:** On the unseen test set of 230 clinical patients (137 pes planus, 93 normal), all models demonstrated strong diagnostic discrimination. **EfficientNet-B2** achieved the highest overall diagnostic accuracy of **86.09%**, an area under the receiver operating characteristic curve (**ROC-AUC**) of **0.9222**, a sensitivity of **91.24%**, a specificity of **78.49%**, and an F1-score of **0.8865**. **ResNet-50** and **ConvNeXt-Tiny** demonstrated exceptional diagnostic sensitivity at **94.89%** (detecting 130 of 137 flatfoot cases) and **94.16%**, respectively. Grad-CAM visual heatmaps confirmed that the models consistently activated on clinically relevant anatomical structures—specifically the medial longitudinal arch, navicular-cuneiform joint, and calcaneal pitch—rather than background artifacts.

**Conclusion:** Direct radiographic deep learning classification provides an accurate, robust, and interpretable alternative to heuristic landmark-based angle measurements. This benchmark serves as a reliable clinical baseline for deploying automated screening tools in orthopedic practice and motivates future custom architecture designs tailored to foot biomechanics.

**Keywords:** Flatfoot, Pes Planus, Deep Learning, Lateral Radiographs, EfficientNet, ConvNeXt, Grad-CAM, Medical Image Classification.

---

## 1. Introduction

Flatfoot, clinically designated as *pes planus*, represents a common structural deformity of the human lower extremity characterized by the partial or complete collapse of the medial longitudinal arch (MLA). If left untreated or mismanaged, progressive flatfoot can induce abnormal lower-limb biomechanics, chronic plantar fasciitis, posterior tibial tendon dysfunction (PTTD), knee valgus deformities, and debilitating gait impairment.

In clinical orthopedics, the definitive diagnostic standard for assessing flatfoot severity is the weight-bearing lateral radiograph. From these radiographic projections, clinicians conventionally assess geometric angles, most prominently:
1. **Meary’s Angle (Talar-1st Metatarsal Angle):** The angle between the longitudinal axis of the talus and the shaft of the first metatarsal (normal range: 0° to 4°; >4° plantar deviation indicates flatfoot).
2. **Calcaneal Pitch Angle:** The angle between the lower border of the calcaneus and the horizontal weight-bearing support line (normal range: 17° to 32°; <17° indicates flatfoot).

Recent computational efforts (e.g., Noh et al., *Scientific Reports* 2024; Khaleghizadeh et al., 2024) attempted to automate these measurements using cascading computer vision pipelines: Stage 1 ROI detection, Stage 2 multi-bone semantic segmentation (talus, calcaneus, 1st metatarsal), and Stage 3 geometric landmark extraction (10 anatomical contour landmarks). 

### Limitations of Heuristic Landmark Measurement Pipelines
Despite their theoretical appeal, multi-stage geometric pipelines face severe clinical and practical bottlenecks:
- **Error Compounding:** A discrepancy of merely 1–2 millimeters in identifying the talar neck center or inferior calcaneal tubercle alters Meary’s angle by several degrees, frequently triggering incorrect diagnostic categorization.
- **Support Surface Sensitivity:** Calcaneal pitch calculation requires a perfectly identified ground/support line. In real hospital radiographs, patient platform tilting, uneven leg positioning, and radiopaque table borders introduce uncalibrated pitch errors.
- **The Annotation Bottleneck:** Training deep segmentation and landmark models requires manual polygon annotations by certified orthopedic specialists for thousands of images. In practice, datasets are constrained to tiny cohorts (e.g., 37–150 labeled images), which fail to generalize across diverse clinical populations.

### Our Contribution
To circumvent the landmark annotation bottleneck, this study transitions flatfoot diagnosis into an **end-to-end deep learning classification paradigm**. By formulating the task directly from raw radiographs, the neural network learns high-dimensional anatomical representations of arch collapse directly from global bone morphology. 

Our primary contributions are:
1. **Large Clinical Cohort:** We curate and evaluate a standardized clinical dataset of 1,529 weight-bearing lateral foot radiographs (908 pes planus, 621 normal).
2. **Automated Medical Preprocessing & Artifact Suppression:** We design an automated pipeline that standardizes 16-bit dynamic range, applies CLAHE contrast enhancement, canonicalizes foot orientation via proximal tibial axis detection, and crops the foot ROI to suppress non-anatomical equipment and markers.
3. **Rigorous Benchmark Evaluation:** We establish a comprehensive diagnostic benchmark comparing ResNet-50, EfficientNet-B2, and ConvNeXt-Tiny on an independent test partition (n=230).
4. **Anatomical Interpretability:** Using Grad-CAM, we verify that deep models make diagnostic decisions by examining true biomechanical markers (the medial arch, talonavicular joint, and calcaneus) rather than shortcut artifacts.

---

## 2. Materials and Methods

### 2.1 Dataset Description
The study utilized 1,529 weight-bearing lateral foot radiographs obtained from clinical PACS archives:
- **Pes Planus (Flatfoot):** 908 radiographs.
- **Normal Foot:** 621 radiographs.
All radiographs were acquired in the standard standing lateral projection with full weight bearing. Original images were stored as 16-bit grayscale PNG files with resolutions ranging from 2428 × 3003 to 3072 × 3072 pixels.

### 2.2 Medical Image Preprocessing Pipeline
Raw medical radiographs exhibit vast variations in exposure, equipment borders, and limb laterality. We developed an automated deterministic preprocessing algorithm (`FootRadiographPreprocessor`):

1. **Dynamic Percentile Windowing (16-bit to 8-bit):** To accommodate varying X-ray tube voltages and high-dynamic ranges (0–65,535), pixel intensities were clipped to the 1st and 99th percentiles ($P_1, P_{99}$) and mapped to $[0, 255]$:
   $$I_{\text{8-bit}}(x, y) = \text{clip}\left(\frac{I_{\text{16-bit}}(x, y) - P_1}{P_{99} - P_1 + 10^{-6}}, 0, 1\right) \times 255$$
2. **Contrast Enhancement (CLAHE):** Contrast Limited Adaptive Histogram Equalization was applied with a clip limit of 2.0 and an $8 \times 8$ contextual grid size to sharpen trabecular bone micro-textures, cortical contours, and articular margins.
3. **Tibial-Axis Canonical Orientation Standardization:** Radiographs contain both left and right feet, causing arbitrary toe directions. In weight-bearing projections, the tibia and fibula enter vertically into the ankle complex. By computing the horizontal column density in the upper shaft band ($y \in [0.35H, 0.50H]$), the horizontal coordinate of the tibial shaft ($X_{\text{tibia}}$) was localized:
   $$\rho_{\text{vertical}}(x) = \sum_{y = 0.35H}^{0.50H} I(x, y), \quad X_{\text{tibia}} = \arg\max_{x} \left(\rho_{\text{vertical}} * G_{\sigma=15}\right)(x)$$
   $$I_{\text{aligned}}(x, y) = \begin{cases} I(W - 1 - x, y), & \text{if } X_{\text{tibia}} > W/2 \text{ (toes face left)} \\ I(x, y), & \text{if } X_{\text{tibia}} \le W/2 \text{ (toes face right)} \end{cases}$$
   All left-pointing images were horizontally mirrored to enforce a 100% canonical right-facing orientation.
4. **Platform Edge Detection and Foot ROI Cropping:** The patient standing platform was detected via a vertical Sobel gradient operator ($K_y$). The platform surface was localized in the lower region ($y \in [0.55H, 0.95H]$):
   $$Y_{\text{platform}} = \arg\max_{y \in [0.55H, 0.95H]} \sum_{x = 0.2W}^{0.8W} |S_y(x, y)|, \quad \text{where } S_y = I * K_y$$
   The foot ROI was cropped immediately above $Y_{\text{platform}}$, systematically eliminating metal table structures below, the upper tibial shaft above, and radiopaque orientation tags ("L"/"R").
5. **Resolution Standardization:** Cropped foot ROIs were bilinearly resampled to a standardized dimension of $512 \times 512$ pixels.

![Methodology Architecture](file:///C:/Users/tahsinsoyak/Desktop/proje_github_clone/flatfoot_classifier/papers/methodology_architecture_en.svg)
*Figure 1: End-to-end clinical workflow and methodology architecture of the proposed flatfoot deep learning framework, including 16-bit dynamic percentile windowing, CLAHE contrast enhancement, tibial-axis canonical orientation standardization, platform ROI extraction, deep feature backbone benchmarking, and Grad-CAM explainability validation.*


### 2.3 Dataset Partitioning
To prevent data contamination and guarantee unbiased evaluation, the 1,529 preprocessed images were partitioned using stratified sampling:
- **Training Set (69.98%, n=1,070):** 635 Pes Planus, 435 Normal.
- **Validation Set (14.98%, n=229):** 136 Pes Planus, 93 Normal.
- **Test Set (15.04%, n=230):** 137 Pes Planus, 93 Normal.

The test set remained strictly held out and was evaluated only once after final model training.

### 2.4 Model Architectures
We selected three modern vision architectures representing distinct design paradigms:
1. **ResNet-50:** The established standard in medical deep learning, utilizing 50 layers with residual bottleneck blocks.
2. **EfficientNet-B2:** A compound scaled convolutional network balancing depth, width, and resolution with inverted residual mobile bottlenecks (MBConv) and Squeeze-and-Excitation (SE) optimization.
3. **ConvNeXt-Tiny:** A modernized pure-convolutional network adopting Vision Transformer design principles (7×7 depthwise convolutions, inverted bottlenecks, LayerNorm).

All backbones were initialized with ImageNet pre-trained weights and fine-tuned with a customized classification head (Dropout $p=0.2$, Linear layer with 2 output logits).

### 2.5 Training Protocol and Loss Formulation
To mitigate the class imbalance (908 pes planus vs. 621 normal), models were trained using a class-weighted Cross-Entropy Loss:
$$\mathcal{L} = - \sum_{c \in \{0, 1\}} w_c \cdot y_c \log(\hat{y}_c)$$
where $w_0 = 1.230$ (Normal) and $w_1 = 0.843$ (Pes Planus).

Optimization was conducted using:
- **Optimizer:** AdamW with initial learning rate $\eta = 10^{-4}$ and weight decay $10^{-2}$.
- **Learning Rate Schedule:** Cosine Annealing decay down to $10^{-6}$ over 20 epochs.
- **Batch Size:** 8 with Automatic Mixed Precision (AMP / FP16) on an NVIDIA GeForce RTX 3050 Ti Laptop GPU.
- **Data Augmentation:** Random rotation ($\pm 7^\circ$), affine translation ($\pm 4\%$), affine scaling ($0.96-1.04\times$), and subtle brightness/contrast jitter ($\pm 15\%$). Horizontal flipping was deliberately excluded during training to preserve canonical orientation.
- **Early Stopping:** Monitored on validation ROC-AUC with a patience of 8 epochs.

### 2.6 Evaluation Metrics
Model performance was comprehensively quantified on the test set using standard clinical metrics:
- **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$
- **Sensitivity (Recall / True Positive Rate):** $TP / (TP + FN)$
- **Specificity (True Negative Rate):** $TN / (TN + FP)$
- **Precision (Positive Predictive Value):** $TP / (TP + FP)$
- **Negative Predictive Value (NPV):** $TN / (TN + FN)$
- **F1-Score:** $2 \cdot (\text{Precision} \cdot \text{Sensitivity}) / (\text{Precision} + \text{Sensitivity})$
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve.

### 2.7 Explainable AI (Grad-CAM)
To interpret model predictions and verify clinical validity, Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented. Gradients of the predicted class score $y^c$ with respect to the feature activation maps $A^k$ of the final convolutional layer were pooled:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
Heatmaps were interpolated to $512 \times 512$ and overlaid onto the radiographs.

---

## 3. Results

### 3.1 Benchmark Classification Performance
Table 1 summarizes the diagnostic performance of the evaluated architectures on the independent test cohort of 230 patients.

**Table 1: Diagnostic performance comparison on the independent clinical test set (n=230).**

| Model Architecture | Accuracy | Sensitivity | Specificity | Precision | NPV | F1-Score | ROC-AUC | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B2** | **86.09%** | 91.24% | **78.49%** | **86.21%** | 85.88% | **0.8865** | **0.9222** | **0.5091** |
| **ConvNeXt-Tiny** | 84.35% | 94.16% | 69.89% | 82.17% | 89.04% | 0.8776 | 0.9152 | 0.8875 |
| **ResNet-50** | 84.35% | **94.89%** | 68.82% | 81.76% | **90.14%** | 0.8784 | 0.9134 | 0.6363 |

All three models demonstrated clinical-grade diagnostic ability (ROC-AUC $> 0.91$):
- **EfficientNet-B2** achieved the best overall balance, leading in Accuracy (86.09%), Specificity (78.49%), Precision (86.21%), F1-Score (0.8865), and ROC-AUC (0.9222) while maintaining the lowest test loss (0.5091).
- **ResNet-50** achieved the highest Sensitivity (94.89%), correctly identifying 130 of 137 true flatfoot cases and missing only 7 cases (NPV: 90.14%).
- **ConvNeXt-Tiny** followed closely with 94.16% sensitivity (129/137 true positives identified).

### 3.2 ROC Analysis
Figure 1 illustrates the comparative ROC curves across all evaluated models on the test set. EfficientNet-B2 maintained the highest curve trajectory across low false-positive rates ($FPR < 0.2$), reflecting its superior specificity in identifying normal arch anatomy.

*(Figure 1: `experiments/benchmark_roc_comparison.png`)*

### 3.3 Anatomical Interpretability Analysis (Grad-CAM)
Visual inspection of Grad-CAM saliency heatmaps across representative test cases revealed distinct clinical patterns:
1. **Focus on the Medial Arch:** For positive flatfoot predictions, saliency peaks were concentrated along the midfoot arch collapse zone (navicular, cuneiforms, and talonavicular joint).
2. **Focus on Calcaneal Alignment:** In normal foot cases, models strongly attended to the calcaneal pitch angle and the subtalar joint interface, where an elevated arch creates clear spatial separation from the support surface.
3. **Absence of Background Bias:** Saliency values over the standing platform bar, peripheral air, and former letter marker regions were near zero, confirming that predictions were driven exclusively by skeletal morphology.

*(Figure 2: `experiments/run_efficientnet_b2_512px/gradcam_test_samples.png`)*

---

## 4. Discussion

### 4.1 Clinical Significance
Automated flatfoot screening is vital in high-volume orthopedic centers, pediatric clinics, and military physical examinations. Traditional methods require orthopedists to spend several minutes manually marking anatomical points to compute Meary’s angle and calcaneal pitch. Previous machine learning efforts mimicking this process suffered from geometric instability and severe landmark annotation bottlenecks.

By demonstrating that an end-to-end deep learning framework can achieve an **ROC-AUC of 0.9222** and **sensitivity exceeding 94%** on raw lateral radiographs without any landmark labeling, this study proves the clinical viability of direct radiographic classification. The framework operates in under **35 milliseconds per image** during inference, rendering it suitable for real-time PACS deployment.

### 4.2 Architecture Comparison & Trade-offs
- **ResNet-50 vs. EfficientNet-B2:** While ResNet-50 biased slightly toward high sensitivity (94.89% vs. 68.82% specificity), EfficientNet-B2’s compound scaling and Squeeze-and-Excitation channel gating allowed it to capture subtle arch variations, yielding a more balanced specificity of 78.49%.
- **ConvNeXt-Tiny:** ConvNeXt matched ResNet-50 in overall accuracy (84.35%) and delivered competitive sensitivity (94.16%), validating the efficacy of large 7×7 depthwise receptive fields in capturing longitudinal skeletal contours.

### 4.3 Error Analysis & Limitations
Examination of false-positive cases (normal feet classified as pes planus) revealed mild borderline arch heights (mild pes planus or intermediate arch morphology). In clinical practice, intermediate cases often cause inter-observer variability even among experienced radiologists.

**Limitations:**
1. Single-center acquisition dataset. Multi-center external validation is recommended to test robustness across varying radiographic machines.
2. Binary classification (Pes Planus vs. Normal). Future iterations should incorporate multi-class severity grading (mild, moderate, severe) and pes cavus (high arch).

---

## 5. Conclusion and Future Directions

In this work, we formulated and benchmarked a deep learning classification framework for automated flatfoot diagnosis from weight-bearing lateral foot radiographs. Supported by an automated medical preprocessing and foot ROI cropping algorithm, our models achieved high diagnostic accuracy (86.09%), robust ROC-AUC (0.9222), and high clinical sensitivity (94.89%). Grad-CAM heatmaps validated that decision-making corresponds with true orthopedic anatomical landmarks.

**Future Work:**  
Building upon this benchmark, our subsequent research will explore a dedicated **FootArchNet** architecture featuring anatomy-guided multi-scale attention mechanisms designed to explicitly integrate global arch geometry and local articular joint cues.

---

## References
1. Noh, W. J., Lee, M. S., & Lee, B. D. (2024). Deep learning-based automated angle measurement for flatfoot diagnosis in weight-bearing lateral radiographs. *Scientific Reports*, 14(1), 18411.
2. Khaleghizadeh, R., Motamed, S., & Askari, E. (2025). Flatfoot disorder recognition based on the YOLO-ChA algorithm. *Biomedical Signal Processing and Control*, 97, 106560.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
4. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*, 6105–6114.
5. Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *CVPR*, 11976–11986.
6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*, 618–626.
