# Academic Benchmark Comparison (Test Set - 229 Unseen Clinical Patients)

Comparison of standard deep backbones versus FootArchNet-V1, FootArchNet-V2, and the Clinical Ensemble:

| Model | Accuracy | Sensitivity | Specificity | Precision | NPV | F1-Score | ROC-AUC | Test Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet-50 | 84.35% | 94.89% | 68.82% | 81.76% | 90.14% | 87.84% | 0.9134 | 0.6363 |
| ConvNeXt-Tiny | 84.35% | 94.16% | 69.89% | 82.17% | 89.04% | 87.76% | 0.9152 | 0.8875 |
| EfficientNet-B2 | 86.09% | 91.24% | 78.49% | 86.21% | 85.88% | 88.65% | 0.9222 | 0.5091 |
| FootArchNet-V1 | 85.59% | 87.50% | 82.80% | 88.15% | 81.91% | 87.82% | 0.9302 | 0.3577 |
| **FootArchNet-V2 (Proposed)** | 84.28% | 83.82% | 84.95% | 89.06% | 78.22% | 86.36% | 0.9435 | 0.4063 |
| **Clinical Ensemble (V1+V2)** | 86.90% | 86.03% | 88.17% | 91.41% | 81.19% | 88.64% | 0.9492 | N/A |

*All models evaluated on the standardized letterbox test cohort.* 
