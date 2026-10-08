# Academic Benchmark Comparison (Test Set - 229 Unseen Clinical Patients)

Evaluation of standard radiology architectures vs. dedicated FootArchNet architectures and Multi-Scale Super Ensemble:

| Model | Accuracy | Sensitivity | Specificity | Precision | NPV | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet-50 | 84.35% | 94.89% | 68.82% | 81.76% | 90.14% | 87.84% | 0.9134 |
| EfficientNet-B2 | 86.09% | 91.24% | 78.49% | 86.21% | 85.88% | 88.65% | 0.9222 |
| ConvNeXt-Tiny | 84.35% | 94.16% | 69.89% | 82.17% | 89.04% | 87.76% | 0.9152 |
| DenseNet-201 | 86.90% | 88.24% | 84.95% | 89.55% | 83.16% | 88.89% | 0.9434 |
| FootArchNet-V1 | 85.59% | 87.50% | 82.80% | 88.15% | 81.91% | 87.82% | 0.9302 |
| **FootArchNet-V2** | 84.28% | 83.82% | 84.95% | 89.06% | 78.22% | 86.36% | 0.9435 |
| **FootArchNet-Ultra** | 87.77% | 89.71% | 84.95% | 89.71% | 84.95% | 89.71% | 0.9506 |
| **Super Ensemble (TTA, Calibrated)** | 90.39% | 88.24% | 93.55% | 95.24% | 84.47% | 91.60% | 0.9564 |

*All models evaluated on the standardized letterbox test cohort (136 pes planus, 93 normal controls).* 
