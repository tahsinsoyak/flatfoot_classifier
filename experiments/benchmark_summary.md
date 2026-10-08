# Academic Benchmark Comparison (Test Set)

Comparison of standard deep learning classification backbones versus the proposed FootArchNet architecture on unseen clinical lateral radiographs:

| Model | Accuracy | Sensitivity | Specificity | Precision | NPV | F1-Score | ROC-AUC | Test Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet-50** | 84.35% | 94.89% | 68.82% | 81.76% | 90.14% | 87.84% | 0.9134 | 0.6363 |
| **ConvNeXt-Tiny** | 84.35% | 94.16% | 69.89% | 82.17% | 89.04% | 87.76% | 0.9152 | 0.8875 |
| **EfficientNet-B2** | 86.09% | 91.24% | 78.49% | 86.21% | 85.88% | 88.65% | 0.9222 | 0.5091 |
| **FootArchNet (Proposed)** | 85.59% | 87.50% | 82.80% | 88.15% | 81.91% | 87.82% | 0.9302 | 0.3577 |

*All models evaluated under stratified test split with letterbox standardized canonical orientation.* 
