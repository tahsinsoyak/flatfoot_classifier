# Academic Benchmark Comparison (Test Set - 230 Cases)

Comparison of standard deep learning classification backbones on unseen clinical lateral radiographs:

| model           | accuracy   | sensitivity   | specificity   | precision   | npv    | f1_score   |   roc_auc |   test_loss |
|:----------------|:-----------|:--------------|:--------------|:------------|:-------|:-----------|----------:|------------:|
| resnet50        | 84.35%     | 94.89%        | 68.82%        | 81.76%      | 90.14% | 87.84%     |    0.9134 |      0.6363 |
| efficientnet_b2 | 86.09%     | 91.24%        | 78.49%        | 86.21%      | 85.88% | 88.65%     |    0.9222 |      0.5091 |
| convnext_tiny   | 84.35%     | 94.16%        | 69.89%        | 82.17%      | 89.04% | 87.76%     |    0.9152 |      0.8875 |

*All models evaluated under identical stratified test split (137 Pes Planus, 93 Normal).* 
