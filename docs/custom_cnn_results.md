\# Custom CNN — Evaluation Results



\## 1. Model Details



\- Model: Custom CNN

\- Architecture: 4 convolutional blocks

\- Parameters: 389,410

\- Input size: 224 × 224 × 3

\- Optimizer: Adam

\- Learning rate: 0.001

\- Batch size: 16

\- Training epochs: 20

\- Random seed: 42

\- Best checkpoint: Epoch 19



\## 2. Dataset



\- Dataset: Real vs Fake Faces

\- Configuration: 80:20

\- Training images: 8,000

\- Validation images: 1,000

\- Test images: 1,000

\- Real/Fake classes: 50:50 balanced



\## 3. Test Results



| Metric | Score |

|---|---:|

| Test Loss | 0.3640 |

| Accuracy | 83.40% |

| Precision | 87.95% |

| Recall | 77.40% |

| F1 Score | 82.34% |

| ROC-AUC | 0.9201 |



\## 4. Confusion Matrix



Rows = Actual, Columns = Predicted



| | Predicted Real | Predicted Fake |

|---|---:|---:|

| Actual Real | 447 | 53 |

| Actual Fake | 113 | 387 |



\## 5. Observations



\- The model correctly classified 834 out of 1,000 test images.

\- Fake image precision was 87.95%.

\- Fake image recall was 77.40%.

\- The model missed 113 fake images.

\- The ROC-AUC score of 0.9201 indicates strong class discrimination on this test split.



\## 6. Limitations



\- Evaluation was performed on one dataset.

\- Cross-dataset generalization has not yet been evaluated.

\- Performance on unseen manipulation methods is unknown.

\- The model may be sensitive to dataset-specific patterns.



\## 7. Conclusion



The custom CNN achieved 83.40% test accuracy and an F1 score of 82.34% on the held-out test split.



These results establish the baseline for comparison with the pretrained Xception model.



No claim of real-world generalization is made at this stage.

