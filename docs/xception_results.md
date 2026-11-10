

\# Xception — Training and Evaluation Results



\## 1. Model Details



\- Model: Xception

\- Architecture: Pretrained Xception (ImageNet weights)

\- Implementation: timm

\- Total Parameters: 20,811,050

\- Classifier: Linear(2048, 2)

\- Input Size: 224 × 224 × 3

\- Optimizer: Adam

\- Learning Rate: 0.0001

\- Batch Size: 8

\- Training Epochs: 15

\- Random Seed: 42

\- Mixed Precision: FP16

\- GPU: NVIDIA GeForce RTX 4050 Laptop GPU



\## 2. Dataset



\- Dataset: Real vs Fake Faces

\- Configuration: 80:20

\- Training Images: 8,000

\- Validation Images: 1,000

\- Test Images: 1,000

\- Real Images in Test Set: 500

\- Fake Images in Test Set: 500



\## 3. Best Checkpoint



\- Best Checkpoint: checkpoints/xception\_best.pth

\- Best Epoch: 11

\- Best Validation F1: 0.99298



The best checkpoint was selected using validation F1 score.



\## 4. Test Evaluation Results



The saved best checkpoint was evaluated on the untouched test split.



| Metric | Score |

|---|---:|

| Test Loss | 0.0329 |

| Accuracy | 99.00% |

| Precision | 98.80% |

| Recall | 99.20% |

| F1 Score | 99.00% |

| ROC-AUC | 99.96% |



\## 5. Confusion Matrix



Rows = Actual, Columns = Predicted



| | Predicted Real | Predicted Fake |

|---|---:|---:|

| Actual Real | 494 | 6 |

| Actual Fake | 4 | 496 |



\### Classification Summary



\- Correct predictions: 990

\- Incorrect predictions: 10

\- Real images correctly classified: 494

\- Fake images correctly classified: 496

\- Real images incorrectly classified as fake: 6

\- Fake images incorrectly classified as real: 4



\## 6. Observations



\- Xception achieved 99% test accuracy.

\- Precision was 98.80%.

\- Recall was 99.20%.

\- ROC-AUC was 99.96%.

\- The model demonstrated strong discrimination between real and fake images on this dataset's test split.



\## 7. Limitations



\- Evaluation was performed on a single dataset.

\- Cross-dataset generalization has not yet been established.

\- Identity-level and source-video-level split leakage have not yet been ruled out.

\- Near-duplicate and dataset-specific artifact analysis remains necessary.

\- Real-world performance on unseen manipulation techniques is unknown.



\## 8. Conclusion



The pretrained Xception model achieved 99.00% test accuracy and an F1 score of 99.00% on the held-out test split.



It outperformed the custom CNN baseline on this dataset.



These results establish Xception as the stronger model in the current experiment. Further cross-dataset evaluation and leakage analysis are required before making claims about real-world generalization.



