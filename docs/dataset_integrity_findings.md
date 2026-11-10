
# Dataset Integrity Audit — Clone Catcher

## 1. Dataset Information

- Dataset: Real vs Fake Faces Balanced Dataset
- Selected configuration: 8020
- Total images: 10,000
- Training images: 8,000
- Validation images: 1,000
- Test images: 1,000
- Classes: Real and Fake

## 2. Exact Duplicate Audit

SHA-256 hashing was performed across the dataset.

| Check | Result |
|---|---:|
| Total images checked | 10,000 |
| Exact duplicate groups | 0 |
| Cross-split exact duplicates | 0 |

No exact file-level duplicates were detected.

## 3. Perceptual Hash Audit

Perceptual hashing (pHash) was used to identify visually similar images across dataset splits.

Hamming distance threshold: 6

| Split comparison | Candidate pairs |
|---|---:|
| Train vs Validation | 46 |
| Train vs Test | 33 |
| Validation vs Test | 2 |
| Total | 81 |

These are candidate matches, not confirmed duplicates.

### Train-Test Breakdown

| Metric | Result |
|---|---:|
| Candidate pairs | 33 |
| Unique training images involved | 32 |
| Unique test images involved | 29 |
| Distance 0 | 2 |
| Distance 2 | 2 |
| Distance 4 | 8 |
| Distance 6 | 21 |

### Class Combinations

| Combination | Pairs |
|---|---:|
| Fake-Fake | 27 |
| Fake-Real | 2 |
| Real-Fake | 3 |
| Real-Real | 1 |

## 4. Visual Inspection

Two contact sheets were generated:

- `near_duplicate_review.png`: 12 strongest candidate pairs.
- `train_test_contact_sheet.png`: 33 train-test candidate pairs.

Visual inspection identified multiple strongly similar facial images, including images with nearly identical poses, expressions, backgrounds and framing.

Some candidate pairs were visually different, indicating perceptual-hash false positives.

Therefore, not all candidate matches can be classified as duplicates.

## 5. Pixel-Level Comparison

The 12 strongest candidate pairs were compared after resizing images to 256 × 256 RGB.

| Metric | Result |
|---|---:|
| Pairs checked | 12 |
| Exact pixel matches | 0 |
| Minimum MAE | 4.445 |
| Maximum MAE | 13.396 |
| Changed pixels range | 12.33%–45.16% |

Pixel-level differences do not rule out near-duplicate content. They also do not independently establish common source provenance.

Detailed results: `pixel_similarity_audit.csv`

## 6. Integrity Concerns

The audit identified potential cross-split near-duplicate contamination.

This raises concerns that models may encounter visually similar images in training and evaluation sets.

However, the available checks do not establish:

- Whether images originate from the same source photograph.
- Whether multiple images belong to the same facial identity.
- Whether manipulated images originate from the same source video.
- Whether the dataset split is independent at source-identity level.

## 7. Impact on Model Evaluation

Previously obtained test results:

| Model | Accuracy |
|---|---:|
| Custom CNN | 83.40% |
| Xception | 99.00% |

These results are retained as preliminary experimental results.

They should not be interpreted as proof of generalization to unseen identities or independent source media.

A more rigorous evaluation protocol is required before making generalization claims.

## 8. Next Actions

1. Investigate dataset documentation and source provenance.
2. Determine whether identity or source-level metadata is available.
3. Investigate all cross-split candidate pairs.
4. Establish a defensible independent evaluation split if possible.
5. Retrain models using the revised training and validation data.
6. Evaluate once on an untouched, independent test set.
7. Document limitations and reproducibility details.

## 9. Research Integrity

No images have been deleted or modified as part of this audit.

Existing model checkpoints and evaluation results are preserved.

No performance metrics have been fabricated or silently replaced.

**Audit status:** Preliminary integrity investigation completed; source-level independence remains unverified.



## 10. Original Dataset Provenance

The selected Zenodo dataset is a restructured version of RVF10K.

Original dataset record:
- Title: Real vs Fake Faces Balanced Dataset with Multiple Dataset Splits
- Authors: Debasish Samal, Prateek Agrawal, Vishu Madaan
- Version: 1.0
- DOI: 10.5281/zenodo.14532968
- Original RVF10K source: Kaggle

According to the Zenodo documentation:
- Real images originate from NVIDIA's Flickr-Faces-HQ (FFHQ).
- Fake images originate from Bojan Tunguz's 1 Million Fake Faces dataset.

### Research Scope Implication

The current dataset primarily represents real photographs versus GAN-generated synthetic faces.

It does not independently establish coverage of face-swapping or other facial manipulation techniques.

The exact generator/version and identity-level or source-level split methodology remain unverified.

Consequently, experiments on this dataset must be described as real-versus-synthetic-face classification, not as comprehensive face-swap detection.

### Provenance Reference

Samal, D., Agrawal, P., & Madaan, V. (2024).
Real vs Fake Faces Balanced Dataset with Multiple Dataset Splits.
Zenodo. https://doi.org/10.5281/zenodo.14532968

