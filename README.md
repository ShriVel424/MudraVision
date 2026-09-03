# MudraVision

Real-time **Bharatanatyam mudra recognition** using computer vision and machine learning.

MudraVision uses **MediaPipe** to extract 21 hand landmarks and a **KNN classifier** to recognize six Bharatanatyam mudras from a webcam.

### Features

* Real-time hand tracking
* 6 Bharatanatyam mudras
* Landmark normalization
* KNN classification
* Confidence scores and mudra meanings
* Bharatanatyam-inspired UI

### Tech Stack

**Python · MediaPipe · OpenCV · NumPy · Pandas · scikit-learn · Pillow**

### Results

**192 samples · 6 classes · 100% test accuracy**

| Mudra        | Samples |
| ------------ | ------: |
| Pataka       |      28 |
| Tripataka    |      28 |
| Ardhapataka  |      40 |
| Kartarimukha |      32 |
| Alapadma     |      28 |
| Musti        |      36 |

> Current model is a proof-of-concept trained on a self-collected dataset.
