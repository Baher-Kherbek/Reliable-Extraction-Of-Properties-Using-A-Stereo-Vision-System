# 🍎 Fruit Property Extraction & Maturity Grading System

> A computer vision graduation project that combines YOLOv6 detection, GrabCut segmentation, K-Means color clustering, and stereo vision depth estimation to automatically grade fruit maturity and size — without physical contact.

---

## 🎯 Project Overview

This project is the **graduation project for a Bachelor's degree in Mechatronics Engineering at Tishreen University**. It presents a fully automated pipeline to extract two critical fruit properties:

- **Size** — measured in centimeters using stereo vision disparity
- **Color** — dominant color palette extracted via K-Means clustering

These properties are then used to determine **fruit maturity grade**, enabling non-contact, automated quality control suitable for agricultural and food processing applications.

---

## 🔬 Technical Pipeline

### Stage 1 — Fruit Detection (`yolo.py`)
- **YOLOv6** object detection model trained on a multi-fruit Kaggle dataset
- Outputs a **bounding box** and **classification label** per detected fruit

### Stage 2 — Object Isolation (`main.py`, `frame.py`)
- **GrabCut Algorithm** isolates the fruit from the background within the bounding box
- Morphological operations (dilation, erosion) clean up the binary mask
- Outputs a crisp, isolated fruit image for further analysis

### Stage 3 — Size Measurement (Stereo Vision Regression)
- A **stereo camera** (Camera Binoculars 3D-1mP02) provides two offset viewpoints
- **Disparity** between the left/right views is computed per pixel
- A rational regression curve `α = a / (d - b) + c` maps disparity → real-world scale factor
- Multiplying α by the pixel dimension of the fruit yields the **actual size in cm**
- Regression fitting performed via `scipy.optimize.curve_fit` (Dogbox method)

### Stage 4 — Color Analysis (`points.py`)
- **K-Means Clustering** (unsupervised ML) identifies the K dominant colors of the isolated fruit
- The dominant color clusters are used to infer **ripeness stage** based on color thresholds

---

## 📁 Repository Structure

```
├── src/
│   ├── main.py                    # Full pipeline orchestration
│   ├── yolo.py                    # YOLOv6 detection wrapper
│   ├── frame.py                   # Frame acquisition + GrabCut isolation
│   ├── points.py                  # K-Means color clustering
│   └── Regression/
│       ├── DataCollect.py         # Stereo calibration data collection
│       ├── Regression.py          # Curve fitting (disparity → size)
│       └── Plot.py                # Regression visualization
├── Results/
│   ├── Isolation/                 # GrabCut segmentation samples
│   ├── Regression/Images/         # Stereo calibration images (26 samples)
│   └── Results.png                # Summary result visualization
└── Project Reports/
    ├── Project Draft.pdf
    └── Regression.pdf
```

---

## 📊 Results

- ✅ Accurate fruit detection across multiple species
- ✅ Clean object isolation via GrabCut with morphological refinement
- ✅ Real-world size estimation calibrated from stereo disparity regression
- ✅ Color-based maturity grading via K-Means dominant color extraction
- Full results will be uploaded upon publication of the accompanying research paper

---

## 🚀 Getting Started

```bash
git clone https://github.com/Baher-Kherbek/Reliable-Extraction-Of-Properties-Using-A-Stereo-Vision-System.git
cd Reliable-Extraction-Of-Properties-Using-A-Stereo-Vision-System

pip install numpy opencv-python scipy matplotlib ultralytics

# Run calibration regression
python src/Regression/Regression.py

# Run full pipeline
python src/main.py
```

> **Note**: Requires a stereo camera setup. Adjust camera indices and calibration paths in `DataCollect.py`.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-GrabCut%20|%20Stereo-green?logo=opencv)
![YOLOv6](https://img.shields.io/badge/YOLOv6-Object%20Detection-red)
![SciPy](https://img.shields.io/badge/SciPy-Curve%20Fitting-orange)
![KMeans](https://img.shields.io/badge/K--Means-Color%20Clustering-purple)

---

## 🙏 Acknowledgements

This project is the graduation project for a Bachelor's in Mechatronics Engineering at **Tishreen University**.  
Special acknowledgement to the project supervisor, **Dr. Nael Daoud**.

---

## 👤 Author

**Baher Kherbek** — Robotics Engineer & AI Systems Developer  
[github.com/Baher-Kherbek](https://github.com/Baher-Kherbek)
