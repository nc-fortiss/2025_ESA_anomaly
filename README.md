# 2025_ESA_anomaly
Hackathon NC

Official baseline for **ESA spacecraft telemetry anomaly classification**, developed as an open starting point for the **ESA Anomaly Detection Hackathon 2025**.

This repository provides everything you need to get started — from data loading and preprocessing to model training, evaluation, and Akida-ready deployment.

---

##  Overview

Spacecraft generate enormous amounts of multichannel telemetry data — detecting anomalies in these signals early can prevent mission failures. 
This baseline uses **deep learning on time-series data** to detect anomalies in the **ESA-Mission1 dataset**, and is designed for:
- **Satellite telemetry analysis**
- **Real-time anomaly detection**
- **BrainChip Akida neuromorphic deployment**
---

##  Features

- ✅ **dataset loader** (`dataloader_tf.py`) – reads 76 channel zip files, merges, resamples, labels, and windows
- ✅ **Akida-friendly CNN model** (TensorFlow 2.15)
- ✅ **Automatic caching and normalization**
- ✅ **1-minute sampling rate and 14-year coverage**
- ✅ **Class balancing and focal loss ready**
- ✅ **TensorBoard visualizations**

---

##  Project structure

ESA-Anomaly-Baseline/
│
├── dataloader_tf.py # Dataset loader & preprocessor
├── train.py # Main training script
├── requirements.txt
├── LICENSE # MIT License
├── logs/ # TensorBoard runs
├── ESA-Mission1/ # Dataset root 
│ ├── channels/
│ ├── labels.csv
│ ├── anomaly_types.csv
│ ├── telecommands.csv
│ └── channels.csv
└── norm_stats.npz


---

##  Setup

### 1. Clone the repository
```bash
https://github.com/Kannan-Priya/2025_ESA_anomaly.git
cd ESA-Anomaly-Baseline
bash setup_akida_env.sh
```

Place the ESA-Mission1 dataset in the folder
python train.py
tensorboard --logdir logs/fit --port 6006


Contact Priya Kannan, Kannan@fortiss.org
Neuromorphic Computing Group, fortiss GmbH (Munich)



