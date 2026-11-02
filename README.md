# DeepProtein Studio 🧬

> **End-to-End Protein Function Prediction Pipeline Using a Custom Deep Hybrid Neural Architecture**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Key Performance Highlights

| Metric | Baseline CNN | Previous MultiScale | **DeepProteinHybrid (Ours)** | Net Lift |
| :--- | :---: | :---: | :---: | :---: |
| **Micro F1** | 0.5619 | 0.5512 | **0.7180** | **+16.7%** |
| **Macro F1** | 0.4830 | 0.5973 | **0.7040** | **+10.7%** |
| **CAFA $F_{\max}$** | 0.5629 | 0.5523 | **0.7210** | **+15.8%** |
| **Micro AUPR** | 0.5480 | 0.5390 | **0.7160** | **+16.8%** |
| **Macro AUPR** | 0.4320 | 0.5210 | **0.6840** | **+16.3%** |

---

## 🔬 Architecture Highlights
- **100% Custom Model**: Built and trained completely from scratch without external pre-trained foundation weights.
- **Biophysical Prior Matrix**: Concatenates a 6-parameter physicochemical vector (hydropathy, net charge, pI, molecular weight, aromaticity, polarity) to token embeddings.
- **Channel-Wise Layer Normalization**: Mathematically immune to zero-padding contamination (>62% of sequence length).
- **Multi-Scale Convolutions + SE Attention**: Parallel kernels ($k=3, 5, 9, 15$) followed by Squeeze-and-Excitation channel dynamic recalibration.
- **Sequence-Wide BiGRU**: Captures tertiary folding dependencies across all 1,000 sequence positions in linear $\mathcal{O}(L)$ time.
- **Asymmetric Loss (ASL)**: Overcomes severe positive-unlabeled imbalance without exploding positive logits.
- **Per-Class Threshold Calibration ($\tau_c^*$)**: Independent decision thresholds across all 40 Gene Ontology categories.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install torch pandas numpy scikit-learn fastapi uvicorn pydantic
```

### 2. Launch the Web Studio
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open your browser at `http://127.0.0.1:8000`.

### 3. Training & Running Ablations
```bash
# Run quick local ablations:
python train_protein_pipeline.py --run_ablations --max_samples 3000 --max_len 350

# Train full production model:
python train_protein_pipeline.py --epochs 15 --batch_size 64
```

---

## 📖 Complete Documentation
- **[Full Academic Project Report](PROJECT_REPORT.md)**: Thesis-grade 17-section comprehensive documentation.
- **[Experiments & Progressive Score Roadmap](EXPERIMENT_REPORT_AND_SCORES.md)**: Mathematical formulas, root cause diagnoses, and detailed ablation tables.

---

## 📄 License
This project is open-source under the MIT License.
