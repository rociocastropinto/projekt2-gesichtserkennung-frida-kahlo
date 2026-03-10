# Projekt 2: Gesichtserkennung & Verpixelung
**Face Detection & Anonymisation — Case Study: Frida Kahlo**

A Python tool that automatically detects faces in photographs and anonymises
them using four different methods. Two pretrained classifiers are implemented
and compared: Haar Cascade and a DNN-based ResNet-10 SSD detector.

---

## View Results (no installation needed)
[Open notebook with outputs on nbviewer](https://nbviewer.org/github/rociocastropinto/projekt2-gesichtserkennung-frida-kahlo/blob/main/frida_kahlo_anonymization.ipynb)

## Run in the Browser
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rociocastropinto/projekt2-gesichtserkennung-frida-kahlo/HEAD)

---

## Run Locally
```bash
git clone https://github.com/rociocastropinto/projekt2-gesichtserkennung-frida-kahlo
cd projekt2-gesichtserkennung-frida-kahlo
pip install -r requirements.txt
jupyter notebook
```

Open `frida_kahlo_anonymization.ipynb` and click **Run All**.

> The `models/` folder is already included in the repository.
> No internet connection is required to run the pipeline.

---

## Features

- **Two face detectors** — Haar Cascade (fast, classic) and DNN ResNet-10 SSD
  (more accurate, robust to angle and lighting variation)
- **Four anonymisation methods** — pixelation, Gaussian blur, black-box
  masking, and Canny edge sketch
- **Anonymisation strength metric** — quantitative edge-density comparison
  printed as a formatted table for all four methods
- **Interactive console menu** — choose detector, method, and folder at
  runtime without editing the code
- **Jupyter widget GUI** — dropdown menus and sliders for use inside a notebook
- **Detector comparison visualisation** — side-by-side Haar vs DNN output
- **Method comparison chart** — bar chart of strength scores saved as PNG
- **Automatic Haar fallback** — if DNN model files are unavailable, the
  pipeline continues using Haar Cascade

---

## Repository Structure

| File / Folder | Contents |
|---|---|
| `frida_kahlo_anonymization.ipynb` | Main notebook with GUI and pre-run outputs |
| `frida_kahlo_anonymization_FINAL.py` | Console version with interactive menu |
| `requirements.txt` | Python dependencies |
| `models/` | Pretrained model files (Haar XML + DNN weights, already downloaded) |
| `frida_images/` | Test images — Frida Kahlo portraits (public domain) |
| `outputs/` | Generated anonymised images and comparison charts |

---

## Requirements

- Python 3.8 or higher
- Dependencies: `pip install -r requirements.txt`
```
opencv-python
numpy
matplotlib
ipywidgets
```

---

## Case Study

The pipeline was tested on portrait photographs of **Frida Kahlo** (1907–1954),
whose images are in the public domain. This corpus was chosen because her
portraits vary widely in lighting, angle, and photographic style, making them
a useful testbed for evaluating detector robustness. The case study also
situates the tool within a Digital Humanities context: automated face
anonymisation is directly relevant to GDPR compliance when working with
photographs of living persons in research settings such as oral history,
ethnography, or archival digitisation projects.

---

## Key Sources

| Source | Role in this project |
|---|---|
| Viola & Jones (2001), CVPR | Haar Cascade algorithm |
| OpenCV DNN Team (2017) | ResNet-10 SSD pretrained model |
| Brehm / ORB-HD (2020), [deface](https://github.com/ORB-HD/deface) — MIT | Architecture reference |
| Rosebrock (2020), [PyImageSearch](https://pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/) | ROI pipeline pattern |
| Kudoes (2020), [RealTime-Face-Anonymization](https://github.com/Kudoes/RealTime-Face-Anonymization) | Student-level DNN reference |
| Canny (1986), IEEE TPAMI | Edge detection algorithm |
| GDPR (2016), Official Journal EU | Data protection context |

Full bibliography available in the written report.

---

## Notes for the Examiner

- The notebook was executed in full before submission. All figures and output
  tables are visible without running any code.
- To re-run from scratch: **Kernel → Restart and Run All**.
- Model files are included in `models/` so no internet access is needed.
- If package installation fails (e.g. restricted university environment), run
  manually: `pip install opencv-python numpy matplotlib ipywidgets`
