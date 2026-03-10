"""
Projekt 2: Gesichtserkennung & Verpixelung
Case Study: Frida Kahlo

Improvements over baseline:
  - Two classifiers: Haar Cascade + DNN (ResNet-10 SSD)
  - Four anonymisation methods: pixel, blur, black_box, canny
  - OOP design: FaceDetector, Anonymizer, Pipeline classes
  - Interactive ipywidgets GUI
  - Detector comparison visualisation
  - Summary bar chart (report-ready)
  - Full type hints + docstrings
"""

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES & ATTRIBUTION
#
# This code was developed by combining and extending ideas from:
#
# [1] ORB-HD/deface (MIT License)
#     https://github.com/ORB-HD/deface
#     Inspiration for the DNN-detector + interchangeable anonymisation filter
#       architecture. The deface tool uses a similar detect→mask pipeline.
#
# [2] charlsefrancis/Blur-and-anonymize-faces-with-OpenCV-and-Python
#     https://github.com/charlsefrancis/Blur-and-anonymize-faces-with-OpenCV-and-Python
#     Based on: Rosebrock, A. (2020). PyImageSearch.
#     The four-step ROI approach: detect → extract ROI → anonymise → replace back.
#       The pixelate() and blur_face() implementations follow this pattern.
#
# [3] Kudoes/RealTime-Face-Anonymization
#     https://github.com/Kudoes/RealTime-Face-Anonymization
#     Use of the OpenCV res10_300x300_ssd DNN model for face detection
#       combined with Gaussian blur and pixelation output methods.
#
# Extensions beyond these sources (original contributions):
#   - Object-oriented design (HaarDetector, DNNDetector, Anonymizer, Pipeline)
#   - Anonymisation strength metric (edge-density reduction)
#   - Detector comparison visualisation (compare_detectors)
#   - Method comparison bar chart (plot_method_comparison)
#   - ipywidgets interactive GUI (launch_gui)
#   - Frida Kahlo Digital Humanities case study
# ─────────────────────────────────────────────────────────────────────────────

# ─── Install required packages (run once) ────────────────────────────────────
# NOTE: If this fails, install manually:
#   pip install opencv-python numpy matplotlib ipywidgets
import subprocess, sys
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "opencv-python", "numpy", "matplotlib", "ipywidgets", "--quiet"],
        check=True
    )
    print("✓ Packages installed.\n")
except Exception as e:
    print(f"⚠ Auto-install failed: {e}")
    print("  → Please run manually: pip install opencv-python numpy matplotlib ipywidgets\n")
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import os
import urllib.request
from pathlib import Path
from matplotlib import pyplot as plt
from typing import List, Tuple, Optional

# ─── 0. Model download helpers ────────────────────────────────────────────────
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

_DNN_PROTO   = MODEL_DIR / "deploy.prototxt"
_DNN_WEIGHTS = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
_HAAR_XML    = MODEL_DIR / "haarcascade_frontalface_default.xml"

_MODEL_URLS = {
    _DNN_PROTO:   "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    _DNN_WEIGHTS: "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    _HAAR_XML:    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
}

def download_models() -> bool:
    all_ok = True
    for path, url in _MODEL_URLS.items():
        if path.exists():
            print(f"  ✓ {path.name} (cached)")
            continue
        try:
            print(f"  Downloading {path.name} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  ✓ {path.name}")
        except Exception as e:
            print(f"  ⚠ Could not download {path.name}: {e}")
            all_ok = False
    if all_ok:
        print("✓ Models ready.\n")
    else:
        print("⚠ Some models missing. DNN detector may be unavailable; Haar will be used as fallback.\n")
    return all_ok

_DNN_AVAILABLE = download_models()


# ─── 1. Face Detectors ────────────────────────────────────────────────────────

class HaarDetector:
    """
    Classic Viola-Jones Haar Cascade classifier (2001).
    Fast and CPU-efficient; best for frontal, well-lit faces.
    """

    def __init__(self, xml_path: str = str(_HAAR_XML)):
        self.clf = cv2.CascadeClassifier(xml_path)
        if self.clf.empty():
            raise RuntimeError(f"Failed to load Haar XML: {xml_path}")

    def detect(
        self,
        img: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> List[Tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) face bounding boxes."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves detection in variable lighting
        faces = self.clf.detectMultiScale(
            gray, scaleFactor=scale_factor,
            minNeighbors=min_neighbors, minSize=min_size
        )
        return [tuple(f) for f in faces] if len(faces) > 0 else []


class DNNDetector:
    """
    Deep Neural Network detector using ResNet-10 SSD (OpenCV DNN module).
    More robust to rotation, partial occlusion, and lighting variation.
    Outputs a confidence score per detection.
    """

    def __init__(
        self,
        proto:   str = str(_DNN_PROTO),
        weights: str = str(_DNN_WEIGHTS),
    ):
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)

    def detect(
        self,
        img: np.ndarray,
        confidence_thr: float = 0.5,
    ) -> List[Tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) face bounding boxes above confidence threshold."""
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > confidence_thr:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes


# Singletons (loaded once, reused throughout)
haar_detector = HaarDetector()

# ── DNN fallback: use Haar if models could not be downloaded ──────────────────
if _DNN_AVAILABLE:
    dnn_detector = DNNDetector()
else:
    print("⚠ DNN model unavailable — using Haar Cascade as fallback for 'DNN' requests.")
    dnn_detector = haar_detector   # fallback: same object, graceful degradation

DETECTORS = {"Haar": haar_detector, "DNN": dnn_detector}


# ─── 2. Anonymisation methods ─────────────────────────────────────────────────

class Anonymizer:
    """
    Collection of static anonymisation methods for face ROIs.

    Methods
    -------
    pixel    – mosaic pixelation (GDPR-standard in press)
    blur     – Gaussian blur (visually smooth, hard to reverse)
    black_box– solid black rectangle (maximum information removal)
    canny    – edge-sketch (creative / artistic extension)
    """

    @staticmethod
    def pixel(roi: np.ndarray, pixel_size: int = 16, **_) -> np.ndarray:
        """Pixelate by downsampling then upsampling with NEAREST interpolation."""
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def blur(roi: np.ndarray, strength: int = 51, **_) -> np.ndarray:
        """Apply strong Gaussian blur. Kernel must be odd."""
        s = strength if strength % 2 == 1 else strength + 1
        return cv2.GaussianBlur(roi, (s, s), 30)

    @staticmethod
    def black_box(roi: np.ndarray, **_) -> np.ndarray:
        """Replace region with solid black."""
        return np.zeros_like(roi)

    @staticmethod
    def canny(roi: np.ndarray, **_) -> np.ndarray:
        """Artistic edge-sketch anonymisation (inverted Canny edges)."""
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 100, 200)
        return cv2.cvtColor(cv2.bitwise_not(edges), cv2.COLOR_GRAY2BGR)

    @classmethod
    def apply(cls, roi: np.ndarray, method: str, **kwargs) -> np.ndarray:
        """Dispatch to the named method. Raises ValueError for unknown names."""
        fn = getattr(cls, method, None)
        if fn is None:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Choose from: pixel, blur, black_box, canny")
        return fn(roi, **kwargs)


# ─── 3. Anonymisation strength metric ────────────────────────────────────────

def anonymization_strength_percent(
    original: np.ndarray,
    anonymized: np.ndarray,
) -> float:
    """
    Edge-density reduction metric:
        strength = 100 * (1 - edges_after / edges_before)

    Interpretation:
        ~100% → almost all edges removed (strong anonymisation)
        ~0%   → edge structure largely preserved
    """
    o = cv2.cvtColor(original,   cv2.COLOR_BGR2GRAY)
    a = cv2.cvtColor(anonymized, cv2.COLOR_BGR2GRAY)
    o_dens = np.mean(cv2.Canny(o, 80, 160) > 0)
    a_dens = np.mean(cv2.Canny(a, 80, 160) > 0)
    if o_dens == 0:
        return 0.0
    return float(np.clip(100.0 * (1.0 - a_dens / o_dens), 0.0, 100.0))


# ─── 4. Core pipeline ─────────────────────────────────────────────────────────

class Pipeline:
    """
    Orchestrates detection → anonymisation → metric → save.

    Parameters
    ----------
    detector_name : "Haar" or "DNN"
    method        : "pixel", "blur", "black_box", "canny"
    output_dir    : directory to save results
    draw_metrics  : whether to overlay bounding box + strength % on output
    pixel_size    : granularity for pixelation method
    """

    def __init__(
        self,
        detector_name: str = "DNN",
        method:        str = "pixel",
        output_dir:    str = "outputs",
        draw_metrics:  bool = True,
        pixel_size:    int  = 16,
    ):
        self.detector      = DETECTORS[detector_name]
        self.detector_name = detector_name
        self.method        = method
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.draw_metrics  = draw_metrics
        self.pixel_size    = pixel_size

    def process_image(
        self,
        img: np.ndarray,
        show: bool = True,
        save_name: Optional[str] = None,
    ) -> dict:
        """
        Process a single BGR image.

        Returns
        -------
        dict with keys: output, faces, metrics, avg_strength
        """
        faces   = self.detector.detect(img)
        output  = img.copy()
        metrics = []

        for (x, y, w, h) in faces:
            original_roi = img[y:y+h, x:x+w].copy()
            anon_roi     = Anonymizer.apply(
                output[y:y+h, x:x+w],
                self.method,
                pixel_size=self.pixel_size
            )
            output[y:y+h, x:x+w] = anon_roi

            strength = anonymization_strength_percent(original_roi, anon_roi)
            metrics.append(strength)

            if self.draw_metrics:
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output, f"{strength:.1f}%",
                            (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        avg = (sum(metrics) / len(metrics)) if metrics else 0.0

        if show:
            self._show_comparison(img, output, avg)

        if save_name:
            out_path = self.output_dir / save_name
            cv2.imwrite(str(out_path), output)
            print(f"  ✓ Saved: {out_path}  |  Faces: {len(faces)}  |  "
                  f"Avg strength: {avg:.1f}%")

        return {"output": output, "faces": faces,
                "metrics": metrics, "avg_strength": avg}

    def _show_comparison(self, original: np.ndarray, result: np.ndarray,
                         avg_strength: float):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=12, weight="bold"); axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title(
            f"Anonymised ({self.method}) via {self.detector_name} "
            f"| Strength: {avg_strength:.1f}%",
            fontsize=12, weight="bold"
        )
        axes[1].axis("off")
        plt.tight_layout(); plt.show()

    def process_folder(
        self,
        folder_path: str,
        show: bool = True,
        limit: Optional[int] = 13,
    ) -> List[dict]:
        """Batch process all images in a folder."""
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}"); return []

        files = sorted([f for f in folder.iterdir()
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if limit:
            files = files[:limit]

        results = []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"⚠ Cannot read: {f.name}"); continue
            save_name = f"{f.stem}_anon_{self.method}_{self.detector_name}{f.suffix}"
            result = self.process_image(img, show=show, save_name=save_name)
            result["filename"] = f.name
            results.append(result)
        return results


# ─── 5. Detector comparison visualisation ────────────────────────────────────

def compare_detectors(img: np.ndarray, title: str = "") -> None:
    """
    Side-by-side visualisation of Haar Cascade vs DNN detections.
    Demonstrates use of both classifiers as required by the brief.
    """
    h_faces = haar_detector.detect(img)
    d_faces = dnn_detector.detect(img)

    h_img, d_img = img.copy(), img.copy()
    for (x, y, w, h) in h_faces:
        cv2.rectangle(h_img, (x, y), (x+w, y+h), (255, 100, 0), 3)
        cv2.putText(h_img, "Haar", (x, max(y-8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    for (x, y, w, h) in d_faces:
        cv2.rectangle(d_img, (x, y), (x+w, y+h), (0, 200, 100), 3)
        cv2.putText(d_img, "DNN", (x, max(y-8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(h_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Haar Cascade  [{len(h_faces)} face(s)]",
                      fontsize=12, weight="bold", color="darkorange"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"DNN ResNet-10  [{len(d_faces)} face(s)]",
                      fontsize=12, weight="bold", color="seagreen"); axes[1].axis("off")
    fig.suptitle(f"Classifier Comparison | {title}", fontsize=13, weight="bold")
    plt.tight_layout(); plt.show()

    print(f"  Haar: {len(h_faces)} face(s)  |  DNN: {len(d_faces)} face(s)")


# ─── 6. Summary bar chart (report-ready) ──────────────────────────────────────

def plot_method_comparison(
    img: np.ndarray,
    methods: List[str] = ["pixel", "blur", "black_box", "canny"],
) -> None:
    """
    For a single image, compute anonymisation strength for each method
    and visualise as a bar chart + example strip.
    Useful for the written report (Bericht 25%).
    """
    faces = dnn_detector.detect(img)
    if not faces:
        print("No faces detected for comparison plot."); return

    x, y, w, h = faces[0]
    original_roi = img[y:y+h, x:x+w].copy()

    strengths, examples = [], []
    for m in methods:
        anon_roi = Anonymizer.apply(original_roi.copy(), m)
        strengths.append(anonymization_strength_percent(original_roi, anon_roi))
        examples.append(cv2.cvtColor(anon_roi, cv2.COLOR_BGR2RGB))

    fig = plt.figure(figsize=(14, 6))
    gs  = fig.add_gridspec(1, 2, width_ratios=[2, 1])

    # — example strip —
    ax_strip = fig.add_subplot(gs[0])
    strip = np.concatenate(examples, axis=1)
    ax_strip.imshow(strip)
    ax_strip.set_xticks(
        [(i + 0.5) * original_roi.shape[1] for i in range(len(methods))]
    )
    ax_strip.set_xticklabels(methods, fontsize=11)
    ax_strip.set_yticks([])
    ax_strip.set_title("Anonymised ROI per Method", fontsize=12, weight="bold")

    # — bar chart —
    ax_bar = fig.add_subplot(gs[1])
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax_bar.barh(methods, strengths, color=colors[:len(methods)])
    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Anonymisation Strength (%)", fontsize=11)
    ax_bar.set_title("Metric Comparison", fontsize=12, weight="bold")
    for bar, s in zip(bars, strengths):
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{s:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("outputs/method_comparison_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Chart saved to outputs/method_comparison_chart.png")


# ─── 7. Interactive GUI (ipywidgets) ──────────────────────────────────────────

def launch_gui():
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        print("ipywidgets not available – run: pip install ipywidgets")
        return

    folder_in  = widgets.Text(value="frida_images",
                               description="Folder:", layout=widgets.Layout(width="50%"))
    det_dd     = widgets.Dropdown(options=["DNN", "Haar"], value="DNN",
                                   description="Detector:")
    method_dd  = widgets.Dropdown(options=["pixel", "blur", "black_box", "canny"],
                                   value="pixel", description="Method:")
    psize_sl   = widgets.IntSlider(value=16, min=4, max=64, step=4,
                                   description="Pixel size:", continuous_update=False)
    limit_sl   = widgets.IntSlider(value=5, min=1, max=13,
                                   description="Max images:")
    save_cb    = widgets.Checkbox(value=True, description="Save results")
    run_btn    = widgets.Button(description="▶ Run", button_style="success",
                                layout=widgets.Layout(width="120px"))
    out        = widgets.Output()

    display(widgets.VBox([
        widgets.HBox([folder_in]),
        widgets.HBox([det_dd, method_dd]),
        widgets.HBox([psize_sl, limit_sl, save_cb]),
        run_btn, out
    ]))

    def on_run(_):
        with out:
            clear_output(wait=True)
            pipe = Pipeline(
                detector_name=det_dd.value,
                method=method_dd.value,
                output_dir="outputs",
                pixel_size=psize_sl.value,
            )
            pipe.process_folder(folder_in.value, show=True, limit=limit_sl.value)

    run_btn.on_click(on_run)


# ─── 8. Console input helper ─────────────────────────────────────────────────

def ask_user_choices() -> tuple:
    """
    Interactive console prompts so the user can choose detector,
    anonymisation method, and image folder without editing the code.
    Returns (folder, detector_name, method).
    """
    print("=" * 60)
    print("  Frida Kahlo – Face Anonymisation Pipeline")
    print("  UDigital Humanities")
    print("=" * 60)

    # ── Folder ────────────────────────────────────────────────────────────────
    folder = input("\nImage folder path [default: frida_images]: ").strip()
    if not folder:
        folder = "frida_images"

    # ── Detector ──────────────────────────────────────────────────────────────
    print("\nAvailable detectors:")
    print("  [1] DNN  – ResNet-10 SSD  (more accurate, recommended)")
    print("  [2] Haar – Haar Cascade   (faster, classic)")
    det_input = input("Choose detector [1/2, default: 1]: ").strip()
    detector  = "Haar" if det_input == "2" else "DNN"

    # ── Method ────────────────────────────────────────────────────────────────
    print("\nAvailable anonymisation methods:")
    print("  [1] pixel     – Mosaic pixelation (GDPR-standard)")
    print("  [2] blur      – Gaussian blur (smooth, hard to reverse)")
    print("  [3] black_box – Solid black rectangle (maximum removal)")
    print("  [4] canny     – Edge-sketch (artistic/creative)")
    print("  [5] all       – Run all four methods and print comparison table")
    meth_input = input("Choose method [1-5, default: 1]: ").strip()
    mapping    = {"1": "pixel", "2": "blur", "3": "black_box", "4": "canny", "5": "all"}
    method     = mapping.get(meth_input, "pixel")

    return folder, detector, method


# ─── 9. Numerical comparison table ───────────────────────────────────────────

def print_comparison_table(img: np.ndarray) -> None:
    """
    Detect faces with DNN, apply all four anonymisation methods to the
    first detected face ROI, and print a formatted numerical comparison
    table of anonymisation strength scores to the console.
    Useful as a quick report reference.
    """
    faces = dnn_detector.detect(img)
    if not faces:
        print("  No faces detected – cannot compute comparison table.")
        return

    x, y, w, h = faces[0]
    original_roi = img[y:y+h, x:x+w].copy()

    methods = ["pixel", "blur", "black_box", "canny"]
    results = []
    for m in methods:
        anon_roi = Anonymizer.apply(original_roi.copy(), m)
        strength = anonymization_strength_percent(original_roi, anon_roi)
        results.append((m, strength))

    print("\n" + "─" * 45)
    print(f"  {'Method':<12}  {'Strength (%)':>12}  {'Assessment'}")
    print("─" * 45)
    for method, score in results:
        if score >= 90:
            label = "★★★  Very strong"
        elif score >= 70:
            label = "★★   Strong"
        elif score >= 40:
            label = "★    Moderate"
        else:
            label = "○    Low"
        print(f"  {method:<12}  {score:>11.1f}%  {label}")
    print("─" * 45)
    best = max(results, key=lambda r: r[1])
    print(f"  Best: {best[0]} ({best[1]:.1f}%)")
    print("─" * 45 + "\n")


# ─── 10. Main execution block ─────────────────────────────────────────────────

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)

    FOLDER, DETECTOR, METHOD = ask_user_choices()

    files = sorted(Path(FOLDER).glob("*.jpg"))
    img0  = cv2.imread(str(files[0])) if files else None

    # ── Step 1: Detector comparison ──────────────────────────────────────────
    if img0 is not None:
        print("\n[1/3] Detector comparison (Haar vs DNN):")
        compare_detectors(img0, title=files[0].name)

    # ── Step 2: Anonymisation (chosen method or all) ──────────────────────────
    run_methods = ["pixel", "blur", "black_box", "canny"] if METHOD == "all" else [METHOD]
    for m in run_methods:
        print(f"\n[2/3] Processing: method = '{m}', detector = {DETECTOR}")
        pipe = Pipeline(detector_name=DETECTOR, method=m, output_dir="outputs")
        pipe.process_folder(FOLDER, show=True, limit=13)

    # ── Step 3: Numerical comparison table + chart ────────────────────────────
    if img0 is not None:
        print("\n[3/3] Numerical method comparison:")
        print_comparison_table(img0)
        plot_method_comparison(img0)

    print("\n✓ All done. Results in ./outputs/")