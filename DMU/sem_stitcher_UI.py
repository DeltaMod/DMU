"""
SEM Panorama Stitcher — Applet
===============================

A thin PyQt6 GUI wrapped around your existing SEM stitching pipeline.

Left column
-----------
- "Far pics" list: the frames that get warped/blended into the panorama.
  Add files, reorder with Up/Down, and toggle "Scalebar" per image
  (off by default — burn a scalebar into any far pic you like).
- "Near pics" list: the detail insets shown alongside the panorama, in the
  same order as the far pics they correspond to. Scalebar is ON by default
  but toggleable per image, same as far pics.
- DPI spinner for the saved output resolution.
- Generate / Save buttons.

Right column
------------
- Live matplotlib canvas showing the generated composite.

--------------------------------------------------------------------------
BEFORE RUNNING — fix these two things for your machine:
--------------------------------------------------------------------------
1. The import block below assumes your helper functions
   (SEM_Strip_Banner_And_Enhance, SEM_Scalebar_Generator, ANY_Image_Enhance,
   svg_to_pil, kwarg_aliasing, dmp.get_tab20bc) live in a module called
   `DMU.sem_utils` / `DMU.plotting`. Point these at wherever they actually
   live in your DMU package.
2. INKSCAPE_PATH is a Windows path in your original snippet — set it to
   your Linux inkscape binary (probably just "inkscape" if it's on $PATH,
   or "/usr/bin/inkscape").

--------------------------------------------------------------------------
Bugs I noticed and fixed while adapting your code (please sanity-check):
--------------------------------------------------------------------------
- `scale1` / `scale2` / `savefig` were referenced but never assigned in
  your original snippet. I've inferred scale1/scale2 as the far-pic and
  near-pic pixel sizes of the LAST image pair (matching how you compute
  `scale_difference` elsewhere as `pix_size_near / pix_size_far`). Verify
  this is what you actually want — it's a guess based on context.
- `kwarg_aliasing(scalebar_style, seam_dict_defaults, ...)` and the
  equivalent line for `txt_style` were both defaulting against
  `seam_dict_defaults` in your original code. I fixed these to use
  `scalebar_style_defaults` / `txt_style_defaults` respectively.
- Dropped the separate `figsolo` panorama-only figure — the composite
  figure's `ax_pan` already draws the panorama with the coloured frame
  overlays, so the applet just shows/saves that one.
- The rectangle overlay size (`w_rect`/`h_rect`) is still taken from a
  single pixel-size ratio (last image pair) applied to *all* far pics, same
  as your original — this is only correct if your far pics all share pixel
  size. Flag it if that's not the case and I'll make it per-image.
"""
import sys
if __package__ is None or __package__ == "":
    
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    __package__ = "DMU"

import os    
os.environ["QT_API"] = "PyQt5"

import matplotlib
import traceback
import ssl

import numpy as np
import cv2 as cv
import torch
import PIL.Image
import certifi

import kornia.feature as KF
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QCheckBox,
    QFileDialog, QSpinBox, QSplitter, QMessageBox,
)
from matplotlib.figure import Figure


# --------------------------------------------------------------------- #
# >>> ADJUST THESE IMPORTS to match your DMU package layout <<<
# --------------------------------------------------------------------- #
from . import sem_tools as dmsem
from . import plot_utils as dmp

INKSCAPE_PATH = "inkscape"  # adjust if not on $PATH, e.g. "/usr/bin/inkscape"


# --------------------------------------------------------------------- #
# Core geometry helpers (unchanged from your original code)
# --------------------------------------------------------------------- #

def stitcher_add_rectangles(ax, centers, widths, heights, colours):
    for i, (cx, cy) in enumerate(centers):
        w = widths[i] if isinstance(widths, (list, np.ndarray)) else widths
        h = heights[i] if isinstance(heights, (list, np.ndarray)) else heights
        color = colours[i % len(colours)]
        rect = Rectangle(
            (cx - w / 2, cy - h / 2), w, h,
            edgecolor=color, facecolor="none", linewidth=2,
        )
        ax.add_patch(rect)


def stitcher_enhance_for_matching(img):
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_uint8)
    return img_eq.astype(np.float32) / 255.0


def stitcher_enforce_similarity(H_affine):
    A = H_affine[:, :2]
    t = H_affine[:, 2:]
    U, S, Vt = np.linalg.svd(A)
    scale = S.mean()
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    A_sim = scale * R
    H_sim = np.hstack([A_sim, t])
    return H_sim.astype(np.float32)


# --------------------------------------------------------------------- #
# Adapted stitching pipeline — list + per-image scalebar flags driven
# --------------------------------------------------------------------- #

def run_stitch(
    farpic_paths, farpic_scalebar_flags,
    nearpic_paths, nearpic_scalebar_flags,
    seam_dict=None, scalebar_style=None, txt_style=None,
    inkscape_path=INKSCAPE_PATH, matching_model=None, progress_cb=None,
):
    scalebar_style_defaults = {
        "frame": True, "framepad": [30, 2], "stroke_width": 6, "stroke_style": "line",
        "bar_color": "white", "frame_color": "black", "frame_opacity": 0.6,
        "location": "lower right", "location_padding": [0.03, 0.05],
        "bar_ratio": [1 / 6 * 1.05, 1 / 40],
    }
    txt_style_defaults = {
        "font_family": "Arial", "fontsize": "Auto", "font_weight": "normal",
        "font_style": "normal", "text_decoration": "none", "color": "white",
    }
    seam_dict_defaults = dict(transition=400, gamma=3.0)
    seam_aliases = dict(transition=["transition"], gamma=["gamma"])

    seam_dict = dmsem.kwarg_aliasing(seam_dict or {}, seam_dict_defaults, aliases=seam_aliases)
    scalebar_style = dmsem.kwarg_aliasing(scalebar_style or {}, scalebar_style_defaults, aliases=None)
    txt_style = dmsem.kwarg_aliasing(txt_style or {}, txt_style_defaults, aliases=None)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if matching_model is None:
        matcher = KF.LoFTR(pretrained="outdoor").to(DEVICE)
    else:
        model_name = matching_model.lower()
        if model_name == "resnet18":
            ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
            matcher = torch.hub.load("pytorch/vision:v0.15.2", "resnet18", pretrained=True).to(DEVICE)
        else:
            raise ValueError(f"Unknown matching_model '{matching_model}'")

    IMGD = dict(farpic=[], nearpic=[], filenames=[], sem_metadata_f=[], sem_metadata_c=[])

    n = len(farpic_paths)
    for i in range(n):
        farpic = farpic_paths[i]
        if progress_cb:
            progress_cb(f"Loading {i + 1}/{n}: {os.path.basename(farpic)}")

        img_proc, sem_metadata = dmsem.SEM_Strip_Banner_And_Enhance(farpic, filterdict=dict(expand_range=False))
        img_proc = np.asarray(img_proc)

        if farpic_scalebar_flags[i]:
            sb = dmsem.SEM_Scalebar_Generator(
                img_proc, "temp.svg", scalebar_style=scalebar_style, txt_style=txt_style,
                remove_annotation=False, sem_metadata=sem_metadata,
            )
            img_proc = np.asarray(dmsem.svg_to_pil(sb["svg"], inkscape_path).convert("L"))

        imgn_proc, sem_metadata2, sbimg = None, None, None
        if i < len(nearpic_paths):
            nearpic = nearpic_paths[i]
            try:
                imgn_proc, sem_metadata2 = dmsem.SEM_Strip_Banner_And_Enhance(nearpic, filterdict=dict(expand_range=False))
                if i < len(nearpic_scalebar_flags) and nearpic_scalebar_flags[i]:
                    sb = dmsem.SEM_Scalebar_Generator(
                        imgn_proc, "temp.svg", scalebar_style=scalebar_style, txt_style=txt_style,
                        remove_annotation=False, sem_metadata=sem_metadata2,
                    )
                    sbimg = dmsem.svg_to_pil(sb["svg"], inkscape_path)
                else:
                    sbimg = PIL.Image.fromarray(np.asarray(imgn_proc))
            except Exception:
                imgn_proc, sem_metadata2, sbimg = None, None, None

        IMGD["farpic"].append(img_proc)
        IMGD["nearpic"].append(sbimg)
        IMGD["sem_metadata_f"].append(sem_metadata)
        IMGD["sem_metadata_c"].append(sem_metadata2)
        IMGD["filenames"].append(farpic)

    # ---- accumulate affine transforms across the far-pic chain ----
    H_list = [np.eye(2, 3, dtype=np.float32)]
    for i in range(len(IMGD["farpic"]) - 1):
        img1 = stitcher_enhance_for_matching(IMGD["farpic"][i])
        img2 = stitcher_enhance_for_matching(IMGD["farpic"][i + 1])

        t1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        t2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            out = matcher({"image0": t1, "image1": t2})
            mkpts0 = out["keypoints0"].cpu().numpy()
            mkpts1 = out["keypoints1"].cpu().numpy()

        if progress_cb:
            progress_cb(f"Matching pair {i}-{i + 1}: {len(mkpts0)} matches")

        if len(mkpts0) < 4:
            H_list.append(H_list[-1].copy())
            continue

        H_translation, _ = cv.estimateAffinePartial2D(mkpts1, mkpts0, method=cv.RANSAC)
        if H_translation is None:
            H_list.append(H_list[-1].copy())
            continue

        MIN_MATCHES_FOR_REFINEMENT = 10
        if len(mkpts0) < MIN_MATCHES_FOR_REFINEMENT:
            H_final = H_translation
        else:
            H_affine, mask = cv.estimateAffine2D(mkpts1, mkpts0, method=cv.RANSAC)
            if H_affine is not None:
                H_refined = stitcher_enforce_similarity(H_affine)
                mkpts1_refined = cv.transform(mkpts1.reshape(-1, 1, 2), H_refined)[:, 0, :]
                mean_error_refined = np.linalg.norm(mkpts1_refined - mkpts0, axis=1).mean()

                mkpts1_trans = cv.transform(mkpts1.reshape(-1, 1, 2), H_translation)[:, 0, :]
                mean_error_translation = np.linalg.norm(mkpts1_trans - mkpts0, axis=1).mean()

                if mean_error_refined < mean_error_translation:
                    angle = np.arctan2(H_refined[1, 0], H_refined[0, 0]) * 180 / np.pi
                    H_final = H_refined if abs(angle) < 5 else H_translation
                else:
                    H_final = H_translation
            else:
                H_final = H_translation

        H_last_h = np.vstack([H_list[-1], [0, 0, 1]])
        H_final_h = np.vstack([H_final, [0, 0, 1]])
        H_accum_h = H_last_h @ H_final_h
        H_list.append(H_accum_h[:2])

    # ---- panorama canvas extent ----
    all_corners = []
    for img, H in zip(IMGD["farpic"], H_list):
        h, w = img.shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        warped = cv.transform(corners.reshape(-1, 1, 2), H)
        all_corners.append(warped.reshape(-1, 2))

    all_pts = np.vstack(all_corners)
    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
    W = x_max - x_min
    Hh = y_max - y_min

    # ---- per-image centres, for the coloured frame overlays ----
    centers = []
    for img, H in zip(IMGD["farpic"], H_list):
        h_img, w_img = img.shape
        H_h = np.vstack([H, [0, 0, 1]])
        shift_h = np.eye(3, dtype=np.float32)
        shift_h[0, 2] = -x_min
        shift_h[1, 2] = -y_min
        H_shift_h = shift_h @ H_h
        center_img = np.array([[w_img / 2, h_img / 2]], dtype=np.float32).reshape(-1, 1, 2)
        center_panorama = cv.perspectiveTransform(center_img, H_shift_h)
        centers.append(tuple(center_panorama[0, 0]))

    # ---- pixel-size ratio (far vs. near), from the last pair — see docstring note ----
    try:
        scale1 = IMGD["sem_metadata_f"][-1]["pix_size"]
        scale2 = IMGD["sem_metadata_c"][-1]["pix_size"]
    except (KeyError, TypeError, IndexError):
        scale1 = scale2 = 1.0

    h_img, w_img = IMGD["farpic"][-1].shape
    w_rect = w_img * (scale1 / scale2)
    h_rect = h_img * (scale1 / scale2)

    # ---- cosmetic enhancement pass on far pics ----
    filterdict = dict(brightness=1, contrast=1.1, sharpness=1.1, expand_range=True)
    IMGD["farpic"] = [dmsem.ANY_Image_Enhance(im, **filterdict) for im in IMGD["farpic"]]

    # ---- warp + distance-transform seam blend ----
    tbc = dmp.get_tab20bc(grouping="pairs", output="list")[0::2]
    warped_imgs, weights = [], []
    TRANSITION = seam_dict["transition"]
    gamma = seam_dict["gamma"]

    for img, H in zip(IMGD["farpic"], H_list):
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)

        H_h = np.vstack([H, [0, 0, 1]])
        shift_h = np.eye(3, dtype=np.float32)
        shift_h[0, 2] = -x_min
        shift_h[1, 2] = -y_min
        H_shift = (shift_h @ H_h)[:2]

        warped = cv.warpAffine(img, H_shift, (W, Hh))
        mask = (warped > 0).astype(np.uint8)
        dist = cv.distanceTransform(mask, cv.DIST_L2, 5)
        dist = np.minimum(dist, TRANSITION) / TRANSITION
        dist = dist ** gamma

        warped_imgs.append(warped.astype(np.float32))
        weights.append(dist.astype(np.float32))

    weight_sum = np.sum(weights, axis=0) + 1e-8
    weights = [w / weight_sum for w in weights]

    panorama_avg = np.zeros_like(warped_imgs[0], dtype=np.float32)
    for warped, w in zip(warped_imgs, weights):
        panorama_avg += warped * w

    # ---- layout: panorama + strip of near-pic insets ----
    n_near = len(IMGD["nearpic"])
    if panorama_avg.shape[0] > panorama_avg.shape[1]:
        rcstart = [0, 1]
        nrows = 2
        ncols = max(int(np.ceil(n_near / nrows)) + 1, 2)
        ifrac = 2.5
        Gspec = plt.GridSpec(nrows, ncols, width_ratios=[1 / ifrac] + [(1 - 1 / ifrac) / (ncols - 1)] * (ncols - 1))
        PanGS = Gspec[:, 0]
    else:
        rcstart = [1, 0]
        ncols = 2
        nrows = max(int(np.ceil(n_near / ncols)) + 1, 2)
        Gspec = plt.GridSpec(nrows, ncols, width_ratios=[0.3] * ncols)
        PanGS = Gspec[0, :]

    gslist = [Gspec[r, c] for r in range(rcstart[0], nrows) for c in range(rcstart[1], ncols)]

    fig = plt.figure(figsize=(4 * ncols * 1.5, 4 * nrows))
    ax_pan = fig.add_subplot(PanGS)
    ax_ins = [fig.add_subplot(gs) for gs in gslist]
    fig.subplots_adjust(wspace=0, hspace=0)

    ax_pan.imshow(panorama_avg, cmap="gray")
    ax_pan.set_axis_off()

    for i, img in enumerate(IMGD["nearpic"]):
        if i >= len(ax_ins):
            break
        if img is None:
            ax_ins[i].set_axis_off()
            continue
        ax_ins[i].imshow(img, cmap="gray")
        ax_ins[i].set_axis_off()
        rect = Rectangle((0, 0), img.width, img.height, linewidth=6,
                          edgecolor=tbc[i % len(tbc)], facecolor="none")
        ax_ins[i].add_patch(rect)

    for i in range(len(IMGD["nearpic"]), len(ax_ins)):
        ax_ins[i].set_axis_off()

    stitcher_add_rectangles(ax_pan, centers, w_rect, h_rect, tbc)

    if progress_cb:
        progress_cb("Panorama complete.")

    return fig


# --------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------- #

class FileListItemWidget(QWidget):
    def __init__(self, filepath, scalebar_default=False, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self.label = QLabel(os.path.basename(filepath))
        self.label.setToolTip(filepath)
        self.checkbox = QCheckBox("Scalebar")
        self.checkbox.setChecked(scalebar_default)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.checkbox)

    def is_scalebar_on(self):
        return self.checkbox.isChecked()


class FileListPanel(QWidget):
    """A reorderable, checkbox-annotated file list with Add/Remove/Up/Down."""

    def __init__(self, title, scalebar_default, file_filter="TIFF images (*.tif *.tiff);;All files (*)", parent=None):
        super().__init__(parent)
        self.scalebar_default = scalebar_default
        self.file_filter = file_filter

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{title}</b>"))

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add files…")
        self.remove_btn = QPushButton("Remove")
        self.up_btn = QPushButton("↑ Up")
        self.down_btn = QPushButton("↓ Down")
        for b in (self.add_btn, self.remove_btn, self.up_btn, self.down_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self.add_btn.clicked.connect(self.add_files)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.up_btn.clicked.connect(lambda: self.move_selected(-1))
        self.down_btn.clicked.connect(lambda: self.move_selected(1))

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select images", "", self.file_filter)
        # Note: the file dialog does not guarantee it returns files in the
        # order you clicked them (this varies by OS/desktop theme) — use
        # Up/Down afterwards to fix ordering if it comes out wrong.
        for p in paths:
            self._add_item(p)

    def _add_item(self, path):
        item = QListWidgetItem(self.list_widget)
        widget = FileListItemWidget(path, scalebar_default=self.scalebar_default)
        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)

    def remove_selected(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)

    def move_selected(self, direction):
        row = self.list_widget.currentRow()
        new_row = row + direction
        if row < 0 or new_row < 0 or new_row >= self.list_widget.count():
            return

        widget = self.list_widget.itemWidget(self.list_widget.item(row))
        path = widget.filepath
        checked = widget.checkbox.isChecked()

        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(new_row, item)

        new_widget = FileListItemWidget(path, scalebar_default=checked)
        item.setSizeHint(new_widget.sizeHint())
        self.list_widget.setItemWidget(item, new_widget)
        self.list_widget.setCurrentRow(new_row)

    def get_entries(self):
        paths, flags = [], []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            paths.append(widget.filepath)
            flags.append(widget.is_scalebar_on())
        return paths, flags


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM Panorama Stitcher")
        self.resize(1400, 800)
        self.figure = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_panel = QWidget()
        left_panel.setFixedWidth(420)
        left_layout = QVBoxLayout(left_panel)

        self.far_panel = FileListPanel("Far pics (panorama frames)", scalebar_default=False)
        self.near_panel = FileListPanel("Near pics (detail insets)", scalebar_default=True)
        left_layout.addWidget(self.far_panel)
        left_layout.addWidget(self.near_panel)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Output DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(150)
        self.dpi_spin.setSingleStep(50)
        res_row.addWidget(self.dpi_spin)
        left_layout.addLayout(res_row)

        self.generate_btn = QPushButton("Generate panorama")
        self.save_btn = QPushButton("Save image…")
        self.save_btn.setEnabled(False)
        left_layout.addWidget(self.generate_btn)
        left_layout.addWidget(self.save_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(fig)
        right_layout.addWidget(self.canvas)

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.generate_btn.clicked.connect(self.on_generate)
        self.save_btn.clicked.connect(self.on_save)

    def set_status(self, text):
        self.status_label.setText(text)
        QApplication.processEvents()

    def on_generate(self):
        far_paths, far_flags = self.far_panel.get_entries()
        near_paths, near_flags = self.near_panel.get_entries()

        if len(far_paths) < 1:
            QMessageBox.warning(self, "No images", "Add at least one far pic.")
            return

        self.generate_btn.setEnabled(False)
        self.set_status("Stitching… this can take a while on first run (model download).")

        try:
            fig = run_stitch(far_paths, far_flags, near_paths, near_flags, progress_cb=self.set_status)
        except Exception as e:
            traceback.print_exc()  # prints full stack trace to terminal
        
            QMessageBox.critical(self, "Stitching failed", traceback.format_exc())
        
            self.generate_btn.setEnabled(True)
            self.set_status("Failed.")
            return

        self.figure = fig
        old_canvas = self.canvas
        self.canvas = FigureCanvas(fig)
        layout = old_canvas.parentWidget().layout()
        layout.replaceWidget(old_canvas, self.canvas)
        old_canvas.setParent(None)
        self.canvas.draw()

        self.save_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.set_status("Done.")

    def on_save(self):
        if self.figure is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save panorama", "panorama.png",
            "PNG image (*.png);;TIFF image (*.tif);;PDF (*.pdf)",
        )
        if not path:
            return
        dpi = self.dpi_spin.value()
        self.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        self.set_status(f"Saved to {path}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()