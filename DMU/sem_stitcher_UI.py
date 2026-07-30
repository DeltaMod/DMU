"""
SEM Panorama Stitcher
===============================

A thin PyQt6 GUI wrapped around SEM stitcher pipeline.

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
import io 
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
    QFileDialog, QSpinBox, QSplitter, QMessageBox, QLineEdit, QComboBox,QSizePolicy
)

from PyQt5.QtGui import (QPainter, QColor, QImage)

from PyQt5.QtCore import (Qt, QSettings, QByteArray, QSize, QRectF)

from matplotlib.figure import Figure



from . import sem_tools as dmsem
from . import plot_utils as dmp

INKSCAPE_PATH = "inkscape"  # adjust if not on $PATH, e.g. "/usr/bin/inkscape"


# --------------------------------------------------------------------- #
# Core geometry helpers 
# --------------------------------------------------------------------- #

class AspectRatioCanvas(FigureCanvas):
    def __init__(self, figure):
        super().__init__(figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(42, 42, 42))  # dark grey
        self.setPalette(pal)
        self._display_image = None   # QImage rendered from the figure

    def _render_figure_to_image(self, dpi=100):
        if self.figure is None:
            return None
        buf = io.BytesIO()
        self.figure.savefig(buf, format='png', dpi=dpi)   # no bbox_inches/pad_inches
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.read())
        buf.close()
        return qimg

    def paintEvent(self, event):
        if self.figure is None:
            super().paintEvent(event)
            return

        widget_w = self.width()
        widget_h = self.height()
        if widget_w <= 0 or widget_h <= 0:
            return

        # Render the figure once (at the figure's DPI, fallback 100)
        if self._display_image is None:
            dpi = getattr(self.figure, 'dpi', 100)
            self._display_image = self._render_figure_to_image(dpi)
            if self._display_image is None:
                return

        img = self._display_image
        img_w = img.width()
        img_h = img.height()
        if img_w <= 0 or img_h <= 0:
            return

        aspect = img_w / img_h

        # Compute target size that fits in the widget, preserving aspect
        if widget_w / widget_h > aspect:
            target_w = int(widget_h * aspect)
            target_h = widget_h
        else:
            target_w = widget_w
            target_h = int(widget_w / aspect)

        if target_w <= 0 or target_h <= 0:
            return

        # Scale using nearest‑neighbour (FastTransformation) – no smoothing
        scaled = img.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.FastTransformation)

        # Centre it
        painter = QPainter(self)
        dx = (widget_w - target_w) // 2
        dy = (widget_h - target_h) // 2
        painter.drawImage(dx, dy, scaled)
        painter.end()

    def resizeEvent(self, event):
        # The cached image stays the same; we only need to repaint (scale again)
        super().resizeEvent(event)

    def set_figure(self, fig):
        self.figure = fig
        self._display_image = None   # force re‑render on next paint
        self.update()
        
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
    orientation="Horizontal", dpi=150
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
        if farpic_scalebar_flags[i]:
            sb = dmsem.SEM_Scalebar_Generator(
                img_proc, "temp.svg", scalebar_style=scalebar_style, txt_style=txt_style,
                remove_annotation=False, sem_metadata=sem_metadata,
            )
            img_proc = dmsem.svg_to_pil(sb["svg"], inkscape_path).convert("L")
        img_proc = np.asarray(img_proc)

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

    try:
        scale1 = IMGD["sem_metadata_f"][-1]["pix_size"]
        scale2 = IMGD["sem_metadata_c"][-1]["pix_size"]
    except (KeyError, TypeError, IndexError):
        scale1 = scale2 = 1.0

    h_img, w_img = IMGD["farpic"][-1].shape
    w_rect = w_img * (scale2 / scale1)
    h_rect = h_img * (scale2 / scale1)

    filterdict = dict(brightness=1, contrast=1.1, sharpness=1.1, expand_range=True)
    IMGD["farpic"] = [dmsem.ANY_Image_Enhance(im, **filterdict) for im in IMGD["farpic"]]

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

    n_near = len(IMGD["nearpic"])

    if n_near == 0:
        fig = plt.figure(figsize=(W/dpi, Hh/dpi),dpi=dpi)
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot(111)
        ax.set_adjustable('box')
        ax.margins(0)
        ax.imshow(panorama_avg, cmap='gray', aspect='auto')
        ax.set_xlim(0, W)
        ax.set_ylim(Hh, 0)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        if progress_cb:
            progress_cb("Panorama complete.")
        return fig

    if orientation == "Auto":
        orientation = "Horizontal" if panorama_avg.shape[1] >= panorama_avg.shape[0] else "Vertical"

    nrows_near = 2
    ncols_near = int(np.ceil(n_near / nrows_near))

    valid_near = [img for img in IMGD["nearpic"] if img is not None]
    if not valid_near:
        fig = plt.figure(figsize=(W/dpi, Hh/dpi),dpi=dpi)
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot(111)
        ax.set_adjustable('box')
        ax.margins(0)
        ax.imshow(panorama_avg, cmap='gray', aspect='auto')
        ax.set_xlim(0, W)
        ax.set_ylim(Hh, 0)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return fig

    # ---- Determine uniform cell size for near images ----
    max_near_w = max([img.width for img in valid_near])
    max_near_h = max([img.height for img in valid_near])
    cell_w = max_near_w
    cell_h = max_near_h

    # ---- Compute total figure dimensions in pixels ----
    if orientation == "Horizontal":
        # Panorama on left, near grid on right
        # Total height = sum of row heights (all rows equal cell_h)
        total_height_px = nrows_near * cell_h
        # Panorama width to fit that height while preserving aspect
        panorama_width_px = total_height_px * (W / Hh)
        # Total width = panorama width + sum of column widths (all columns equal cell_w)
        total_width_px = panorama_width_px + ncols_near * cell_w

        # Build GridSpec: nrows_near rows, ncols_near + 1 columns
        width_ratios = [panorama_width_px] + [cell_w] * ncols_near
        height_ratios = [cell_h] * nrows_near

    else:  # Vertical
        # Panorama on top, near grid below
        # Total width = sum of column widths (all columns equal cell_w)
        total_width_px = ncols_near * cell_w
        # Panorama height to fit that width while preserving aspect
        panorama_height_px = total_width_px * (Hh / W)
        # Total height = panorama height + sum of row heights (all rows equal cell_h)
        total_height_px = panorama_height_px + nrows_near * cell_h

        # Build GridSpec: nrows_near + 1 rows, ncols_near columns
        width_ratios = [cell_w] * ncols_near
        height_ratios = [panorama_height_px] + [cell_h] * nrows_near

    # ---- Create figure ----
    fig = plt.figure(figsize=(total_width_px/dpi, total_height_px/dpi),dpi=dpi)
    fig.patch.set_facecolor('black')

    # ---- Create GridSpec and axes ----
    if orientation == "Horizontal":
        gs = plt.GridSpec(nrows_near, ncols_near + 1,
                  width_ratios=width_ratios,
                  height_ratios=height_ratios,
                  left=0, right=1, bottom=0, top=1,   # <-- added
                  wspace=0, hspace=0)

        ax_pan = fig.add_subplot(gs[:, 0])  # all rows, first column
        ax_pan.set_adjustable('box')
        ax_pan.margins(0)
        ax_pan.imshow(panorama_avg, cmap='gray', aspect='auto')
        ax_pan.set_xlim(0, W)
        ax_pan.set_ylim(Hh, 0)
        ax_pan.set_axis_off()
        ax_pan.set_facecolor('black')

        ax_ins = []
        for r in range(nrows_near):
            for c in range(ncols_near):
                ax = fig.add_subplot(gs[r, c+1])
                ax.set_adjustable('box')
                ax.margins(0)
                ax.set_axis_off()
                ax.set_facecolor('black')
                ax_ins.append(ax)

    else:  # Vertical
        gs = plt.GridSpec(nrows_near + 1, ncols_near,
                  width_ratios=width_ratios,
                  height_ratios=height_ratios,
                  left=0, right=1, bottom=0, top=1,   # <-- added
                  wspace=0, hspace=0)

        ax_pan = fig.add_subplot(gs[0, :])  # first row, all columns
        ax_pan.set_adjustable('box')
        ax_pan.margins(0)
        ax_pan.imshow(panorama_avg, cmap='gray', aspect='auto')
        ax_pan.set_xlim(0, W)
        ax_pan.set_ylim(Hh, 0)
        ax_pan.set_axis_off()
        ax_pan.set_facecolor('black')

        ax_ins = []
        for r in range(nrows_near):
            for c in range(ncols_near):
                ax = fig.add_subplot(gs[r+1, c])
                ax.set_adjustable('box')
                ax.margins(0)
                ax.set_axis_off()
                ax.set_facecolor('black')
                ax_ins.append(ax)

    # ---- Place near images ----
    for i, img in enumerate(IMGD["nearpic"]):
        if i >= len(ax_ins):
            break
        if img is None:
            ax_ins[i].set_axis_off()
            ax_ins[i].set_facecolor('black')
            continue
        w, h = img.width, img.height
        ax_ins[i].imshow(img, cmap='gray', aspect='auto')
        ax_ins[i].set_xlim(0, w)
        ax_ins[i].set_ylim(h, 0)
        ax_ins[i].set_axis_off()
        ax_ins[i].set_facecolor('black')
        rect = Rectangle((0, 0), w, h, linewidth=6,
                         edgecolor=tbc[i % len(tbc)], facecolor='none')
        ax_ins[i].add_patch(rect)

    # Fill remaining axes (if any)
    for i in range(n_near, len(ax_ins)):
        ax_ins[i].set_axis_off()
        ax_ins[i].set_facecolor('black')

    # ---- Overlay rectangles on the panorama ----
    stitcher_add_rectangles(ax_pan, centers, w_rect, h_rect, tbc)
    for patch in ax_pan.patches:
        patch.set_zorder(10)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

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
    def __init__(self, title, scalebar_default, file_filter="images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)", parent=None):
        super().__init__(parent)
        self.scalebar_default = scalebar_default
        self.file_filter = file_filter
        self.root_dir = ""

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

    def set_root_directory(self, path):
        self.root_dir = path

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select images", self.root_dir, self.file_filter
        )
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

        self.settings = QSettings("DMU", "SEMStitcher")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left panel (fixed width)
        left_panel = QWidget()
        left_panel.setFixedWidth(420)
        left_layout = QVBoxLayout(left_panel)

        # Root directory
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Root directory:"))
        self.root_line = QLineEdit()
        self.root_line.setPlaceholderText("Enter path or leave empty for home")
        root_row.addWidget(self.root_line)
        left_layout.addLayout(root_row)

        # Layout orientation
        orient_row = QHBoxLayout()
        orient_row.addWidget(QLabel("Layout:"))
        self.orient_combo = QComboBox()
        self.orient_combo.addItems(["Horizontal", "Vertical", "Auto"])
        self.orient_combo.setCurrentText("Horizontal")
        orient_row.addWidget(self.orient_combo)
        left_layout.addLayout(orient_row)

        self.far_panel = FileListPanel("Far pics (panorama frames)", scalebar_default=False)
        self.near_panel = FileListPanel("Near pics (detail insets)", scalebar_default=True)
        left_layout.addWidget(self.far_panel)
        left_layout.addWidget(self.near_panel)

        self.root_line.textChanged.connect(self.update_root_dirs)

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

        # Right panel – will hold the aspect-ratio canvas
        self.right_container = QWidget()
        self.right_container.setStyleSheet("background-color: #2a2a2a;")
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = None
        self.placeholder = QLabel("No panorama yet")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.placeholder)

        # Splitter
        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(self.right_container)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.generate_btn.clicked.connect(self.on_generate)
        self.save_btn.clicked.connect(self.on_save)

        # ----- Restore saved state -----
        self.restore_settings(splitter)

    def restore_settings(self, splitter):
        # Restore window geometry
        geometry = self.settings.value("geometry", QByteArray())
        if not geometry.isEmpty():
            self.restoreGeometry(geometry)

        # Restore splitter state
        splitter_state = self.settings.value("splitterState", QByteArray())
        if not splitter_state.isEmpty():
            splitter.restoreState(splitter_state)

        # Restore root directory
        root_dir = self.settings.value("rootDirectory", "")
        if root_dir:
            self.root_line.setText(root_dir)

        # Restore DPI
        dpi = self.settings.value("dpi", 150, type=int)
        self.dpi_spin.setValue(dpi)

        # Restore orientation
        orient_index = self.settings.value("orientationIndex", 0, type=int)
        if 0 <= orient_index < self.orient_combo.count():
            self.orient_combo.setCurrentIndex(orient_index)

    def closeEvent(self, event):
        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())

        # Save splitter state (find the splitter in the layout)
        splitter = self.findChild(QSplitter)
        if splitter:
            self.settings.setValue("splitterState", splitter.saveState())

        # Save root directory
        self.settings.setValue("rootDirectory", self.root_line.text())

        # Save DPI
        self.settings.setValue("dpi", self.dpi_spin.value())

        # Save orientation index
        self.settings.setValue("orientationIndex", self.orient_combo.currentIndex())

        event.accept()

    def update_root_dirs(self, text):
        self.far_panel.set_root_directory(text)
        self.near_panel.set_root_directory(text)

    def set_status(self, text):
        self.status_label.setText(text)
        QApplication.processEvents()

    def on_generate(self):
        far_paths, far_flags = self.far_panel.get_entries()
        near_paths, near_flags = self.near_panel.get_entries()
    
        if len(far_paths) < 1:
            QMessageBox.warning(self, "No images", "Add at least one far pic.")
            return
    
        orientation = self.orient_combo.currentText()
    
        # Save original window size before doing anything
        original_width = self.width()
        original_height = self.height()
    
        self.generate_btn.setEnabled(False)
        self.set_status("Stitching… this can take a while on first run (model download).")
    
        try:
            dpi = self.dpi_spin.value()
            fig = run_stitch(
                far_paths, far_flags,
                near_paths, near_flags,
                progress_cb=self.set_status,
                orientation=orientation,
                dpi=dpi
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Stitching failed", traceback.format_exc())
            self.generate_btn.setEnabled(True)
            self.set_status("Failed.")
            return
    
        # Compute the figure's native pixel size
        fig_width_px = int(fig.get_size_inches()[0] * fig.dpi)
        fig_height_px = int(fig.get_size_inches()[1] * fig.dpi)
    
        # Remove placeholder if present
        if self.placeholder is not None:
            self.right_layout.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None
    
        if self.canvas is not None:
            self.right_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
    
        # Create and add the canvas
        self.canvas = AspectRatioCanvas(fig)
        self.right_layout.addWidget(self.canvas)
        self.canvas.show()
    
        # --- Temporarily resize window to match figure size ---
        splitter = self.findChild(QSplitter)
        if splitter is not None:
            # Left panel is fixed at 420 px; set right panel width to figure width
            splitter.setSizes([420, fig_width_px])
    
        # Compute how much we need to enlarge the window to fit the figure
        current_right_width = self.right_container.width()
        current_right_height = self.right_container.height()
        delta_width = fig_width_px - current_right_width
        delta_height = fig_height_px - current_right_height
        self.resize(self.width() + delta_width, self.height() + delta_height)
    
        # --- Immediately restore the original window size ---
        self.resize(original_width, original_height)
    
        # Set a dark grey background for the right container (to avoid artifacts)
        self.right_container.setStyleSheet("background-color: #2a2a2a;")
    
        self.save_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.set_status("Done.")

    def on_save(self):
        if self.canvas is None or self.canvas.figure is None:
            return
    
        far_paths, _ = self.far_panel.get_entries()
        if far_paths:
            base = os.path.splitext(os.path.basename(far_paths[0]))[0]
            default_name = f"{base}_panorama.png"
        else:
            default_name = "panorama.png"
    
        root_dir = self.root_line.text().strip()
        if root_dir:
            default_path = os.path.join(root_dir, default_name)
        else:
            default_path = default_name
    
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save panorama",
            default_path,
            "PNG image (*.png);;TIFF image (*.tif);;PDF (*.pdf)",
        )
        if not path:
            return
    
        dpi = self.dpi_spin.value()
        self.canvas.figure.savefig(path, dpi=dpi)
        self.set_status(f"Saved to {path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()