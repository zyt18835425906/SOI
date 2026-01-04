import os
import time
import traceback
from io import BytesIO

import numpy as np
import pandas as pd
import laspy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QTabWidget,
    QToolBar, QDoubleSpinBox, QMessageBox
)
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, Slot

# ===== CE_TO_CSC_plot（tree） =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


# ==========================
# 1. Canopy Entropy Calculation Core Class (KDE Based on All Points) #Updated September 2025
# ==========================

class CanopyEntropyCalculator:
    """
    Calculate canopy entropy using three-plane (XY / XZ / YZ) KDE + numerical integration:
    1. Point cloud projections onto XYZ planes；
    2. Estimate continuous probability density using Gaussian KDE f(x,y)；
    3. Sampling on a regular grid, approximating cell mass p_ij ≈ f(x_ij,y_ij)*ΔA；
    4. entropy H = -Σ p_ij ln p_ij。
    """

    def __init__(self, bandwidth_factor, pixel_size,
                 max_grid_points=300000,
                 logger=None):
        """
        bandwidth_factor: KDE Bandwidth scaling factor (1.0 = Scott Default，<1 more sensitive，>1 Smoother)
        pixel_size: Initial pixel size (m)，Control grid resolution
        max_grid_points: Maximum number of grid points per plane
        logger: Log Function
        """
        self.bandwidth_factor = float(bandwidth_factor)
        self.pixel_size = float(pixel_size)
        self.max_grid_points = int(max_grid_points)
        self.logger = logger

    def log(self, msg: str):
        if self.logger is not None:
            self.logger(msg)

    # ---------- LAS ----------

    def read_las_file(self, file_path):
        """Read LAS, return (N,3) float32"""
        self.log(f"Reading LAS file: {file_path}")
        las = laspy.read(file_path)

        x = np.asarray(las.x, dtype=np.float32)
        y = np.asarray(las.y, dtype=np.float32)
        z = np.asarray(las.z, dtype=np.float32)
        points = np.vstack((x, y, z)).T

        # Remove NaN
        if np.isnan(points).any():
            mask = ~np.isnan(points).any(axis=1)
            points = points[mask]

        self.log(f"Read to {points.shape[0]} point，shape {points.shape}")
        return points

    # ---------- 3-Planar Entropy + Total Entropy  ----------

    def calculate_canopy_entropy(self, points):
        """
        Calculate the four entropy indices for plot：
        CE_XY / CE_XZ / CE_YZ / CE_Total
        """
        self.log("Begin calculating the canopy entropy index for plot...")

        if points.ndim != 2 or points.shape[1] != 3:
            points = points.reshape(-1, 3)

        xy = points[:, [0, 1]]  # XY
        xz = points[:, [0, 2]]  # XZ
        yz = points[:, [1, 2]]  # YZ

        ce_xy = self._plane_entropy_kde(xy, self.pixel_size, "XY")
        ce_xz = self._plane_entropy_kde(xz, self.pixel_size, "XZ")
        ce_yz = self._plane_entropy_kde(yz, self.pixel_size, "YZ")

        ce_total = float(np.sqrt(ce_xy ** 2 + ce_xz ** 2 + ce_yz ** 2))
        return ce_xy, ce_xz, ce_yz, ce_total

    # ---------- plane KDE + Shannon Entropy ----------

    def _plane_entropy_kde(self, pts2d, pixel_size, plane_name):
        """
        XY, XZ, YZ plane：
        - KDE fitting；
        - Adaptively adjust pixel size and control the number of grid points；
        - If KDE is too extreme (unstable values), revert to the histogram alternative。
        """
        self.log(f"  calculate {plane_name} Entropy  (KDE，points)...")

        n_pts = pts2d.shape[0]
        if n_pts == 0:
            self.log(f"  {plane_name} plane no points, entropy set to 0")
            return 0.0

        x = pts2d[:, 0].astype(np.float64)
        y = pts2d[:, 1].astype(np.float64)

        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        if x_max == x_min or y_max == y_min:
            self.log(f"  {plane_name} entropy set to 0。")
            return 0.0

        dx = x_max - x_min
        dy = y_max - y_min
        pad = 0.05 * max(dx, dy)
        x_min -= pad
        x_max += pad
        y_min -= pad
        y_max += pad

        cur_pixel = float(pixel_size)
        while True:
            x_grid = np.arange(x_min, x_max + cur_pixel, cur_pixel)
            y_grid = np.arange(y_min, y_max + cur_pixel, cur_pixel)
            nx = len(x_grid)
            ny = len(y_grid)
            grid_points = nx * ny
            if grid_points <= self.max_grid_points:
                break
            cur_pixel *= 1.5
            self.log(
                f"    {plane_name} The pixels are too fine (Grid point {grid_points:,})，"
                f"Auto zoom to {cur_pixel:.4f} m To control memory and computational load"
            )

        self.log(
            f"    {plane_name} Planar Grid: {nx} x {ny} = {nx*ny:,} point, Pixel = {cur_pixel:.4f} m，"
            f"KDE Based on {n_pts:,} point。"
        )

        # KDE 
        data = pts2d.T  # shape: (2, n)

        # bandwidth：factor * Scott Default
        try:
            if self.bandwidth_factor <= 0:
                kde = gaussian_kde(data)  # Scott Default
                self.log(f"    {plane_name}  gaussian_kde Default bandwidth (Scott)。")
            else:
                kde = gaussian_kde(data, bw_method=self.bandwidth_factor)
                self.log(
                    f"    {plane_name} bandwidth scaling factor bw_factor={self.bandwidth_factor:.3f}。"
                )
        except Exception as e:
            self.log(f"    {plane_name} KDE Fitting failed: {str(e)}，Switch to using histogram approximation for entropy")
            return self._plane_entropy_histogram(pts2d, cur_pixel, plane_name)

        # Evaluate on a grid of KDE
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])  # (2, nx*ny)

        try:
            density = kde(grid_coords)  # Continuous density f(x,y)
        except Exception as e:
            self.log(f"    {plane_name} KDE Fitting failed: {str(e)}，Switch to using histogram approximation for entropy")
            return self._plane_entropy_histogram(pts2d, cur_pixel, plane_name)

        density_grid = density.reshape(yy.shape)

        # Numerical Integration: Continuous Density → Probability Mass
        cell_area = cur_pixel * cur_pixel
        mass_grid = density_grid * cell_area   # Approximate Unit Probability Mass
        total_mass = mass_grid.sum()

        if total_mass <= 0 or not np.isfinite(total_mass):
            self.log(f"    {plane_name} KDE The integral is non-positive or non-finite, and entropy set to 0")
            return 0.0

        p = mass_grid / total_mass
        p = p[p > 0]

        if p.size == 0:
            self.log(f"    {plane_name} All probabilities have zero mass, and entropy set to 0")
            return 0.0

        entropy = -float(np.sum(p * np.log(p)))
        self.log(f"    {plane_name} entropy = {entropy:.6f}")
        return entropy

    # ---------- KDE Alternative Histogram for Failure Cases ----------

    def _plane_entropy_histogram(self, pts2d, pixel_size, plane_name):
        """Alternative: 2D Histogram Shannon Entropy"""
        self.log(f"    {plane_name} Approximating Entropy Using the 2D Histogram...")

        x = pts2d[:, 0].astype(np.float64)
        y = pts2d[:, 1].astype(np.float64)

        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        if x_max == x_min or y_max == y_min:
            return 0.0

        dx = x_max - x_min
        dy = y_max - y_min
        pad = 0.05 * max(dx, dy)
        x_min -= pad
        x_max += pad
        y_min -= pad
        y_max += pad

        cur_pixel = float(pixel_size)
        while True:
            x_bins = max(1, int(np.ceil((x_max - x_min) / cur_pixel)))
            y_bins = max(1, int(np.ceil((y_max - y_min) / cur_pixel)))
            grid_points = x_bins * y_bins
            if grid_points <= self.max_grid_points:
                break
            cur_pixel *= 1.5
            self.log(
                f"      {plane_name} histogram pixels too fine (Grid point {grid_points:,})，"
                f"automatic pixel zoom to {cur_pixel:.4f} m。"
            )

        hist, _, _ = np.histogram2d(
            x, y,
            bins=[x_bins, y_bins],
            range=[[x_min, x_max], [y_min, y_max]]
        )

        total = hist.sum()
        if total <= 0:
            return 0.0

        p = hist / total
        p = p[p > 0]
        if p.size == 0:
            return 0.0

        entropy = -float(np.sum(p * np.log(p)))
        self.log(f"      {plane_name} Histogram Entropy = {entropy:.6f}")
        return entropy


# ==========================
# 2. Calculate canopy entropy plot 
# ==========================

class EntropyWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(float)
    status_signal = Signal(str)
    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, folder_path, bandwidth_factor, pixel_size,
                 parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.bandwidth_factor = bandwidth_factor
        self.pixel_size = pixel_size

    def run(self):
        try:
            self.status_signal.emit("Scanning folder...")
            las_files = [f for f in os.listdir(self.folder_path)
                         if f.lower().endswith('.las')]
            total_files = len(las_files)

            if total_files == 0:
                self.log_signal.emit("The LAS file was not found")
                self.status_signal.emit("Completed - File not found")
                self.finished_signal.emit(pd.DataFrame())
                return

            self.log_signal.emit(f"Found {total_files} LAS file")
            self.status_signal.emit("Calculating canopy entropy index...")

            results = []
            for idx, las_file in enumerate(las_files, start=1):
                file_path = os.path.join(self.folder_path, las_file)
                file_id = os.path.splitext(las_file)[0]

                try:
                    self.log_signal.emit(f"Begin processing the file: {las_file}")

                    calculator = CanopyEntropyCalculator(
                        bandwidth_factor=self.bandwidth_factor,
                        pixel_size=self.pixel_size,
                        max_grid_points=300000,
                        logger=self.log_signal.emit
                    )

                    points = calculator.read_las_file(file_path)
                    ce_xy, ce_xz, ce_yz, ce_total = calculator.calculate_canopy_entropy(points)

                    results.append({
                        'ID': file_id,
                        'Points_Count': int(points.shape[0]),
                        'CE_XY': ce_xy,
                        'CE_XZ': ce_xz,
                        'CE_YZ': ce_yz,
                        'CE_Total': ce_total
                    })

                    self.log_signal.emit(
                        f"file {file_id} Completed: CE_XY={ce_xy:.6f}, CE_XZ={ce_xz:.6f}, "
                        f"CE_YZ={ce_yz:.6f}, CE_Total={ce_total:.6f}"
                    )

                except Exception as e:
                    err_msg = f"processing file {file_id} Error occurred: {str(e)}\n{traceback.format_exc()}"
                    self.log_signal.emit(err_msg)
                    results.append({
                        'ID': file_id,
                        'Points_Count': -1,
                        'CE_XY': -1,
                        'CE_XZ': -1,
                        'CE_YZ': -1,
                        'CE_Total': -1
                    })

                self.progress_signal.emit(100.0 * idx / total_files)

            df = pd.DataFrame(results)

            # Save CSV
            csv_path = os.path.join(self.folder_path, "canopy_entropy_KDE_results.csv")
            try:
                os.makedirs(self.folder_path, exist_ok=True)

                if os.path.exists(csv_path):
                    try:
                        with open(csv_path, 'a'):
                            pass
                    except IOError:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        csv_path = os.path.join(
                            self.folder_path,
                            f"canopy_entropy_KDE_results_{timestamp}.csv"
                        )
                        self.log_signal.emit(f"CSV file is occupied: {csv_path}")

                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                self.log_signal.emit(f"Results saved to: {csv_path}")

            except PermissionError:
                user_docs = os.path.join(os.path.expanduser("~"), "Documents")
                csv_path = os.path.join(user_docs, "canopy_entropy_KDE_results.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                self.log_signal.emit(
                    f"Unable to write to the target directory. Results have been saved to the document directory: {csv_path}"
                )
            except Exception as e:
                self.log_signal.emit(f"Error occurred while saving CSV: {str(e)}")

            self.status_signal.emit("Processing complete")
            self.finished_signal.emit(df)

        except Exception as e:
            err_msg = f"Errors occurred during processing: {str(e)}\n{traceback.format_exc()}"
            self.error_signal.emit(err_msg)


# ==========================
# 3. Main Window: Parameter Settings + Visualization
# ==========================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Canopy Entropy Index")
        self.resize(1100, 650)

        self.results_df = None
        self.worker = None

        self._init_ui()

    # ---------- UI ----------

    def _init_ui(self):
        self._create_toolbar()

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self._setup_param_tab()
        self._setup_visual_tab()

    def _create_toolbar(self):
        toolbar = QToolBar("Toolbar", self)
        self.addToolBar(toolbar)

        open_action = QAction("Select Folder", self)
        open_action.triggered.connect(self.choose_folder)
        toolbar.addAction(open_action)

        start_action = QAction("calculating canopy entropy", self)
        start_action.triggered.connect(self.on_start_clicked)
        toolbar.addAction(start_action)

    # ---------- Tab1：Parameters + Logs ----------

    def _setup_param_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # Input
        input_group = QGroupBox("Input data")
        input_layout = QHBoxLayout(input_group)
        self.folder_edit = QLineEdit()
        browse_btn = QPushButton("Scan...")
        browse_btn.clicked.connect(self.choose_folder)
        input_layout.addWidget(QLabel("LAS folder："))
        input_layout.addWidget(self.folder_edit)
        input_layout.addWidget(browse_btn)

        # Parameter Settings
        param_group = QGroupBox("Canopy Entropy KDE Parameter Settings")
        param_layout = QGridLayout(param_group)

        # 1) KDE Bandwidth scaling factor
        self.bandwidth_spin = QDoubleSpinBox()
        self.bandwidth_spin.setRange(0.1, 3.0)
        self.bandwidth_spin.setSingleStep(0.1)
        self.bandwidth_spin.setDecimals(2)
        self.bandwidth_spin.setValue(1.0)

        param_layout.addWidget(
            QLabel("KDE Bandwidth scaling factor (Dimensionless)："), 0, 0, Qt.AlignLeft
        )
        param_layout.addWidget(self.bandwidth_spin, 0, 1)

        bw_hint = QLabel(
            "Note: This controls the bandwidth factor of the Gaussian KDE; \n"
            "1.0 = Scott Scott default bandwidth；\n"
            "< 1：Nuclear functions are narrower,higher entropy values；\n"
            "> 1：kernel function becomes broader,entropy value decreases slightly。\n"
           
        )
        bw_hint.setWordWrap(True)
        param_layout.addWidget(bw_hint, 0, 2)

        # 2) Pixel size
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.01, 1.0)
        self.pixel_size_spin.setSingleStep(0.01)
        self.pixel_size_spin.setDecimals(3)
        self.pixel_size_spin.setValue(0.1)

        param_layout.addWidget(
            QLabel("KDE Grid pixel size (m)："), 1, 0, Qt.AlignLeft
        )
        param_layout.addWidget(self.pixel_size_spin, 1, 1)

        px_hint = QLabel(
            "Note: Controls the mesh resolution for the three projection views (XY/XZ/YZ)。\n"

        px_hint.setWordWrap(True)
        param_layout.addWidget(px_hint, 1, 2)

        # Status + Progress Bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)

        # Button
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Calculating canopy entropy")
        self.start_btn.clicked.connect(self.on_start_clicked)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addStretch()

        # LOG
        log_group = QGroupBox("LOG")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)

        # Combination Layout
        main_layout.addWidget(input_group)
        main_layout.addWidget(param_group)
        main_layout.addLayout(status_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(log_group)

        self.tab_widget.addTab(tab, "Parameter Settings ")

    # ---------- Tab2：Result Visualizatio ----------

    def _setup_visual_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        top_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Visualization")
        self.refresh_btn.clicked.connect(self.refresh_visualization)
        self.refresh_btn.setEnabled(False)
        top_layout.addWidget(self.refresh_btn)
        top_layout.addStretch()

        canvas_layout = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setScaledContents(True)

        self.corr_label = QLabel()
        self.corr_label.setAlignment(Qt.AlignCenter)
        self.corr_label.setScaledContents(True)

        canvas_layout.addWidget(self.summary_label)
        canvas_layout.addWidget(self.corr_label)

        self.summary_label.setText("statistical information")
        self.corr_label.setText("entropy correlation matrix")

        main_layout.addLayout(top_layout)
        main_layout.addLayout(canvas_layout)

        self.tab_widget.addTab(tab, "Result Visualization")

    # ---------- Tool function ----------

    def append_log(self, text: str):
        self.log_edit.append(text)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select the LAS folder")
        if folder:
            self.folder_edit.setText(folder)

    # ---------- startup calculation----------

    def on_start_clicked(self):
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.critical(self, "Error", "Please select a valid LAS folder")
            return

        bandwidth_factor = self.bandwidth_spin.value()
        pixel_size = self.pixel_size_spin.value()

        self.log_edit.clear()
        self.status_label.setText("Calculation is starting...")
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)

        self.worker = EntropyWorker(
            folder_path=folder,
            bandwidth_factor=bandwidth_factor,
            pixel_size=pixel_size
        )
        self.worker.log_signal.connect(self.on_worker_log)
        self.worker.progress_signal.connect(self.on_worker_progress)
        self.worker.status_signal.connect(self.on_worker_status)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.error_signal.connect(self.on_worker_error)

        self.worker.start()

    # ---------- worker ----------

    @Slot(str)
    def on_worker_log(self, text: str):
        self.append_log(text)

    @Slot(float)
    def on_worker_progress(self, value: float):
        self.progress_bar.setValue(int(value))

    @Slot(str)
    def on_worker_status(self, text: str):
        self.status_label.setText(text)

    @Slot(object)
    def on_worker_finished(self, df: object):
        self.start_btn.setEnabled(True)
        self.worker = None

        if isinstance(df, pd.DataFrame) and not df.empty:
            self.results_df = df
            self.status_label.setText("Processing complete")
            QMessageBox.information(self, "complete", "All LAS files have been processed")
            self.refresh_btn.setEnabled(True)
            self.tab_widget.setCurrentIndex(1)
            self.refresh_visualization()
        else:
            self.status_label.setText("Completed (No valid results)")

    @Slot(str)
    def on_worker_error(self, err_msg: str):
        self.start_btn.setEnabled(True)
        self.worker = None
        self.append_log(err_msg)
        QMessageBox.critical(self, "Error", "Error occurred during processing")

    # ---------- Visualization ----------

    def _fig_to_pixmap(self, fig):
        """matplotlib Figure -> QPixmap"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), 'PNG')
        return pixmap

    def refresh_visualization(self):
        if self.results_df is None or self.results_df.empty:
            return

        df = self.results_df.copy()


        df_valid = df.replace([-1], np.nan).dropna()

   
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=100)

        # 1. CE_Total Histogram
        if not df_valid.empty:
            axes[0, 0].hist(df_valid['CE_Total'], bins=20,
                            alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Canopy Entropy CE_Total Distribution')
            axes[0, 0].set_xlabel('CE_Total')
            axes[0, 0].set_ylabel('Frequency')
        else:
            axes[0, 0].text(0.5, 0.5, "No valid data", ha='center', va='center')
            axes[0, 0].set_axis_off()

        # 2. Counting vs CE_Total
        if not df_valid.empty:
            axes[0, 1].scatter(df_valid['Points_Count'], df_valid['CE_Total'],
                               alpha=0.7, color='lightcoral', s=20)
            axes[0, 1].set_title('Point Count VS Canopy Entropy')
            axes[0, 1].set_xlabel('Points_Count')
            axes[0, 1].set_ylabel('CE_Total')
        else:
            axes[0, 1].text(0.5, 0.5, "No valid data", ha='center', va='center')
            axes[0, 1].set_axis_off()

        # 3. Mean Entropy Planes
        if not df_valid.empty:
            ce_means = [
                df_valid['CE_XY'].mean(),
                df_valid['CE_XZ'].mean(),
                df_valid['CE_YZ'].mean()
            ]
            axes[1, 0].bar(['CE_XY', 'CE_XZ', 'CE_YZ'], ce_means,
                           color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
            axes[1, 0].set_title('Mean canopy entropy')
            axes[1, 0].set_ylabel('Mean canopy entropy')
        else:
            axes[1, 0].text(0.5, 0.5, "No valid data", ha='center', va='center')
            axes[1, 0].set_axis_off()

        # 4. Text Statistics
        axes[1, 1].axis('off')
        if not df_valid.empty:
            stats_text = f"""
Number of plots: {len(df_valid)}
Mean Canopy Entropy CE_Total: {df_valid['CE_Total'].mean():.6f}
STD: {df_valid['CE_Total'].std():.6f}
MAX: {df_valid['CE_Total'].max():.6f}
MIN: {df_valid['CE_Total'].min():.6f}

Mean CE_XY: {df_valid['CE_XY'].mean():.6f}
Mean CE_XZ: {df_valid['CE_XZ'].mean():.6f}
Mean CE_YZ: {df_valid['CE_YZ'].mean():.6f}
"""
            axes[1, 1].text(0.02, 0.98, stats_text,
                            fontsize=10, va='top', ha='left')
        else:
            axes[1, 1].text(0.5, 0.5, "No valid data", ha='center', va='center')

        fig.tight_layout()
        pixmap_summary = self._fig_to_pixmap(fig)
        self.summary_label.setPixmap(pixmap_summary)

        # --- Correlation Matrix：CE VS point ---
        metrics = ['Points_Count', 'CE_XY', 'CE_XZ', 'CE_YZ', 'CE_Total']
        corr_df = df_valid[metrics]
        corr = corr_df.corr()

        fig2, ax = plt.subplots(figsize=(5, 4), dpi=100)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(metrics)

        for i in range(len(metrics)):
            for j in range(len(metrics)):
                val = corr.iloc[i, j]
                txt = "" if np.isnan(val) else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", color="black")

        ax.set_title("Correlation Matrix")
        fig2.colorbar(im, ax=ax)
        fig2.tight_layout()

        pixmap_corr = self._fig_to_pixmap(fig2)
        self.corr_label.setPixmap(pixmap_corr)


# ==========================
# 4. output
# ==========================

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
