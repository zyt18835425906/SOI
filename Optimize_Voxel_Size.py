# -*- coding: utf-8 -*-
"""
Voxel size optimization by maximizing correlation (Pearson r)
Workflow:
1) Select LAS folder (crown point clouds), and select CSV with TreeID & CSVolume
2) For voxel sizes from 0.20 down to 0.02 step 0.02:
   - voxelize each LAS -> voxel volume = n_occupied_voxels * s^3
   - write voxel volume into df column named by voxel size
   - compute Pearson r between CSVolume and that voxel volume column
3) Best voxel size = argmax(r)
4) Output:
   (a) original CSV + voxel volume columns
   (b) voxel_size vs r (and p-value, n)
"""

import os
import time
import traceback
import numpy as np
import pandas as pd
import laspy

from scipy.stats import pearsonr

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QToolBar,
    QLabel, QLineEdit, QPushButton, QGridLayout, QMessageBox,
    QDoubleSpinBox, QSpinBox, QTextEdit, QProgressBar
)


# -------------------------
# Core functions
# -------------------------
def read_las_xyz(las_path: str) -> np.ndarray:
    las = laspy.read(las_path)
    pts = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def voxel_volume(points: np.ndarray, voxel_size: float) -> float:
    """
    Occupancy voxelization volume:
    - Use per-tree min(x,y,z) as origin (stable indices)
    - Count unique voxel indices
    - Volume = nvox * s^3
    """
    if points.size == 0:
        return np.nan
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    mn = points.min(axis=0)
    idx = np.floor((points - mn) / voxel_size).astype(np.int32)
    idx_unique = np.unique(idx, axis=0)
    nvox = idx_unique.shape[0]
    return float(nvox) * (voxel_size ** 3)


def build_sizes(start: float, end: float, step: float):
    if start < end:
        start, end = end, start
    sizes = []
    s = start
    # include end exactly (avoid float drift)
    while s >= end - 1e-12:
        sizes.append(round(float(s), 4))
        s -= step
    # force include end if missed
    if abs(sizes[-1] - end) > 1e-9:
        sizes.append(round(float(end), 4))
    # unique & descending
    return sorted(set(sizes), reverse=True)


def safe_column_name(s: float) -> str:
    return f"VoxVol_{s:.2f}m"


# -------------------------
# Worker (threaded)
# -------------------------
class Worker(QObject):
    log = Signal(str)
    progress = Signal(int, int)   # done, total
    finished = Signal()
    done_outputs = Signal(str, str, float)  # out_csv1, out_csv2, best_size

    def __init__(self, las_dir, in_csv, out_dir, id_col, vol_col,
                 size_start, size_end, size_step):
        super().__init__()
        self.las_dir = las_dir
        self.in_csv = in_csv
        self.out_dir = out_dir
        self.id_col = id_col
        self.vol_col = vol_col
        self.size_start = float(size_start)
        self.size_end = float(size_end)
        self.size_step = float(size_step)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            t_all = time.time()

            # Read CSV
            df = pd.read_csv(self.in_csv)
            if self.id_col not in df.columns:
                raise ValueError(f"CSVMissing column：{self.id_col}")
            if self.vol_col not in df.columns:
                raise ValueError(f"CSVMissing column：{self.vol_col}")

            # Normalize TreeID as string for matching
            df[self.id_col] = df[self.id_col].astype(str)

            # Check duplicates
            if df[self.id_col].duplicated().any():
                dup = df.loc[df[self.id_col].duplicated(), self.id_col].tolist()[:10]
                raise ValueError(f"TreeIDduplicates exist")

            id_to_row = {tid: i for i, tid in enumerate(df[self.id_col].tolist())}

            # LAS files
            las_files = [f for f in os.listdir(self.las_dir) if f.lower().endswith(".las")]
            las_files.sort()
            if len(las_files) == 0:
                raise ValueError("No LAS files were found in the LAS folder.")

            self.log.emit(f"ReadCSV：{len(df)}Rows；ReadLAS：{len(las_files)}file")
            self.log.emit(f"TreeID列：{self.id_col} | Actual Volume Column：{self.vol_col}")

            sizes = build_sizes(self.size_start, self.size_end, self.size_step)
            self.log.emit(f"Voxel size range：{sizes}")

            # For correlation output
            corr_rows = []

            # Main loop over voxel sizes
            total_steps = len(sizes)
            step_done = 0

            for s in sizes:
                if self._stop:
                    self.log.emit("Closed。")
                    break

                col = safe_column_name(s)
                df[col] = np.nan  # init column
                self.log.emit(f"\n--- Voxel size {s:.2f} m -> Write to column {col} ---")

                # loop over LAS files
                for fname in las_files:
                    if self._stop:
                        break
                    tree_id = os.path.splitext(fname)[0]
                    if tree_id not in id_to_row:
                        
                        self.log.emit(f"[Skip] {fname}  TreeID={tree_id} Not in CSV")
                        continue

                    las_path = os.path.join(self.las_dir, fname)
                    try:
                        pts = read_las_xyz(las_path)
                        vv = voxel_volume(pts, s)
                        df.loc[id_to_row[tree_id], col] = vv
                    except Exception as e:
                        self.log.emit(f"[fail] {fname} Voxel volume calculation failed：{e}")
                        df.loc[id_to_row[tree_id], col] = np.nan

                # correlation for this size
                valid = df[[self.vol_col, col]].dropna()
                n = len(valid)
                if n >= 2:
                    r, p = pearsonr(valid[self.vol_col].astype(float), valid[col].astype(float))
                    self.log.emit(f"correlation（Pearson）: r={r:.4f}, p={p:.4g}, n={n}")
                else:
                    r, p = np.nan, np.nan
                    self.log.emit(f"Correlation cannot be calculated（n={n}）")

                corr_rows.append({
                    "voxel_size_m": float(s),
                    "column": col,
                    "r": float(r) if np.isfinite(r) else np.nan,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "n": int(n)
                })

                step_done += 1
                self.progress.emit(step_done, total_steps)

            # If stopped before any result
            if len(corr_rows) == 0:
                self.finished.emit()
                return

            corr_df = pd.DataFrame(corr_rows)

            # Best voxel size by max r (not abs)
            corr_df_valid = corr_df.dropna(subset=["r"])
            if len(corr_df_valid) == 0:
                best_size = np.nan
                self.log.emit("\nNo correlation results found (all r values are NaN)。")
            else:
                best_idx = corr_df_valid["r"].idxmax()
                best_size = float(corr_df.loc[best_idx, "voxel_size_m"])
                best_r = float(corr_df.loc[best_idx, "r"])
                self.log.emit(f"\n>>> Optimal voxel size（max r）：{best_size:.2f} m （r={best_r:.4f}）")

            # Output files
            os.makedirs(self.out_dir, exist_ok=True)

            base_csv_name = os.path.splitext(os.path.basename(self.in_csv))[0]
            out_csv1 = os.path.join(self.out_dir, f"{base_csv_name}_with_voxel_volumes.csv")
            out_csv2 = os.path.join(self.out_dir, f"{base_csv_name}_voxel_r_list.csv")

            df.to_csv(out_csv1, index=False, encoding="utf-8-sig")
            corr_df.to_csv(out_csv2, index=False, encoding="utf-8-sig")

            dt = time.time() - t_all
            self.log.emit(f"\nOutput complete：\n1) {out_csv1}\n2) {out_csv2}\n Total time：{dt:.2f}s")

            self.done_outputs.emit(out_csv1, out_csv2, best_size)

        except Exception as e:
            self.log.emit("Operation failed：")
            self.log.emit(str(e))
            self.log.emit(traceback.format_exc())
        finally:
            self.finished.emit()


# -------------------------
# GUI
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voxel Size Optimization")
        self.resize(1100, 720)

        self.thread = None
        self.worker = None

        tb = QToolBar("Tools")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.btn_pick_las = QPushButton("Select the LAS folder")
        self.btn_pick_csv = QPushButton("Select CSV")
        self.btn_pick_out = QPushButton("Select output directory")
        self.btn_run = QPushButton("start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        tb.addWidget(self.btn_pick_las)
        tb.addWidget(self.btn_pick_csv)
        tb.addWidget(self.btn_pick_out)
        tb.addSeparator()
        tb.addWidget(self.btn_run)
        tb.addWidget(self.btn_stop)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)

        self.ed_las_dir = QLineEdit()
        self.ed_csv = QLineEdit()
        self.ed_out = QLineEdit()

        layout.addWidget(QLabel("LAS folder:"), 0, 0)
        layout.addWidget(self.ed_las_dir, 0, 1, 1, 3)
        layout.addWidget(QLabel("InputCSV:"), 1, 0)
        layout.addWidget(self.ed_csv, 1, 1, 1, 3)
        layout.addWidget(QLabel("Output:"), 2, 0)
        layout.addWidget(self.ed_out, 2, 1, 1, 3)

        # columns
        self.ed_id_col = QLineEdit("TreeID")
        self.ed_vol_col = QLineEdit("CSVolume")
        layout.addWidget(QLabel("TreeID Column:"), 3, 0)
        layout.addWidget(self.ed_id_col, 3, 1)
        layout.addWidget(QLabel("CSVolume Column:"), 3, 2)
        layout.addWidget(self.ed_vol_col, 3, 3)

        # voxel params
        self.sp_start = QDoubleSpinBox()
        self.sp_end = QDoubleSpinBox()
        self.sp_step = QDoubleSpinBox()
        for sp in (self.sp_start, self.sp_end, self.sp_step):
            sp.setDecimals(3)
            sp.setRange(0.001, 10.0)

        self.sp_start.setValue(0.20)
        self.sp_end.setValue(0.02)
        self.sp_step.setValue(0.02)

        layout.addWidget(QLabel("Voxelstart(m):"), 4, 0)
        layout.addWidget(self.sp_start, 4, 1)
        layout.addWidget(QLabel("end(m):"), 4, 2)
        layout.addWidget(self.sp_end, 4, 3)
        layout.addWidget(QLabel("step(m):"), 5, 0)
        layout.addWidget(self.sp_step, 5, 1)

        self.progress = QProgressBar()
        layout.addWidget(self.progress, 6, 0, 1, 4)

        self.logbox = QTextEdit()
        self.logbox.setReadOnly(True)
        layout.addWidget(self.logbox, 7, 0, 1, 4)

        # connections
        self.btn_pick_las.clicked.connect(self.pick_las_dir)
        self.btn_pick_csv.clicked.connect(self.pick_csv)
        self.btn_pick_out.clicked.connect(self.pick_out_dir)
        self.btn_run.clicked.connect(self.run)
        self.btn_stop.clicked.connect(self.stop)

    def append_log(self, msg: str):
        self.logbox.append(msg)
        self.logbox.verticalScrollBar().setValue(self.logbox.verticalScrollBar().maximum())

    def pick_las_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select the LAS folder", "")
        if d:
            self.ed_las_dir.setText(d)

    def pick_csv(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV (*.csv)")
        if f:
            self.ed_csv.setText(f)

    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory", "")
        if d:
            self.ed_out.setText(d)

    def run(self):
        las_dir = self.ed_las_dir.text().strip()
        in_csv = self.ed_csv.text().strip()
        out_dir = self.ed_out.text().strip()

        if not las_dir or not os.path.isdir(las_dir):
            QMessageBox.information(self, "Note", "Please select a valid LAS folder")
            return
        if not in_csv or not os.path.isfile(in_csv):
            QMessageBox.information(self, "Note", "Please select a valid CSV file")
            return
        if not out_dir:
            QMessageBox.information(self, "Note", "Please select the output directory")
            return

        id_col = self.ed_id_col.text().strip()
        vol_col = self.ed_vol_col.text().strip()
        if not id_col or not vol_col:
            QMessageBox.information(self, "Note", "Please enter the TreeID column name and the CSVolume column name")
            return

        s_start = float(self.sp_start.value())
        s_end = float(self.sp_end.value())
        s_step = float(self.sp_step.value())
        if s_step <= 0:
            QMessageBox.information(self, "Note", "step>0")
            return

        self.progress.setValue(0)
        self.logbox.clear()
        self.append_log("Start running...")

        # disable buttons
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pick_las.setEnabled(False)
        self.btn_pick_csv.setEnabled(False)
        self.btn_pick_out.setEnabled(False)

        self.thread = QThread()
        self.worker = Worker(las_dir, in_csv, out_dir, id_col, vol_col, s_start, s_end, s_step)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.on_progress)
        self.worker.done_outputs.connect(self.on_done_outputs)
        self.worker.finished.connect(self.on_finished)

        self.thread.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.append_log("stop command")
        self.btn_stop.setEnabled(False)

    def on_progress(self, done: int, total: int):
        if total <= 0:
            self.progress.setValue(0)
            return
        self.progress.setValue(int(done / total * 100))

    def on_done_outputs(self, out1: str, out2: str, best_size: float):
        if np.isfinite(best_size):
            self.append_log(f"\n Optimal voxel size：{best_size:.2f} m")
        self.append_log(f"\n Output file1（Voxel Volume Array）：{out1}")
        self.append_log(f"Output File 2：{out2}")

    def on_finished(self):
        self.append_log("\n Process completed。")

        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pick_las.setEnabled(True)
        self.btn_pick_csv.setEnabled(True)
        self.btn_pick_out.setEnabled(True)

        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None


def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
