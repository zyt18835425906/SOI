import os
import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
import time
import traceback
from scipy.spatial import ConvexHull, Delaunay, KDTree
from scipy import stats
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  


class CanopyStructureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CR&FD_TO_CSC(plot)")
        self.root.geometry("700x500")

        # Variable
        self.folder_path = tk.StringVar()
        self.grid_size = tk.DoubleVar(value=1.0)  # Grid size
        self.min_height = tk.DoubleVar(value=2.0)  # Minimum Height Threshold
        self.voxel_size = tk.DoubleVar(value=1.0)  # Voxel size
        self.progress = tk.DoubleVar()
        self.status_text = tk.StringVar(value="Ready")

        # Create Interface
        self.create_widgets()

    def create_widgets(self):
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Folder Selection
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Selection", padding="5")
        folder_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Entry(folder_frame, textvariable=self.folder_path, width=50).grid(row=0, column=0, padx=5)
        ttk.Button(folder_frame, text="Scan...", command=self.browse_folder).grid(row=0, column=1, padx=5)

        # Parameter Settings
        param_frame = ttk.LabelFrame(main_frame, text="Parameter Settings", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(param_frame, text="Grid size (m):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=0.1, to=5.0, increment=0.1,
                    textvariable=self.grid_size, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Minimum Height Threshold (m):").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=0.5, to=10.0, increment=0.5,
                    textvariable=self.min_height, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Voxel size (m):").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=0.1, to=5.0, increment=0.1,
                    textvariable=self.voxel_size, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)

        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(progress_frame, textvariable=self.status_text).grid(row=0, column=0, sticky=tk.W)
        ttk.Progressbar(progress_frame, variable=self.progress, maximum=100).grid(row=1, column=0, sticky=(tk.W, tk.E),
                                                                                  pady=5)

        # Button
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Begin processing", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.log_text = tk.Text(log_frame, height=10, width=70)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path.set(folder_selected)

    def log_message(self, message):
        """Adding to the log"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_processing(self):
        """Begin processing the LAS file"""
        folder_path = self.folder_path.get()
        grid_size = self.grid_size.get()
        min_height = self.min_height.get()
        voxel_size = self.voxel_size.get()

        if not folder_path:
            messagebox.showerror("Error", "Select the LAS folder")
            return

        thread = Thread(target=self.process_las_files, args=(folder_path, grid_size, min_height, voxel_size))
        thread.daemon = True
        thread.start()

    def process_las_files(self, folder_path, grid_size, min_height, voxel_size):
        """Process all LAS files in the folder"""
        try:
            self.status_text.set("Being processed...")
            self.progress.set(0)

            # Retrieve all LAS files in the folder
            las_files = [f for f in os.listdir(folder_path) if f.endswith('.las')]
            total_files = len(las_files)

            if total_files == 0:
                self.log_message("The LAS file was not found in the specified folder")
                self.status_text.set("Completed - File not found")
                return

            self.log_message(f"found {total_files} LAS")

            results = []
            processed_files = 0

            for las_file in las_files:
                file_path = os.path.join(folder_path, las_file)
                file_id = os.path.splitext(las_file)[0]

                try:
                    self.log_message(f"Processing files: {las_file}")

                    # Read point cloud data
                    points = self.read_las_file(file_path)

                    # Calculate canopy structure indices
                    structure_metrics = self.calculate_structure_metrics(points, grid_size, min_height, voxel_size)

                    # Record the results
                    result = {
                        'ID': file_id,
                        'Points_Count': len(points),
                        'Canopy_Relief_Ratio_CRR': structure_metrics['CRR'],
                        'Canopy_Cover_CC': structure_metrics['CC'],
                        'Canopy_Porosity': structure_metrics['Porosity'],
                        'Leaf_Area_Index_LAI': structure_metrics['LAI'],
                        'Effective_Layer_Height': structure_metrics['ELH'],
                        'Fractal_Dimension': structure_metrics['FD']
                    }
                    results.append(result)

                    self.log_message(f"file {file_id} Processing complete:")
                    self.log_message(f"  - Canopy Height Ratio (CRR): {structure_metrics['CRR']:.4f}")
                    self.log_message(f"  - Canopy Cover (CC): {structure_metrics['CC']:.4f}")
                    self.log_message(f"  - Canopy Porosity: {structure_metrics['Porosity']:.4f}")
                    self.log_message(f"  - Leaf Area Index (LAI): {structure_metrics['LAI']:.4f}")
                    self.log_message(f"  - Effective Layer Height: {structure_metrics['ELH']:.4f}")
                    self.log_message(f"  - Fractal Dimension: {structure_metrics['FD']:.4f}")

                except Exception as e:
                    error_msg = f"Processing file {file_id} error occurs: {str(e)}\n{traceback.format_exc()}"
                    self.log_message(error_msg)
                    results.append({
                        'ID': file_id,
                        'Points_Count': -1,
                        'Canopy_Relief_Ratio_CRR': -1,
                        'Canopy_Cover_CC': -1,
                        'Canopy_Porosity': -1,
                        'Leaf_Area_Index_LAI': -1,
                        'Effective_Layer_Height': -1,
                        'Fractal_Dimension': -1
                    })

                processed_files += 1
                self.progress.set((processed_files / total_files) * 100)

            # Save results to a CSV 
            csv_path = os.path.join(folder_path, f"canopy_structure_metrics.csv")
            df = pd.DataFrame(results)

            # Save results to a CSV 
            try:
                # Check if the directory exists; if it does not exist, create it
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)

                # Check whether the file is in use by another program
                if os.path.exists(csv_path):
                    try:
                        # Try opening the file
                        with open(csv_path, 'a') as f:
                            pass
                    except IOError:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        csv_path = os.path.join(folder_path, f"canopy_structure_metrics_{timestamp}.csv")
                        self.log_message(f"New file name: {csv_path}")

                # Save CSV file
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                self.log_message(f"Save CSV file {csv_path}")

            except PermissionError:
             
                user_docs = os.path.join(os.path.expanduser("~"), "Documents")
                csv_path = os.path.join(user_docs, "canopy_structure_metrics.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                self.log_message(f"Warning: Results saved to {csv_path}")
                self.root.after(0, lambda: messagebox.showwarning("Permission error",
                                                                  f"Results saved to: {csv_path}"))

            except Exception as e:
                error_msg = f"Error occurred: {str(e)}"
                self.log_message(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error occurred", error_msg))
                return

            # Generate visual charts
            try:
                self.visualize_results(df, folder_path)
            except Exception as e:
                self.log_message(f"Error occurred: {str(e)}")

            self.status_text.set("Processing complete")
            self.root.after(0, lambda: messagebox.showinfo("Complete", f"Processing complete！Results saved to {csv_path}"))

        except Exception as e:
            error_msg = f"Errors occurred: {str(e)}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            self.status_text.set("Processing failed")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Errors occurred: {str(e)}"))

    def read_las_file(self, file_path):
        """Read the LAS file"""
        self.log_message(f"Reading file: {file_path}")
        las = laspy.read(file_path)


        points = np.vstack((las.x, las.y, las.z)).transpose()

      
        if points.ndim == 1:
            self.log_message("Warning:2-dimensions ")
        
            points = points.reshape(-1, 3)

        self.log_message(f"Read {len(points)} point，shape: {points.shape}")
        return points

    def calculate_structure_metrics(self, points, grid_size, min_height, voxel_size):
        """Calculate canopy structure indices"""
        self.log_message("Calculate canopy structure indices...")

    
        if points.ndim == 1:
            points = points.reshape(-1, 3)

        # Canopy Height Ratio (CRR)
        crr = self.calculate_canopy_relief_ratio(points, grid_size)

        # Canopy Cover (CC)
        cc = self.calculate_canopy_cover(points, grid_size, min_height)

        # Canopy Porosity
        porosity = self.calculate_canopy_porosity(points, grid_size, min_height)

        # Leaf Area Index (LAI)
        lai = self.calculate_leaf_area_index(points, grid_size, min_height)

        # Effective Layer Height
        elh = self.calculate_effective_layer_height(points, voxel_size)

        # Fractal Dimension
        fd = self.calculate_fractal_dimension(points, grid_size)

        return {
            'CRR': crr,
            'CC': cc,
            'Porosity': porosity,
            'LAI': lai,
            'ELH': elh,
            'FD': fd
        }

    def calculate_canopy_relief_ratio(self, points, grid_size):
        """Canopy Height Ratio (CRR)"""
        self.log_message("Canopy Height Ratio (CRR)...")

        # Obtain XY coordinates
        xy_points = points[:, :2]

        # Calculate the XY boundaries
        x_min, x_max = np.min(xy_points[:, 0]), np.max(xy_points[:, 0])
        y_min, y_max = np.min(xy_points[:, 1]), np.max(xy_points[:, 1])

        # Create Grid
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        crr_values = []

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                
                in_x = (xy_points[:, 0] >= x_bins[i]) & (xy_points[:, 0] < x_bins[i + 1])
                in_y = (xy_points[:, 1] >= y_bins[j]) & (xy_points[:, 1] < y_bins[j + 1])
                in_cell = in_x & in_y

                if np.any(in_cell):
                 
                    z_values = points[in_cell, 2]
                    z_min = np.min(z_values)
                    z_max = np.max(z_values)
                    z_mean = np.mean(z_values)

                    # CRR
                    if z_max > z_min:
                        crr = (z_mean - z_min) / (z_max - z_min)
                        crr_values.append(crr)

        # Mean CRR
        if crr_values:
            avg_crr = np.mean(crr_values)
        else:
            avg_crr = 0

        return avg_crr

    def calculate_canopy_cover(self, points, grid_size, min_height):
        """Canopy cover (CC)"""
        self.log_message("Canopy cover (CC)...")

        # Obtain XY coordinates
        xy_points = points[:, :2]

        # Calculate the XY boundaries
        x_min, x_max = np.min(xy_points[:, 0]), np.max(xy_points[:, 0])
        y_min, y_max = np.min(xy_points[:, 1]), np.max(xy_points[:, 1])

        # Create Grid
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        total_cells = (len(x_bins) - 1) * (len(y_bins) - 1)
        canopy_cells = 0

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                
                in_x = (xy_points[:, 0] >= x_bins[i]) & (xy_points[:, 0] < x_bins[i + 1])
                in_y = (xy_points[:, 1] >= y_bins[j]) & (xy_points[:, 1] < y_bins[j + 1])
                in_cell = in_x & in_y

                if np.any(in_cell):
                    
                    z_values = points[in_cell, 2]
                    
                    if np.any(z_values >= min_height):
                        canopy_cells += 1

        # Canopy Cover
        if total_cells > 0:
            cc = canopy_cells / total_cells
        else:
            cc = 0

        return cc

    def calculate_canopy_porosity(self, points, grid_size, min_height):
        """Canopy Porosity"""
        self.log_message("Canopy Porosity...")

        # Canopy porosity = 1 - Canopy cover
        cc = self.calculate_canopy_cover(points, grid_size, min_height)
        porosity = 1 - cc

        return porosity

    def calculate_leaf_area_index(self, points, grid_size, min_height):
        """Leaf Area Index (LAI)"""
        self.log_message("Leaf Area Index (LAI)...")

        # Gap Ratio To LAI
        # LAI = -ln(P) * k, P is the canopy porosity, and k is the extinction coefficient (typically taken as 0.5–0.7).
        porosity = self.calculate_canopy_porosity(points, grid_size, min_height)

    
        if porosity <= 0:
            porosity = 0.001
        elif porosity >= 1:
            porosity = 0.999

        # Extinction Coefficient k = 0.5
        k = 0.5
        lai = -np.log(porosity) * k

        return lai

    def calculate_effective_layer_height(self, points, voxel_size):
        """Effective Layer Height"""
        self.log_message("Effective Layer Height...")

        
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

       
        z_bins = np.arange(z_min, z_max + voxel_size, voxel_size)

       
        layer_density = np.zeros(len(z_bins) - 1)

        for k in range(len(z_bins) - 1):
            
            in_z = (points[:, 2] >= z_bins[k]) & (points[:, 2] < z_bins[k + 1])
            layer_density[k] = np.sum(in_z)

        
        if np.sum(layer_density) > 0:
            layer_heights = (z_bins[:-1] + z_bins[1:]) / 2
            effective_height = np.sum(layer_heights * layer_density) / np.sum(layer_density)
        else:
            effective_height = np.mean(points[:, 2])

        return effective_height

    def calculate_fractal_dimension(self, points, grid_size):
        """Fractal Dimension"""
        self.log_message("Fractal Dimension...")


        xy_points = points[:, :2]

        x_min, x_max = np.min(xy_points[:, 0]), np.max(xy_points[:, 0])
        y_min, y_max = np.min(xy_points[:, 1]), np.max(xy_points[:, 1])

     
        box_sizes = np.logspace(np.log10(grid_size), np.log10(min(x_max - x_min, y_max - y_min) / 2), 10)
        box_counts = []

        for size in box_sizes:
           
            x_bins = np.arange(x_min, x_max + size, size)
            y_bins = np.arange(y_min, y_max + size, size)

           
            boxes = set()
            for point in xy_points:
                x_idx = int((point[0] - x_min) / size)
                y_idx = int((point[1] - y_min) / size)
                boxes.add((x_idx, y_idx))

            box_counts.append(len(boxes))

        
        if len(box_counts) > 1:
            log_sizes = np.log(1 / box_sizes)
            log_counts = np.log(box_counts)

          
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
            fractal_dimension = slope
        else:
            fractal_dimension = 0

        return fractal_dimension

    def visualize_results(self, df, output_folder):
        """Generate visual results"""
        try:
            self.log_message("Generate visual results...")

         
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. Crown Height Ratio Distribution
            axes[0, 0].hist(df['Canopy_Relief_Ratio_CRR'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Canopy Relief Ratio (CRR) Distribution')
            axes[0, 0].set_xlabel('CRR')
            axes[0, 0].set_ylabel('Frequency')

            # 2. Canopy Cover Distribution
            axes[0, 1].hist(df['Canopy_Cover_CC'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_title('Canopy Cover (CC) Distribution')
            axes[0, 1].set_xlabel('CC')
            axes[0, 1].set_ylabel('Frequency')

            # 3. Canopy Porosity Distribution
            axes[0, 2].hist(df['Canopy_Porosity'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 2].set_title('Canopy Porosity Distribution')
            axes[0, 2].set_xlabel('Porosity')
            axes[0, 2].set_ylabel('Frequency')

            # 4. Leaf Area Index Distribution
            axes[1, 0].hist(df['Leaf_Area_Index_LAI'], bins=20, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 0].set_title('Leaf Area Index (LAI) Distribution')
            axes[1, 0].set_xlabel('LAI')
            axes[1, 0].set_ylabel('Frequency')

            # 5. Effective Layer Height Distribution
            axes[1, 1].hist(df['Effective_Layer_Height'], bins=20, alpha=0.7, color='violet', edgecolor='black')
            axes[1, 1].set_title('Effective Layer Height Distribution')
            axes[1, 1].set_xlabel('Effective Layer Height (m)')
            axes[1, 1].set_ylabel('Frequency')

            # 6. Fractal Dimension Distribution
            axes[1, 2].hist(df['Fractal_Dimension'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 2].set_title('Fractal Dimension Distribution')
            axes[1, 2].set_xlabel('Fractal Dimension')
            axes[1, 2].set_ylabel('Frequency')

            plt.tight_layout()

            # Save Image
            output_filename = os.path.join(output_folder, "canopy_structure_metrics_analysis.png")
            plt.savefig(output_filename, dpi=300)
            plt.close()
            self.log_message(f"Visualization results saved as: {output_filename}")

            # Correlation Matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            metrics = ['Canopy_Relief_Ratio_CRR', 'Canopy_Cover_CC', 'Canopy_Porosity',
                       'Leaf_Area_Index_LAI', 'Effective_Layer_Height', 'Fractal_Dimension']
            correlation_matrix = df[metrics].corr()

            im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(metrics)))
            ax.set_yticks(np.arange(len(metrics)))
            ax.set_xticklabels(['CRR', 'CC', 'Porosity', 'LAI', 'ELH', 'FD'], rotation=45, ha='right')
            ax.set_yticklabels(['CRR', 'CC', 'Porosity', 'LAI', 'ELH', 'FD'])

      
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black")

            ax.set_title('Canopy Structure Metrics Correlation Matrix')
            fig.colorbar(im, ax=ax)
            plt.tight_layout()

            # SAVE Correlation Matrix
            corr_filename = os.path.join(output_folder, "canopy_structure_metrics_correlation.png")
            plt.savefig(corr_filename, dpi=300)
            plt.close()
            self.log_message(f"correlation matrix saved as: {corr_filename}")

        except Exception as e:
            self.log_message(f"Errors occurred: {str(e)}\n{traceback.format_exc()}")


# MAIN
if __name__ == "__main__":
    root = tk.Tk()
    app = CanopyStructureApp(root)
    root.mainloop()