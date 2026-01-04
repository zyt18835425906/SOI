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
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings('ignore')


class PointCloudDensityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ND_PC ")
        self.root.geometry("700x500")

        # Variable
        self.folder_path = tk.StringVar()
        self.resolution = tk.DoubleVar(value=0.5)  # Default resolutionto 0.5 meters
        self.k_neighbors = tk.IntVar(value=50)  # default number of neighborhood points to 50.
        self.normalization_method = tk.StringVar(value="MinMax")  # Normalization Method
        self.progress = tk.DoubleVar()
        self.status_text = tk.StringVar(value="Ready")

        # Create Interface
        self.create_widgets()

    def create_widgets(self):
        # main
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Folder Selection
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Selectio", padding="5")
        folder_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Entry(folder_frame, textvariable=self.folder_path, width=50).grid(row=0, column=0, padx=5)
        ttk.Button(folder_frame, text="Scan...", command=self.browse_folder).grid(row=0, column=1, padx=5)

        # Parameter Settings
        param_frame = ttk.LabelFrame(main_frame, text="Parameter Settings", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(param_frame, text="Voxel resolution (m):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=0.01, to=5.0, increment=0.1,
                    textvariable=self.resolution, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Neighborhood Points:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=5, to=200, increment=5,
                    textvariable=self.k_neighbors, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Normalization Method:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Combobox(param_frame, textvariable=self.normalization_method,
                     values=["MinMax", "Standard", "Log", "Robust"], width=10).grid(row=2, column=1, sticky=tk.W,
                                                                                    padx=5)

        # Progress 
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
        log_frame = ttk.LabelFrame(main_frame, text="Processing Logs", padding="5")
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
        """Add to log"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_processing(self):
        """Begin processing the LAS file"""
        folder_path = self.folder_path.get()
        resolution = self.resolution.get()
        k_neighbors = self.k_neighbors.get()
        normalization_method = self.normalization_method.get()

        if not folder_path:
            messagebox.showerror("Error", "LAS file")
            return

        thread = Thread(target=self.process_las_files,
                        args=(folder_path, resolution, k_neighbors, normalization_method))
        thread.daemon = True
        thread.start()

    def process_las_files(self, folder_path, resolution, k_neighbors, normalization_method):
        """Process all LAS files in the folder"""
        try:
            self.status_text.set("Currently being processed...")
            self.progress.set(0)

            # Retrieve all LAS files in the folder
            las_files = [f for f in os.listdir(folder_path) if f.endswith('.las')]
            total_files = len(las_files)

            if total_files == 0:
                self.log_message("The LAS file was not found in the specified folder.")
                self.status_text.set("Completed - File not found")
                return

            self.log_message(f"Found {total_files} LAS")

            # Stage One: Collect all density measurements
            all_density_metrics = []
            file_info = []

            for i, las_file in enumerate(las_files):
                file_path = os.path.join(folder_path, las_file)
                file_id = os.path.splitext(las_file)[0]

                try:
                    self.log_message(f"Calculating density metric: {las_file}")

                    # Read point cloud data
                    points = self.read_las_file(file_path)

                    # Computational Density Metrics
                    density_metrics = self.calculate_density_metrics(points, resolution, k_neighbors)

                    # Storage Metrics
                    all_density_metrics.append(density_metrics)
                    file_info.append({
                        'ID': file_id,
                        'points_count': len(points),
                        'raw_density': density_metrics[0],
                        'local_density': density_metrics[1],
                        'voxel_density': density_metrics[2]
                    })

                    self.log_message(f"file {file_id} Density completed")

                except Exception as e:
                    error_msg = f"Calculation File {file_id} Error occurred: {str(e)}\n{traceback.format_exc()}"
                    self.log_message(error_msg)
                 
                    all_density_metrics.append([0, 0, 0])
                    file_info.append({
                        'ID': file_id,
                        'points_count': 0,
                        'raw_density': 0,
                        'local_density': 0,
                        'voxel_density': 0
                    })

                self.progress.set((i + 1) / total_files * 50)  

            # Stage 2: Calculate normalized density
            results = []
            normalized_densities = self.normalize_densities(all_density_metrics, normalization_method)

            for i, file_id in enumerate(las_files):
                file_id = os.path.splitext(file_id)[0]
                normalized_density = normalized_densities[i]

                results.append({
                    'ID': file_id,
                    'raw points': file_info[i]['points_count'],
                    'Normalized Density': normalized_density,
                    'Raw density': file_info[i]['raw_density'],
                    'Local density': file_info[i]['local_density'],
                    'Voxel density': file_info[i]['voxel_density']
                })

                self.log_message(f"file {file_id} Normalized Density: {normalized_density:.6f}")
                self.progress.set(50 + (i + 1) / total_files * 50)  
            # Save results to a CSV file
            csv_path = os.path.join(folder_path, f"normalized_density_results_{normalization_method}.csv")
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)

            # Generate visual charts
            self.visualize_results(df, folder_path, normalization_method)

            self.log_message(f"The results saved to {csv_path}")
            self.status_text.set("Processing complete")
            messagebox.showinfo("Completed", f"Processing complete！The results saved to {csv_path}")

        except Exception as e:
            error_msg = f"Errors occurred: {str(e)}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            self.status_text.set("Processing failed")
            messagebox.showerror("Error", f"Errors occurred: {str(e)}")

    def read_las_file(self, file_path):
        """Read the LAS file and return the point cloud data"""
        self.log_message(f"Reading file: {file_path}")
        las = laspy.read(file_path)


        points = np.vstack((las.x, las.y, las.z)).transpose()

        if points.ndim == 1:
            self.log_message("Warning: Point cloud data is 1-dimensional; attempt to reshape it into 2-dimensions.")
          
            points = points.reshape(-1, 3)

        self.log_message(f"Read {len(points)} point，shape: {points.shape}")
        return points

    def calculate_density_metrics(self, points, resolution, k_neighbors):
        """Density Measures for Point Clouds"""
        self.log_message("Calculating density metric...")

        
        if points.ndim == 1:
            points = points.reshape(-1, 3)

        # 1. Calculate point cloud bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        bbox_volume = np.prod(max_coords - min_coords)

        # 2. Calculate the raw density (points per unit volume)
        raw_density = len(points) / bbox_volume if bbox_volume > 0 else 0

        # 3. Calculate local density (using the k-nearest neighbors method)
        if len(points) > k_neighbors:
            try:
                kdtree = KDTree(points)
                distances, _ = kdtree.query(points, k=k_neighbors + 1)  
                # Compute the local density at each point(using the inverse of the average distance of the K nearest neighbors)
                local_densities = 1.0 / np.mean(distances[:, 1:], axis=1)  
                avg_local_density = np.mean(local_densities)
            except:
                avg_local_density = raw_density
        else:
            avg_local_density = raw_density

        # 4. Calculate voxel density
        voxel_grid = {}
        for point in points:
            # Calculate the coordinates of the voxel 
            voxel_coord = tuple((point // resolution).astype(int))
            if voxel_coord not in voxel_grid:
                voxel_grid[voxel_coord] = 0
            voxel_grid[voxel_coord] += 1

        # Calculate the average voxel density
        voxel_densities = list(voxel_grid.values())
        avg_voxel_density = np.mean(voxel_densities) if voxel_densities else 0

        self.log_message(f"Raw density: {raw_density:.6f}, Local density: {avg_local_density:.6f}, Voxel density: {avg_voxel_density:.6f}")

        return [raw_density, avg_local_density, avg_voxel_density]

    def normalize_densities(self, density_metrics, method="MinMax"):
        """Normalized Density"""
        self.log_message(f"do {method} for normalization...")

        # Convert to a NumPy array
        metrics_array = np.array(density_metrics)

        if np.any(np.isnan(metrics_array)) or np.any(np.isinf(metrics_array)):
            self.log_message("Warning: Density metrics containing NaN or Inf values will be replaced with 0.")
            metrics_array = np.nan_to_num(metrics_array)

        # Normalization Methods for Application Selection
        if method == "MinMax":
            # MinMax
            scaler = MinMaxScaler()
            normalized_metrics = scaler.fit_transform(metrics_array)
        elif method == "Standard":
            # Z-score
            scaler = StandardScaler()
            normalized_metrics = scaler.fit_transform(metrics_array)
        elif method == "Log_Nunit":
            # Log_Nunit,Nunit is the global density of the plot.
            min_val = np.min(metrics_array)
            if min_val <= 0:
                metrics_array = metrics_array - min_val + 1e-10  
            normalized_metrics = np.log1p(metrics_array)
          
            scaler = MinMaxScaler()
            normalized_metrics = scaler.fit_transform(normalized_metrics)
        elif method == "Robust":
            
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            normalized_metrics = scaler.fit_transform(metrics_array)
        else:
            # MinMax
            scaler = MinMaxScaler()
            normalized_metrics = scaler.fit_transform(metrics_array)

        # Calculate Weighted Normalized Density
        weights = [0.3, 0.4, 0.3]  # corresponding to the raw density, local density, and voxel density respectively
        normalized_densities = np.dot(normalized_metrics, weights)

        # Ensure all values are within the range [0,1]
        normalized_densities = np.clip(normalized_densities, 0, 1)

        return normalized_densities

    def visualize_results(self, df, output_folder, normalization_method):
        """Generate visual results"""
        try:
            self.log_message("Generate visual results...")

            # Create a chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Normalized Density Distribution Histogram
            axes[0, 0].hist(df['Normalized Density'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Normalized Density Distribution')
            axes[0, 0].set_xlabel('Normalized Density')
            axes[0, 0].set_ylabel('Frequency')

            # 2. Relationship Between Raw Point Scores and Normalized Density
            axes[0, 1].scatter(df['Raw Point Count'], df['Normalized Density'], alpha=0.7, color='lightcoral')
            axes[0, 1].set_title('Relationship Between Raw Point Scores and Normalized Density')
            axes[0, 1].set_xlabel('Raw Points')
            axes[0, 1].set_ylabel('Normalized Density')

            # 3. Normalized Density
            sorted_df = df.sort_values('Normalized Density', ascending=False)
            axes[1, 0].bar(range(len(sorted_df)), sorted_df['Normalized Density'], alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Normalized Density Rank')
            axes[1, 0].set_xlabel('Rank')
            axes[1, 0].set_ylabel('Normalized Density')
            axes[1, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            # 4. Density Statistics
            axes[1, 1].axis('off')
            stats_text = f"""
            Density Statistics:
            Number of files: {len(df)}
            Average normalized density: {df['Normalized Density'].mean():.6f}
            Normalized density standard deviation: {df['Normalized Density'].std():.6f}
            Maximum Normalized Density: {df['Normalized Density'].max():.6f}
            Minimum Normalized Density: {df['Normalized Density'].min():.6f}
            Normalization Method: {normalization_method}
            """
            axes[1, 1].text(0.1, 0.9, stats_text, fontsize=12, verticalalignment='top')

            plt.tight_layout()

            # Save Image
            output_filename = os.path.join(output_folder, f"normalized_density_analysis_{normalization_method}.png")
            plt.savefig(output_filename, dpi=300)
            plt.close()
            self.log_message(f"Visualization results saved as: {output_filename}")

        except Exception as e:
            self.log_message(f"Errors occurred: {str(e)}\n{traceback.format_exc()}")


# main
if __name__ == "__main__":
    root = tk.Tk()
    app = PointCloudDensityApp(root)
    root.mainloop()