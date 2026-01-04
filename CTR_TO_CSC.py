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
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 


class CanopyRugosityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CTR_TO_CSC")
        self.root.geometry("700x500")

        # Parameters
        self.folder_path = tk.StringVar()
        self.voxel_size = tk.DoubleVar(value=1.0)  # Voxel size
        self.grid_size = tk.DoubleVar(value=1.0)  # Grid size
        self.progress = tk.DoubleVar()
        self.status_text = tk.StringVar(value="Ready")

        # Create Interface
        self.create_widgets()

    def create_widgets(self):
        # main
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

        ttk.Label(param_frame, text="Voxel size (m):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=0.1, to=5.0, increment=0.1,
                    textvariable=self.voxel_size, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Grid size (m):").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=0.1, to=5.0, increment=0.1,
                    textvariable=self.grid_size, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        # Progress
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(progress_frame, textvariable=self.status_text).grid(row=0, column=0, sticky=tk.W)
        ttk.Progressbar(progress_frame, variable=self.progress, maximum=100).grid(row=1, column=0, sticky=(tk.W, tk.E),
                                                                                  pady=5)

        # Button
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Processing", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Processing Logs", padding="5")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.log_text = tk.Text(log_frame, height=10, width=70)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure weights to enable component scalability
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
        """Add a message to the log area"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_processing(self):
        """Begin processing the LAS file"""
        folder_path = self.folder_path.get()
        voxel_size = self.voxel_size.get()
        grid_size = self.grid_size.get()

        if not folder_path:
            messagebox.showerror("Error", "Please select the folder containing the LAS file")
            return

        # Run the processing procedur
        thread = Thread(target=self.process_las_files, args=(folder_path, voxel_size, grid_size))
        thread.daemon = True
        thread.start()

    def process_las_files(self, folder_path, voxel_size, grid_size):
        """Process all LAS files in the folder"""
        try:
            self.status_text.set("Being processed...")
            self.progress.set(0)

            # Retrieve all LAS files in the folder
            las_files = [f for f in os.listdir(folder_path) if f.endswith('.las')]
            total_files = len(las_files)

            if total_files == 0:
                self.log_message("The LAS file was not found in the folder")
                self.status_text.set("Completed - File not found")
                return

            self.log_message(f"Found {total_files} LAS file")

            results = []
            processed_files = 0

            for las_file in las_files:
                file_path = os.path.join(folder_path, las_file)
                file_id = os.path.splitext(las_file)[0]

                try:
                    self.log_message(f"Processing files: {las_file}")

                    # Read point cloud data
                    points = self.read_las_file(file_path)

                    # Calculate the canopy roughness index
                    rugosity_metrics = self.calculate_rugosity_metrics(points, voxel_size, grid_size)

                    # Record the results
                    result = {
                        'ID': file_id,
                        'Points_Count': len(points),
                        'Canopy_Rugosity_Rc': rugosity_metrics['Rc'],
                        'Top_Rugosity_RT': rugosity_metrics['RT'],
                        'Rumple_Index': rugosity_metrics['Rumple'],
                        'Effective_Number_of_Layers_ENL': rugosity_metrics['ENL']
                    }
                    results.append(result)

                    self.log_message(f"file {file_id} Processing complete:")
                    self.log_message(f"  - Canopy rugosity (CR): {rugosity_metrics['Rc']:.4f}")
                    self.log_message(f"  - Canopy top rugosity (CTR): {rugosity_metrics['RT']:.4f}")
                    self.log_message(f"  - Canopy surface ratio (Rumple): {rugosity_metrics['Rumple']:.4f}")
                    self.log_message(f"  - Effective Layer Count (ENL): {rugosity_metrics['ENL']:.4f}")

                except Exception as e:
                    error_msg = f"Processing files {file_id} error occurs: {str(e)}\n{traceback.format_exc()}"
                    self.log_message(error_msg)
                    results.append({
                        'ID': file_id,
                        'Points_Count': -1,
                        'Canopy_Rugosity_Rc': -1,
                        'Top_Rugosity_RT': -1,
                        'Rumple_Index': -1,
                        'Effective_Number_of_Layers_ENL': -1
                    })

                processed_files += 1
                self.progress.set((processed_files / total_files) * 100)

            # Save results to a CSV file 
            csv_path = os.path.join(folder_path, f"canopy_rugosity_results.csv")
            df = pd.DataFrame(results)

            # saving CSV files
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # Generate visual charts
            self.visualize_results(df, folder_path)

            self.log_message(f"The results have been saved to {csv_path}")
            self.status_text.set("Processing complete")
            messagebox.showinfo("Completed", f"Processing complete！saved to {csv_path}")

        except Exception as e:
            error_msg = f"Errors occurred: {str(e)}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            self.status_text.set("Processing failed")
            messagebox.showerror("Error", f"Errors occurred: {str(e)}")

    def read_las_file(self, file_path):
        """Read the LAS file and return the point cloud data"""
        self.log_message(f"Reading file: {file_path}")
        las = laspy.read(file_path)

        # Ensure that the point cloud data is a two-dimensional array.
        points = np.vstack((las.x, las.y, las.z)).transpose()

        # Check array dimensions
        if points.ndim == 1:
            self.log_message("Warning：Point cloud data is one-dimensional")
          
            points = points.reshape(-1, 3)

        self.log_message(f"Read {len(points)} points，shape: {points.shape}")
        return points

    def calculate_rugosity_metrics(self, points, voxel_size, grid_size):
        """CTR_TO_CSC"""
        self.log_message("CTR_TO_CSC...")

        if points.ndim == 1:
            points = points.reshape(-1, 3)

        # Canopy rugosity (CR)
        Rc = self.calculate_canopy_rugosity(points, voxel_size)

        # Canopy top rugosity (CTR)
        RT = self.calculate_top_rugosity(points, grid_size)

        # Canopy surface ratio (Rumple)
        rumple = self.calculate_rumple_index(points)

        # Effective Layer Count (ENL)
        enl = self.calculate_effective_number_of_layers(points, voxel_size)

        return {
            'Rc': Rc,
            'RT': RT,
            'Rumple': rumple,
            'ENL': enl
        }

    def calculate_canopy_rugosity(self, points, voxel_size):
        """Canopy rugosity (CR)"""
        self.log_message("Canopy rugosity (CR)...")

        # Getting the boundaries of the point cloud
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Create a voxel mesh
        x_bins = np.arange(min_coords[0], max_coords[0] + voxel_size, voxel_size)
        y_bins = np.arange(min_coords[1], max_coords[1] + voxel_size, voxel_size)
        z_bins = np.arange(min_coords[2], max_coords[2] + voxel_size, voxel_size)

        # Calculate the point density for each voxel
        voxel_density = np.zeros((len(x_bins) - 1, len(y_bins) - 1, len(z_bins) - 1))

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                for k in range(len(z_bins) - 1):
                    # Check whether there is a point within the current voxel
                    in_x = (points[:, 0] >= x_bins[i]) & (points[:, 0] < x_bins[i + 1])
                    in_y = (points[:, 1] >= y_bins[j]) & (points[:, 1] < y_bins[j + 1])
                    in_z = (points[:, 2] >= z_bins[k]) & (points[:, 2] < z_bins[k + 1])
                    voxel_density[i, j, k] = np.sum(in_x & in_y & in_z)

        # Canopy rugosity (CR)
        # CR is a comprehensive measure of voxel density variability in both horizontal and vertical directions
        #where the standard deviation of voxel density is used as an approximation for roughness.

        Rc = np.std(voxel_density[voxel_density > 0])

        return Rc

    def calculate_top_rugosity(self, points, grid_size):
        """Canopy top rugosity (CTR)"""
        self.log_message("Canopy top rugosity (CTR)...")

        # Obtain XY coordinates
        xy_points = points[:, :2]

        # Calculate the XY boundaries of the point cloud
        x_min, x_max = np.min(xy_points[:, 0]), np.max(xy_points[:, 0])
        y_min, y_max = np.min(xy_points[:, 1]), np.max(xy_points[:, 1])

        # Create a grid
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        # Calculate the maximum height for each grid
        max_heights = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                # Check whether there are canopy points within the current grid
                in_x = (xy_points[:, 0] >= x_bins[i]) & (xy_points[:, 0] < x_bins[i + 1])
                in_y = (xy_points[:, 1] >= y_bins[j]) & (xy_points[:, 1] < y_bins[j + 1])
                in_cell = in_x & in_y

                if np.any(in_cell):
                    # Retrieve the Z-coordinate of the point within the current grid
                    z_values = points[in_cell, 2]
                    max_heights[j, i] = np.max(z_values)
                else:
                    max_heights[j, i] = np.nan

        # Canopy top rugosity (CTR) - Standard deviation of maximum height
        RT = np.nanstd(max_heights)

        return RT

    def calculate_rumple_index(self, points):
        """Canopy surface ratio (Rumple)"""
        self.log_message("Canopy surface ratio (Rumple)...")

        # Obtain XY coordinates
        xy_points = points[:, :2]

        # Calculate the area of the convex hull 
        try:
            hull = ConvexHull(xy_points)
            ground_area = hull.volume  # For a 2D convex hull, the volume property represents the area.
        except:
            # If convex hull calculation fails, use bounding box area.
            x_min, x_max = np.min(xy_points[:, 0]), np.max(xy_points[:, 0])
            y_min, y_max = np.min(xy_points[:, 1]), np.max(xy_points[:, 1])
            ground_area = (x_max - x_min) * (y_max - y_min)

        # Calculate canopy surface area (using Delaunay triangulation)
        try:
            tri = Delaunay(xy_points)
            canopy_area = 0
            for simplex in tri.simplices:
                # Obtain the 3-vertices of the triangle
                p1, p2, p3 = points[simplex]
                # Calculate the area of a triangle
                area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
                canopy_area += area
        except:
            # If the triangular subdivision fails, use a simple approximation method.
            canopy_area = ground_area * 1.5  # rough estimate

        # Canopy surface ratio (Rumple)（Ratio of canopy surface area to ground projection area）
        rumple = canopy_area / ground_area if ground_area > 0 else 1.0

        return rumple

    def calculate_effective_number_of_layers(self, points, voxel_size):
        """Effective Layer Count (ENL)"""
        self.log_message("Effective Layer Count (ENL)...")

        # Obtain the Z-coordinate range of the point cloud
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

        # Create a vertical hierarchy
        z_bins = np.arange(z_min, z_max + voxel_size, voxel_size)

        # Calculate the point density for each stratum
        layer_density = np.zeros(len(z_bins) - 1)

        for k in range(len(z_bins) - 1):
            # Check whether there are canopy points within the current stratum.
            in_z = (points[:, 2] >= z_bins[k]) & (points[:, 2] < z_bins[k + 1])
            layer_density[k] = np.sum(in_z)

        # Effective Layer Count (ENL)
        # ENLBased on Shannon's diversity index
        p = layer_density / np.sum(layer_density)  # The proportion of each layer
        p = p[p > 0]  # Only consider the layer with points
        ENL = np.exp(-np.sum(p * np.log(p)))  # Shannon Index

        return ENL

    def visualize_results(self, df, output_folder):
        """Generate visual results"""
        try:
            self.log_message("Generate visual results...")

            # Create chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Canopy rugosity (CR)
            axes[0, 0].hist(df['Canopy_Rugosity_Rc'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Canopy Rugosity (Rc) Distribution')
            axes[0, 0].set_xlabel('Canopy Rugosity (Rc)')
            axes[0, 0].set_ylabel('Frequency')

            # 2. Canopy top rugosity (CTR)
            axes[0, 1].hist(df['Top_Rugosity_RT'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_title('Top Rugosity (RT) Distribution')
            axes[0, 1].set_xlabel('Top Rugosity (RT)')
            axes[0, 1].set_ylabel('Frequency')

            # 3. Canopy surface ratio (Rumple)
            axes[1, 0].hist(df['Rumple_Index'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_title('Rumple Index Distribution')
            axes[1, 0].set_xlabel('Rumple Index')
            axes[1, 0].set_ylabel('Frequency')

            # 4. Effective Layer Count (ENL)
            axes[1, 1].hist(df['Effective_Number_of_Layers_ENL'], bins=20, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 1].set_title('Effective Number of Layers (ENL) Distribution')
            axes[1, 1].set_xlabel('Effective Number of Layers (ENL)')
            axes[1, 1].set_ylabel('Frequency')

            plt.tight_layout()

            # Save Image
            output_filename = os.path.join(output_folder, "canopy_rugosity_analysis.png")
            plt.savefig(output_filename, dpi=300)
            plt.close()
            self.log_message(f"Visualization results have been saved as: {output_filename}")

            # Create a correlation matrix diagram
            fig, ax = plt.subplots(figsize=(10, 8))
            metrics = ['Canopy_Rugosity_Rc', 'Top_Rugosity_RT', 'Rumple_Index', 'Effective_Number_of_Layers_ENL']
            correlation_matrix = df[metrics].corr()

            im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(metrics)))
            ax.set_yticks(np.arange(len(metrics)))
            ax.set_xticklabels(['Rc', 'RT', 'Rumple', 'ENL'])
            ax.set_yticklabels(['Rc', 'RT', 'Rumple', 'ENL'])

            # Add numerical labels
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black")

            ax.set_title('Canopy Rugosity Metrics Correlation Matrix')
            fig.colorbar(im, ax=ax)
            plt.tight_layout()

            # Save the relevance matrix diagram
            corr_filename = os.path.join(output_folder, "canopy_rugosity_correlation.png")
            plt.savefig(corr_filename, dpi=600)
            plt.close()
            self.log_message(f"The correlation matrix diagram has been saved as: {corr_filename}")

        except Exception as e:
            self.log_message(f"An error occurred while generating the visualization results: {str(e)}\n{traceback.format_exc()}")


# MIAN
if __name__ == "__main__":
    root = tk.Tk()
    app = CanopyRugosityApp(root)
    root.mainloop()