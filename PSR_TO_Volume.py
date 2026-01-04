import os
import numpy as np
import open3d as o3d
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import threading
import laspy
import time

#Parameters such as the depth of Poisson surface reconstruction can be customized as needed.
#Updated September 2025.
class PoissonReconstructionTool:
    def __init__(self, root):
        self.root = root
        self.root.title("PSR_TO_Volume")
        self.root.geometry("1000x700")

        # Initialization
        self.input_folder = tk.StringVar()
        self.output_csv = tk.StringVar(value="poisson_reconstruction_results.csv")
        self.depth = tk.IntVar(value=8)
        self.scale = tk.DoubleVar(value=1.1)
        self.linear_fit = tk.BooleanVar(value=False)
        self.n_threads = tk.IntVar(value=-1) 
        self.density_threshold = tk.DoubleVar(value=0.01)  # Density threshold for filtering Poisson reconstruction results
        # Result Storage
        self.results = []
        self.current_mesh = None

        self.create_widgets()

    def create_widgets(self):
        # Main
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Input Folder
        ttk.Label(main_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Scan", command=self.browse_input_folder).grid(row=0, column=2, padx=5)

        # Export CSV
        ttk.Label(main_frame, text="Export CSV:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_csv, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Scan", command=self.browse_output_csv).grid(row=1, column=2, padx=5)

        # Parameter Settings
        params_frame = ttk.LabelFrame(main_frame, text="PSRparameter", padding="10")
        params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(params_frame, text="Depth:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=1, to=15, textvariable=self.depth, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(params_frame, text="Scaling factor:").grid(row=0, column=2, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.scale, width=10).grid(row=0,
                                                                                                            column=3,
                                                                                                            padx=5)

        ttk.Label(params_frame, text="Number of threads:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=-1, to=64, textvariable=self.n_threads, width=10).grid(row=1, column=1, padx=5)
        ttk.Label(params_frame, text="(-1Use all threads)").grid(row=1, column=2, sticky=tk.W, pady=5)

        ttk.Label(params_frame, text="Density threshold:").grid(row=1, column=3, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=0.001, to=1.0, increment=0.001,
                    textvariable=self.density_threshold, width=10).grid(row=1, column=4, padx=5)

        ttk.Checkbutton(params_frame, text="Linear fitting", variable=self.linear_fit).grid(row=0, column=4, padx=5)
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Start", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Visual", command=self.visualize_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Visualize the grid", command=self.visualize_current_mesh).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_results).pack(side=tk.LEFT, padx=5)

        # Progress
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # Status Tag
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)

        # Results Table
        columns = ("File name", "Number", "Surface area", "Volume", "Processing time")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120)

        #Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        scrollbar.grid(row=6, column=2, sticky=(tk.N, tk.S), pady=10)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)

    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Select the folder containing the point cloud files")
        if folder:
            self.input_folder.set(folder)

    def browse_output_csv(self):
        file_path = filedialog.asksaveasfilename(
            title="Save results to a CSV file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            self.output_csv.set(file_path)

    def start_processing(self):
        if not self.input_folder.get():
            messagebox.showerror("Error", "Please select the input folder")
            return

        thread = threading.Thread(target=self.process_files)
        thread.daemon = True
        thread.start()

    def process_files(self):
        self.progress.start()
        self.status_label.config(text="Processing...")

        input_folder = self.input_folder.get()

        point_cloud_files = [f for f in os.listdir(input_folder)
                             if f.lower().endswith(('.las', '.laz', '.ply', '.pcd', '.xyz', '.pts', '.xyzn'))]

        if not point_cloud_files:
            self.status_label.config(text="No point cloud file")
            self.progress.stop()
            return

        self.results = []
        total_files = len(point_cloud_files)

        for i, filename in enumerate(point_cloud_files):
            try:
                # Update status and progress
                self.status_label.config(text=f"Processing {filename} ({i + 1}/{total_files})")
                self.progress['value'] = (i / total_files) * 100
                self.root.update_idletasks()

                # Start time recorded
                start_time = time.time()

                # Read LAS
                file_path = os.path.join(input_folder, filename)

                # Use different reading methods based on file extensions
                if filename.lower().endswith(('.las', '.laz')):
                    pcd = self.read_las_file(file_path)
                else:
                    pcd = o3d.io.read_point_cloud(file_path)

                if not pcd or not pcd.has_points():
                    continue

                # Estimating Normals (Normal information is required for Poisson reconstruction)
                pcd.estimate_normals()

                # Perform Poisson surface reconstruction
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd,
                    depth=self.depth.get(),
                    scale=self.scale.get(),
                    linear_fit=self.linear_fit.get(),
                    n_threads=self.n_threads.get()
                )

                # Apply density threshold filtering to the grid
                vertices_to_remove = densities < np.quantile(densities, self.density_threshold.get())
                mesh.remove_vertices_by_mask(vertices_to_remove)

                # Calculate surface area and volume
                surface_area = self.calculate_surface_area(mesh)
                volume = self.calculate_volume(mesh)

                # Record processing time
                processing_time = time.time() - start_time

                # Save
                result = {
                    "filename": filename,
                    "point_count": len(pcd.points),
                    "surface_area": surface_area,
                    "volume": volume,
                    "processing_time": processing_time
                }

                self.results.append(result)
                self.current_mesh = mesh  
                # Update the table
                self.root.after(0, self.update_table, result)

            except Exception as e:
                print(f"Processing documents {filename} error occurs: {str(e)}")
                self.status_label.config(text=f"Processing documents {filename} error occurs: {str(e)}")

        self.progress['value'] = 100
        self.status_label.config(text=f"Processing complete，Processing {len(self.results)} file")

    def read_las_file(self, file_path):
        """Read LAS files and convert them to Open3D point cloud format"""
        try:
            # Use laspy to read LAS files
            las = laspy.read(file_path)

            # Extract point coordinates
            points = np.vstack((las.x, las.y, las.z)).transpose()

            # Create an Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # If the LAS file contains color information, extract it as well.
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                # Normalized color values (LAS files typically use 16-bit color)
                colors = np.vstack((
                    las.red / 65535.0,
                    las.green / 65535.0,
                    las.blue / 65535.0
                )).transpose()
                pcd.colors = o3d.utility.Vector3dVector(colors)

            return pcd

        except Exception as e:
            print(f"Read LAS files {file_path} error occurs: {str(e)}")
            return None

    def calculate_surface_area(self, mesh):
        """Calculate the surface area of the mesh"""
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)

        # Calculate the area of each triangle.
        areas = []
        for tri in triangles:
            a, b, c = vertices[tri]
            ab = b - a
            ac = c - a
            area = 0.5 * np.linalg.norm(np.cross(ab, ac))
            areas.append(area)

        return sum(areas)

    def calculate_volume(self, mesh):
        """Calculate the volume of the mesh"""
        # Calculating Volume Using the Convex Hull Method
        vertices = np.asarray(mesh.vertices)
        if len(vertices) < 4:  # A convex hull requires at least four points.
            return 0.0

        try:
            hull = ConvexHull(vertices)
            return hull.volume
        except:
            return 0.0

    def update_table(self, result):
        """Update Results Table"""
        self.tree.insert("", "end", values=(
            result["filename"],
            result["point_count"],
            f"{result['surface_area']:.2f}",
            f"{result['volume']:.2f}",
            f"{result['processing_time']:.2f}s"
        ))

    def visualize_results(self):
        """Visualization of Poisson reconstruction results"""
        if not self.results:
            messagebox.showinfo("Information", "No results found")
            return

        # Create a Visualization Window
        vis_window = tk.Toplevel(self.root)
        vis_window.title("Poisson Reconstruction Statistical Results")
        vis_window.geometry("1000x600")

        # Create a matplotlib graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Extract data
        filenames = [r["filename"] for r in self.results]
        areas = [r["surface_area"] for r in self.results]
        volumes = [r["volume"] for r in self.results]

        # Plot a bar chart of surface area
        x_pos = np.arange(len(areas))
        ax1.bar(x_pos, areas, tick_label=filenames)
        ax1.set_title("Surface area")
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylabel("Surface area (units^2)")

        # Plot a volume bar chart
        ax2.bar(x_pos, volumes, tick_label=filenames)
        ax2.set_title("Volume")
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylabel("Volume (cubic units)")

        # Adjust the layout
        plt.tight_layout()

        # Embedding graphics into Tkinter windows
        canvas = FigureCanvasTkAgg(fig, master=vis_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def visualize_current_mesh(self):
        """Visualize the currently processed grid"""
        if self.current_mesh is None:
            messagebox.showinfo("Information", "No grid to display")
            return

        # Launch Open3D visualization in a new thread to avoid blocking the GUI.
        def visualize():
            o3d.visualization.draw_geometries([self.current_mesh])

        thread = threading.Thread(target=visualize)
        thread.daemon = True
        thread.start()

    def save_results(self):
        """Save results to a CSV file"""
        if not self.results:
            messagebox.showinfo("Information", "No results to save")
            return

        if not self.output_csv.get():
            self.browse_output_csv()

        if self.output_csv.get():
            df = pd.DataFrame(self.results)
            df.to_csv(self.output_csv.get(), index=False)
            messagebox.showinfo("Success", f"The results have been saved to {self.output_csv.get()}")


def main():
    root = tk.Tk()
    app = PoissonReconstructionTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()