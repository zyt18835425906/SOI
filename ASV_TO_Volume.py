import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import laspy
import open3d as o3d
from scipy.spatial import KDTree
import trimesh
import os
import threading


class VoxelOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASV_TO_Volume")
        self.root.geometry("600x500")

       
        self.point_cloud_path = tk.StringVar()
        self.alpha_shape_path = tk.StringVar()
        self.voxel_model_path = tk.StringVar()
        self.output_path = tk.StringVar()

        self.setup_ui()

    def setup_ui(self):
       
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        
        title_label = ttk.Label(main_frame, text="ASV_TO_Volume",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        
        self.create_file_selection(main_frame)

      
        self.create_parameter_section(main_frame)

        
        self.create_button_section(main_frame)

      
        self.create_progress_section(main_frame)

    def create_file_selection(self, parent):
        file_frame = ttk.LabelFrame(parent, text="File Options", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

      
        ttk.Label(file_frame, text="Point cloud file (.las):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.point_cloud_path, width=50).grid(row=0, column=1, pady=2)
        ttk.Button(file_frame, text="Scan", command=self.browse_point_cloud).grid(row=0, column=2, pady=2)

       
        ttk.Label(file_frame, text="Alpha Shape (.ply):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.alpha_shape_path, width=50).grid(row=1, column=1, pady=2)
        ttk.Button(file_frame, text="Scan", command=self.browse_alpha_shape).grid(row=1, column=2, pady=2)

       
        ttk.Label(file_frame, text="Voxel (.ply):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.voxel_model_path, width=50).grid(row=2, column=1, pady=2)
        ttk.Button(file_frame, text="Scan", command=self.browse_voxel_model).grid(row=2, column=2, pady=2)

        
        ttk.Label(file_frame, text="Output Path:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.output_path, width=50).grid(row=3, column=1, pady=2)
        ttk.Button(file_frame, text="Scan", command=self.browse_output).grid(row=3, column=2, pady=2)

    def create_parameter_section(self, parent):
        param_frame = ttk.LabelFrame(parent, text="Optimized Parameter", padding="10")
        param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        
        ttk.Label(param_frame, text="Voxel size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.voxel_size = tk.StringVar(value="0.1")
        ttk.Entry(param_frame, textvariable=self.voxel_size, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(param_frame, text="M").grid(row=0, column=2, sticky=tk.W, pady=2)

       
        ttk.Label(param_frame, text="Boundary threshold:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.border_threshold = tk.StringVar(value="0.05")
        ttk.Entry(param_frame, textvariable=self.border_threshold, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Label(param_frame, text="M").grid(row=1, column=2, sticky=tk.W, pady=2)

    def create_button_section(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Start optimization", command=self.start_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

    def create_progress_section(self, parent):
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        
        self.log_text = tk.Text(progress_frame, height=8, width=70)
        self.log_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        
        scrollbar = ttk.Scrollbar(progress_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def browse_point_cloud(self):
        filename = filedialog.askopenfilename(filetypes=[("LAS files", "*.las")])
        if filename:
            self.point_cloud_path.set(filename)

    def browse_alpha_shape(self):
        filename = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        if filename:
            self.alpha_shape_path.set(filename)

    def browse_voxel_model(self):
        filename = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        if filename:
            self.voxel_model_path.set(filename)

    def browse_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY files", "*.ply")])
        if filename:
            self.output_path.set(filename)

    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def reset(self):
        self.point_cloud_path.set("")
        self.alpha_shape_path.set("")
        self.voxel_model_path.set("")
        self.output_path.set("")
        self.voxel_size.set("0.1")
        self.border_threshold.set("0.05")
        self.log_text.delete(1.0, tk.END)
        self.progress['value'] = 0

    def is_point_inside_mesh(self, point, mesh):
        """Judge points inside the grid based on ray tracing"""
        try:
            # Judge points inside the grid based on ray tracing
            ray_directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                                       [0, -1, 0], [0, 0, 1], [0, 0, -1]])

            intersection_counts = 0
            for direction in ray_directions:
                ray_origin = point - direction * 0.001  
                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=[ray_origin],
                    ray_directions=[direction]
                )
                intersection_counts += len(locations) % 2  

            return intersection_counts > 3  
        except:
            
            distance = mesh.nearest.signed_distance([point])
            return distance[0] > -float(self.border_threshold.get())

    def optimize_voxels(self):
        try:
            self.log_message("Start the optimization process...")
            self.progress['value'] = 10

            
            self.log_message("Read point cloud data...")
            if self.point_cloud_path.get():
                las = laspy.read(self.point_cloud_path.get())
                points = np.vstack((las.x, las.y, las.z)).transpose()
            else:
                points = np.array([])

            self.progress['value'] = 20

           
            self.log_message("Read the Alpha Shape...")
            alpha_mesh = trimesh.load_mesh(self.alpha_shape_path.get())
            self.log_message(f"Alpha ShapeNumber of model vertices: {len(alpha_mesh.vertices)}")
            self.log_message(f"Alpha ShapeNumber of model faces: {len(alpha_mesh.faces)}")

            self.progress['value'] = 40

            
            self.log_message("Read voxel...")
            voxel_mesh = trimesh.load_mesh(self.voxel_model_path.get())
            self.log_message(f"Number of vertices in voxel: {len(voxel_mesh.vertices)}")
            self.log_message(f"Number of faces in voxel: {len(voxel_mesh.faces)}")

            self.progress['value'] = 60

           
            self.log_message("Extract voxel center points...")
            voxel_centers = []
            voxel_faces = voxel_mesh.faces

            
            face_groups = {}
            for face in voxel_faces:
                
                center = np.mean(voxel_mesh.vertices[face], axis=0)
                center_key = tuple(np.round(center, 3))
                if center_key not in face_groups:
                    face_groups[center_key] = center

            voxel_centers = list(face_groups.values())
            self.log_message(f"Number of voxels detected: {len(voxel_centers)}")

            self.progress['value'] = 70

           
            self.log_message("Start voxel optimization...")
            preserved_centers = []
            deleted_count = 0
            total_voxels = len(voxel_centers)

            for i, center in enumerate(voxel_centers):
                if self.is_point_inside_mesh(center, alpha_mesh):
                    deleted_count += 1
                else:
                    preserved_centers.append(center)

                
                if i % 100 == 0:
                    progress_pct = 70 + (i / total_voxels) * 25
                    self.progress['value'] = progress_pct
                    self.log_message(f"Processing Progress: {i}/{total_voxels}")

            preserved_count = len(preserved_centers)
            self.log_message(f"Optimization Complete - Delete Voxels: {deleted_count}, retaining voxels: {preserved_count}")

            self.progress['value'] = 95

            
            self.log_message("Generate optimized voxel ...")
            if preserved_centers:
               
                optimized_points = np.array(preserved_centers)

                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(optimized_points)

                # Voxel-based display
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                    pcd, voxel_size=float(self.voxel_size.get()))

                # Save the optimized voxel model
                o3d.io.write_point_cloud(self.output_path.get(), pcd)
                self.log_message(f"optimized model saved: {self.output_path.get()}")
            else:
                self.log_message("Warning: No voxels were retained！")

            # Save statistics to CSV
            self.log_message("Generate statistical reports...")
            stats_data = {
                'N_SUM': [total_voxels],
                'N_delete': [deleted_count],
                'N_couple': [preserved_count]
            }

            stats_df = pd.DataFrame(stats_data)
            csv_path = os.path.splitext(self.output_path.get())[0] + '_stats.csv'
            stats_df.to_csv(csv_path, index=False)
            self.log_message(f"The statistical report saved: {csv_path}")

            self.progress['value'] = 100
            self.log_message("Optimization process completed!")

            messagebox.showinfo("Completed", "Optimization process completed!")

        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Errors occurred: {str(e)}")

    def start_optimization(self):
        
        if not self.alpha_shape_path.get():
            messagebox.showerror("Error", "Please select the Alpha Shape model file")
            return

        if not self.voxel_model_path.get():
            messagebox.showerror("Error", "Please select the voxel model file")
            return

        if not self.output_path.get():
            messagebox.showerror("Error", "Please select the output file path")
            return

        
        thread = threading.Thread(target=self.optimize_voxels)
        thread.daemon = True
        thread.start()


def main():
    root = tk.Tk()
    app = VoxelOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()