import os
import csv
import laspy
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
import threading
import datetime
import open3d as o3d


class CanopyVolumeCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Voxel_TO_CSC(plot&tree)")
        self.root.geometry("600x500")

        
        self.input_folder = tk.StringVar()
        self.output_csv = tk.StringVar()
        self.output_ply_folder = tk.StringVar()
        self.voxel_size = tk.DoubleVar(value=0.05)
        self.progress = tk.DoubleVar()
        self.status_text = tk.StringVar(value="Ready")

        self.create_widgets()

    def create_widgets(self):
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

      
        ttk.Label(main_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Scan", command=self.browse_input_folder).grid(row=0, column=2)

        
        ttk.Label(main_frame, text="Export CSV file:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_csv, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Scan", command=self.browse_output_csv).grid(row=1, column=2)

        
        ttk.Label(main_frame, text="Output PLY folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_ply_folder, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(main_frame, text="Scan", command=self.browse_output_ply_folder).grid(row=2, column=2)

        # Voxel Size Setting
        ttk.Label(main_frame, text="Voxel size (m):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.voxel_size, width=10).grid(row=3, column=1, sticky=tk.W, padx=5)

       
        ttk.Label(main_frame, text="Progress:").grid(row=4, column=0, sticky=tk.W, pady=10)
        ttk.Progressbar(main_frame, variable=self.progress, maximum=100).grid(row=4, column=1, columnspan=2,
                                                                              sticky=(tk.W, tk.E), pady=10)

    
        ttk.Label(main_frame, text="Status:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Label(main_frame, textvariable=self.status_text).grid(row=5, column=1, sticky=tk.W, padx=5)

        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=20)

        ttk.Button(button_frame, text="Start Calculating", command=self.start_calculation).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=10)

        
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Select the folder containing the LAS file")
        if folder:
            self.input_folder.set(folder)

    def browse_output_csv(self):
        file = filedialog.asksaveasfilename(
            title="Select to output CSV file",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if file:
            self.output_csv.set(file)

    def browse_output_ply_folder(self):
        folder = filedialog.askdirectory(title="Select the folder to save the PLY file.")
        if folder:
            self.output_ply_folder.set(folder)

    def start_calculation(self):
       
        if not self.input_folder.get():
            messagebox.showerror("Error", "Please select the input folder")
            return

        if not self.output_csv.get():
            messagebox.showerror("Error", "Please select to output a CSV file")
            return

        if not self.output_ply_folder.get():
            messagebox.showerror("Error", "Please select the output PLY folder")
            return

        if self.voxel_size.get() <= 0:
            messagebox.showerror("Error", "Voxel size greater than 0.")
            return

      
        thread = threading.Thread(target=self.calculate_volume)
        thread.daemon = True
        thread.start()

    def calculate_volume(self):
        try:
            self.status_text.set("Calculating...")

         
            input_path = Path(self.input_folder.get())
            las_files = list(input_path.glob("*.las")) + list(input_path.glob("*.laz"))

            if not las_files:
                self.status_text.set("Error: LAS/LAZ file not found")
                messagebox.showerror("Error", "The LAS/LAZ file was not found in the specified folder")
                return

           
            output_csv_path = Path(self.output_csv.get())
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)

            output_ply_path = Path(self.output_ply_folder.get())
            output_ply_path.mkdir(parents=True, exist_ok=True)

           
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['ID', 'Volume_voxel'])

            
            total_files = len(las_files)
            for idx, las_file in enumerate(las_files):
                try:
                 
                    progress = (idx / total_files) * 100
                    self.progress.set(progress)
                    self.status_text.set(f"Processing files {idx + 1}/{total_files}: {las_file.name}")

                    
                    with laspy.open(las_file) as f:
                        las = f.read()

                    
                    points = np.vstack((las.x, las.y, las.z)).transpose()

                    volume, voxel_grid, min_coords = self.calculate_voxel_volume(points, self.voxel_size.get())

                    self.save_voxel_grid(voxel_grid, min_coords, output_ply_path, las_file.stem, self.voxel_size.get())

                    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([las_file.stem, volume])

                except Exception as e:
                    print(f"Processing files {las_file} error occurs: {e}")
                    
                    continue

        
            self.progress.set(100)
            self.status_text.set(f"Processing complete! total of  {total_files} files have been processed")
            messagebox.showinfo("Complete", f"Processing complete! total of {total_files}  \n Save to: {output_csv_path}")

        except Exception as e:
            self.status_text.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Errors occurred: {str(e)}")

    def calculate_voxel_volume(self, points, voxel_size):
        
        try:
            
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)

           
            dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

           
            voxel_grid = np.zeros(dims, dtype=bool)

            indices = ((points - min_coords) / voxel_size).astype(int)

            indices = np.clip(indices, 0, np.array(dims) - 1)

            for idx in indices:
                voxel_grid[idx[0], idx[1], idx[2]] = True

            occupied_voxels = np.sum(voxel_grid)

            volume = occupied_voxels * (voxel_size ** 3)

            return volume, voxel_grid, min_coords
        except Exception as e:
            print(f"Error occurred: {e}")
            return 0.0, None, None

    def save_voxel_grid(self, voxel_grid, min_coords, output_folder, filename, voxel_size):
     
        try:
            if voxel_grid is None or min_coords is None:
                return

            dim_x, dim_y, dim_z = voxel_grid.shape

            vertices = []
            faces = []

            cube_vertices = np.array([
                [-0.5, -0.5, -0.5],  # 0
                [0.5, -0.5, -0.5],  # 1
                [0.5, 0.5, -0.5],  # 2
                [-0.5, 0.5, -0.5],  # 3
                [-0.5, -0.5, 0.5],  # 4
                [0.5, -0.5, 0.5],  # 5
                [0.5, 0.5, 0.5],  # 6
                [-0.5, 0.5, 0.5]  # 7
            ]) * voxel_size

            cube_faces = np.array([
                [0, 1, 2], [0, 2, 3],  
                [4, 7, 6], [4, 6, 5],  
                [0, 4, 5], [0, 5, 1],  
                [2, 6, 7], [2, 7, 3],  
                [0, 3, 7], [0, 7, 4],  
                [1, 5, 6], [1, 6, 2]  
            ])


            for i in range(dim_x):
                for j in range(dim_y):
                    for k in range(dim_z):
                        if voxel_grid[i, j, k]:
                           
                            center_x = min_coords[0] + (i + 0.5) * voxel_size
                            center_y = min_coords[1] + (j + 0.5) * voxel_size
                            center_z = min_coords[2] + (k + 0.5) * voxel_size

                           
                            current_vertices = cube_vertices + [center_x, center_y, center_z]

                           
                            vertex_offset = len(vertices)
                            vertices.extend(current_vertices)

                            
                            current_faces = cube_faces + vertex_offset
                            faces.extend(current_faces)

           
            vertices = np.array(vertices)
            faces = np.array(faces)

            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)

           
            mesh.compute_vertex_normals()

           
            output_path = output_folder / f"{filename}_voxel_grid.ply"
            o3d.io.write_triangle_mesh(str(output_path), mesh)

        except Exception as e:
            print(f"Error occurred: {e}")


def main():
    root = tk.Tk()
    app = CanopyVolumeCalculator(root)
    root.mainloop()


if __name__ == "__main__":
    main()