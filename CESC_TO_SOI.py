import os
import laspy
import numpy as np
import alphashape
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import trimesh
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
import time
import traceback


class AlphaShapeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CESC_TO_SOI")
        self.root.geometry("600x400")

      
        self.folder_path = tk.StringVar()
        self.alpha_value = tk.DoubleVar(value=5.0)
        self.progress = tk.DoubleVar()
        self.status_text = tk.StringVar(value="Ready")

        
        self.create_widgets()

    def create_widgets(self):
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Selection", padding="5")
        folder_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Entry(folder_frame, textvariable=self.folder_path, width=50).grid(row=0, column=0, padx=5)
        ttk.Button(folder_frame, text="Scan...", command=self.browse_folder).grid(row=0, column=1, padx=5)

        # Parameter Settings
        param_frame = ttk.LabelFrame(main_frame, text="Parameter Settings", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(param_frame, text="Alpha:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=0.1, to=100.0, increment=0.1,
                    textvariable=self.alpha_value, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

       
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(progress_frame, textvariable=self.status_text).grid(row=0, column=0, sticky=tk.W)
        ttk.Progressbar(progress_frame, variable=self.progress, maximum=100).grid(row=1, column=0, sticky=(tk.W, tk.E),
                                                                                  pady=5)

        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Begin processing", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

        # LOG
        log_frame = ttk.LabelFrame(main_frame, text="LOG", padding="5")
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
       
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_processing(self):
        
        folder_path = self.folder_path.get()
        alpha_value = self.alpha_value.get()

        if not folder_path:
            messagebox.showerror("Error", "Select the LAS file.")
            return

        
        thread = Thread(target=self.process_las_files, args=(folder_path, alpha_value))
        thread.daemon = True
        thread.start()

    def process_las_files(self, folder_path, alpha_value):
        
        try:
            self.status_text.set("Processing...")
            self.progress.set(0)

            las_files = [f for f in os.listdir(folder_path) if f.endswith('.las')]
            total_files = len(las_files)

            if total_files == 0:
                self.log_message("NO LAS file")
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

                   
                    alpha_shape = self.generate_alpha_shape(points, alpha_value)

                    if alpha_shape is None:
                        self.log_message(f"Unable to process {file_id} to Alpha Shape")
                        results.append({
                            'ID': file_id,
                            'NUM': -1  
                        })
                        continue

                    # Extract non-manifold vertices
                    non_manifold_vertices = self.extract_non_manifold_vertices(alpha_shape)

                    # Visualization results
                    self.visualize_results(points, alpha_shape, non_manifold_vertices, file_id, alpha_value,
                                           folder_path)

                    # Record the results
                    results.append({
                        'ID': file_id,
                        'NUM': len(non_manifold_vertices)
                    })

                    self.log_message(f"file {file_id} Processing complete，Found {len(non_manifold_vertices)} non-manifold")

                except Exception as e:
                    error_msg = f"Processing {file_id} error occurs: {str(e)}\n{traceback.format_exc()}"
                    self.log_message(error_msg)
                    results.append({
                        'ID': file_id,
                        'NUM': -1  
                    })

                processed_files += 1
                self.progress.set((processed_files / total_files) * 100)

            # Save results to a CSV file
            csv_path = os.path.join(folder_path, f"non_manifold_vertices_alpha_{alpha_value}.csv")
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)

            self.log_message(f"Results saved to {csv_path}")
            self.status_text.set("Processing complete")
            messagebox.showinfo("Completed", f"Processing complete！Saved to {csv_path}")

        except Exception as e:
            error_msg = f"Error occurred: {str(e)}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            self.status_text.set("Error occurred")
            messagebox.showerror("Error", f"Error occurred: {str(e)}")

    def read_las_file(self, file_path):
        
        self.log_message(f"Reading file: {file_path}")
        las = laspy.read(file_path)

       
        points = np.vstack((las.x, las.y, las.z)).transpose()

        
        if points.ndim == 1:
            self.log_message("Warning")
            
            points = points.reshape(-1, 3)

        self.log_message(f"Read {len(points)} points，shape: {points.shape}")
        return points

    def generate_alpha_shape(self, points, alpha):
       
        try:
            self.log_message(f"Generating Alpha Shape value set to: {alpha}")

            
            if points.ndim == 1:
                points = points.reshape(-1, 3)

            
            if len(points) < 4:
                self.log_message("Warning:Insufficient point cloud data")
                return None

            alpha_shape = alphashape.alphashape(points, alpha)
            return alpha_shape
        except Exception as e:
            self.log_message(f"Error occurred: {str(e)}")
            return None

    def extract_non_manifold_vertices(self, alpha_shape):
        """Extract non-manifold vertices"""
        try:
            self.log_message("Extract non-manifold vertices...")

            
            if alpha_shape is None:
                self.log_message("Warning: Alpha Shape is Null")
                return np.array([])

            
            if hasattr(alpha_shape, 'vertices') and hasattr(alpha_shape, 'faces'):
                vertices = alpha_shape.vertices
                faces = alpha_shape.faces

             
                if vertices.ndim == 1:
                    vertices = vertices.reshape(-1, 3)
            else:
               
                try:
                    mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces)
                    vertices = mesh.vertices
                    faces = mesh.faces
                except Exception as e:
                    self.log_message(f"Error occurred: {str(e)}")
                    return np.array([])

            
            vertex_faces = {}
            for i, face in enumerate(faces):
               
                if hasattr(face, '__len__') and len(face) >= 3:
                    for vertex in face[:3]:  
                        if vertex not in vertex_faces:
                            vertex_faces[vertex] = []
                        vertex_faces[vertex].append(i)

            # Detecting non-manifold vertices
            non_manifold_vertices = []
            for vertex, face_list in vertex_faces.items():
               
                if vertex >= len(vertices):
                    continue

                
                normals = []
                for face_idx in face_list:
                    if face_idx >= len(faces):
                        continue

                    face_vertices = faces[face_idx]
                   
                    if len(face_vertices) < 3:
                        continue

                    
                    if any(v >= len(vertices) for v in face_vertices[:3]):
                        continue

                    vec1 = vertices[face_vertices[1]] - vertices[face_vertices[0]]
                    vec2 = vertices[face_vertices[2]] - vertices[face_vertices[0]]
                    normal = np.cross(vec1, vec2)
                    normals.append(normal / np.linalg.norm(normal))

               
                if len(normals) > 1:
                    is_non_manifold = False
                    for i in range(len(normals) - 1):
                        dot_product = np.dot(normals[i], normals[i + 1])
                        if dot_product < 0.7:  
                            is_non_manifold = True
                            break

                    if is_non_manifold:
                        non_manifold_vertices.append(vertices[vertex])

            return np.array(non_manifold_vertices)
        except Exception as e:
            self.log_message(f"Error occurred: {str(e)}\n{traceback.format_exc()}")
            return np.array([])

    def visualize_results(self, points, alpha_shape, non_manifold_vertices, filename, alpha_value, output_folder):
        """Visualization results"""
        try:
            self.log_message("Visualization results...")
            fig = plt.figure(figsize=(15, 10))

            
            ax1 = fig.add_subplot(121, projection='3d')

            
            if points.ndim == 1:
                points = points.reshape(-1, 3)

            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b', alpha=0.5, label='点云')
            ax1.set_title(f' Raw point cloud: {filename}')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()

            # Alpha Shapes and Non-Manifold Vertices
            ax2 = fig.add_subplot(122, projection='3d')

            # Alpha Shape
            if alpha_shape is not None and hasattr(alpha_shape, 'vertices') and hasattr(alpha_shape, 'faces'):
                vertices = alpha_shape.vertices
                faces = alpha_shape.faces

             
                if vertices.ndim == 1:
                    vertices = vertices.reshape(-1, 3)

                mesh = Poly3DCollection(vertices[faces], alpha=0.3, edgecolor='red')
                mesh.set_facecolor('red')
                ax2.add_collection3d(mesh)

         
            if len(non_manifold_vertices) > 0:
              
                if non_manifold_vertices.ndim == 1:
                    non_manifold_vertices = non_manifold_vertices.reshape(-1, 3)

                ax2.scatter(non_manifold_vertices[:, 0],
                            non_manifold_vertices[:, 1],
                            non_manifold_vertices[:, 2],
                            s=50, c='yellow', marker='o', label='Non-Manifold Vertices')

            ax2.set_title(f'Alpha Shape (α={alpha_value}) and Non-Manifold Vertices')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.legend()

            
            for ax in [ax1, ax2]:
                ax.set_xlim([np.min(points[:, 0]), np.max(points[:, 0])])
                ax.set_ylim([np.min(points[:, 1]), np.max(points[:, 1])])
                ax.set_zlim([np.min(points[:, 2]), np.max(points[:, 2])])

            plt.tight_layout()

            # Save Image
            output_filename = os.path.join(output_folder, f"{filename}_alpha_{alpha_value}.png")
            plt.savefig(output_filename, dpi=300)
            plt.close()
            self.log_message(f"Visualization results saved as: {output_filename}")
        except Exception as e:
            self.log_message(f"Visualization error: {str(e)}\n{traceback.format_exc()}")


# main
if __name__ == "__main__":
    root = tk.Tk()
    app = AlphaShapeApp(root)
    root.mainloop()