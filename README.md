3D-ASV Algorithm & SOI Index Establishment and Construction Paradigm
The projects primarily come from the>
[Spatial occupancy index of tree crown: can provide new perspectives for quantifying structural complexity of individual tree crowns]
#############################################################################
# Key processes involved
<div align="center">
<img width="965" height="1355" alt="F2" src="https://github.com/user-attachments/assets/be953759-7812-409b-8ca3-a88e5d833d45" />
</div>

# 3D-ASV
3D-ASV is a novel, optimized algorithm for calculating tree crown volume based on LiDAR point clouds.
Copyright 2025-Present 3D-ASV
Team of Zhao YT, College of Forestry, Northeast Forestry University

## Python Required Dependency
import 【os numpy open3d o3d pandas tkinter filedialog ttk messagebox matplotlib.backends.backend_tkagg FigureCanvasTkAgg
matplotlib.pyplot  scipy.spatial ConvexHull threading laspy time traceback scipy.spatial Delaunay KDTree warnings BytesIO】

## Documentation
<div align="center">
<img width="1910" height="1355" alt="F3" src="https://github.com/user-attachments/assets/c4ca9de7-5e7e-4804-bdc4-4402114dba14" />
</div>

## Tools and Algorithms
-Step1. Surface reconstruction is performed based on the well-filtered point cloud of tree crowns and its surface area S_SR is calculated.
-Step2. Initializing alpha values and building 3D alpha shape models.
-Step3. Building the surface approximation function and calculating the optimal alpha value.
   Create a surface approximation function ∆ based on the optimal alpha value and set the iteration step to 0.05 for the alpha, where δ is the minimum value：
   <img width="743" height="96" alt="F4" src="https://github.com/user-attachments/assets/1dbc751b-7262-440b-8f64-d0c8a90cf47f" />
-Step4. Calculating the crown volume and optimizing the volume using point cloud voxelization. 
   <img width="547" height="284" alt="F5" src="https://github.com/user-attachments/assets/87cf9be8-a2b9-4339-812d-28f964b7872c" />
   
# SOI-Series Index
Combined the crown spatial distribution parameters (i.e., normalized point cloud density and intensity of crown, respectively) with the crown spatial structural complexity parameters (i.e., crown entropy, crown edge structure complexity) to derive the SOI index with mathematical concepts and physical mechanisms based on the gainful-adding theory.
## Documentation
<img width="1098" height="1351" alt="F6" src="https://github.com/user-attachments/assets/47d92590-ce71-4bb3-81b3-9ada823c0ea0" />
## Tools and Algorithms
---Normalized point cloud density---
The calculation formula for the normalized point cloud density is as follows:

---Normalized point cloud intensity---
The formula of the revised normalized point cloud intensity is as follows:

---Canopy Entropy & Crown Entropy---
The specific principle derivation of canopy entropy can be found in Liu et al.2022. 
### Liu X.Q. 2022. A novel entropy-based method to quantify forest canopy structural complexity from multiplatform lidar point clouds. Remote Sensing of Environment. 282, 113280.
The formula for calculating the canopy entropy part in SOI is:









