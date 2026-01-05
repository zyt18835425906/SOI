3D-ASV Algorithm & SOI Index Establishment and Construction Paradigm
The projects primarily come from the>
>[Spatial occupancy index of tree crown: can provide new perspectives
>for quantifying structural complexity of individual tree crowns]
## Key processes involved
<div align="center">
<img width="483" height="678" alt="F2" src="https://github.com/user-attachments/assets/be953759-7812-409b-8ca3-a88e5d833d45" />
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
<img width="955" height="678" alt="F3" src="https://github.com/user-attachments/assets/c4ca9de7-5e7e-4804-bdc4-4402114dba14" />
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
<img width="549" height="676" alt="F6" src="https://github.com/user-attachments/assets/47d92590-ce71-4bb3-81b3-9ada823c0ea0" />

## Tools and Algorithms

---Normalized point cloud density---

-The calculation formula for the normalized point cloud density is as follows:

<img width="352" height="111" alt="F7" src="https://github.com/user-attachments/assets/4f50f8c7-550f-4497-86cf-a3536a2bf2a7" />

---Normalized point cloud intensity---

-The formula of the revised normalized point cloud intensity is as follows:

<img width="299" height="111" alt="F8" src="https://github.com/user-attachments/assets/0a25edb4-eeb9-4a6f-bead-5c2cd0276810" />

---Canopy Entropy & Crown Entropy---

-The specific principle derivation of canopy entropy can be found in Liu et al.2022. 
Liu X.Q. 2022. A novel entropy-based method to quantify forest canopy structural complexity from multiplatform lidar point clouds. Remote Sensing of Environment. 282, 113280.

-The formula for calculating the canopy entropy part in SOI is:

<img width="1081" height="126" alt="F9" src="https://github.com/user-attachments/assets/bde97626-ee74-4e2d-bb4b-bbbca545b4e2" />

---Crown edge structural complexity---

-Step1:Initializing the tetrahedra. Building a set of tetrahedron based on the 3Dalpha shape model,
        ∑▒i (i=0,1,2…n), where the circumsphere of tetrahedron with radius α_optimal.
-Step2:Creating topological connections and generating crown outlines.

-Step3:Extracting critical vertices and calculating CESC.

## Building the SOI Index Series

-The four combined formula for calculating the spatial occupancy index of the tree crowns based on the crown volume is as follows:

<img width="571" height="309" alt="F10" src="https://github.com/user-attachments/assets/c936863b-0258-4540-be83-ac2231a18f1a" />

# Known Issues
-This software is currently in development.







