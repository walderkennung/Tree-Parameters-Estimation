Step 1 – Single Tree Parameter Estimation (Segmented Crowns)

In the first step, tree parameters are estimated using segmented LiDAR crowns. Each crown is processed independently. The model represents the tree crown as a 3D spheroid defined by its center, horizontal radius, vertical radius, and foliage density.

Parameters are initialized from the spatial distribution of the LiDAR points belonging to that crown. A physics-informed likelihood model based on LiDAR attenuation (Beer–Lambert law) is used to explain the observed points. The parameters are optimized using maximum likelihood estimation with PyTorch.

The output is a GeoPackage containing estimated tree parameters for each segmented crown.

Step 2.1 – Multi-Tree Modeling in Unsegmented LiDAR

The second step extends the approach to unsegmented LiDAR scenes. Instead of fitting trees individually, the model estimates the parameters of multiple trees simultaneously within a spatial tile.

Each tree is represented by the same spheroid model as in Step 1. The total vegetation density in the scene is computed as the sum of contributions from all trees, and LiDAR returns are modeled using an occlusion-aware likelihood that accounts for canopy material above each point.

Tree centers are initialized from high canopy points using farthest-point sampling, and all parameters are optimized jointly.

Step 2.3 – Softmax Responsibilities for Crown Separation

To improve separation between neighboring trees, a modified version of the model introduces softmax responsibilities. Instead of allowing multiple trees to explain the same LiDAR point equally, trees compete for points through a soft assignment mechanism.

This encourages each LiDAR return to be primarily explained by one crown, which helps reduce duplicated tree detections and improves crown separation in dense forests.

Step 2.2 – Initialization from Step 1 Results

Another variation improves the multi-tree optimization by initializing tree centers using results from Step 1. Since Step 1 already provides reasonable estimates of tree positions, these values provide a better starting point for the multi-tree model.

This informed initialization helps the optimizer converge more reliably and reduces the risk of poor local minima.

 Evaluation

The estimated tree parameters were evaluated using both quantitative and qualitative methods:

Height validation against forest inventory data

Comparison of modeled crown top height with local canopy peaks

Spatial inspection of estimated tree centers using aerial imagery

These evaluations assess whether the model correctly estimates tree height, crown position, and spatial distribution.
