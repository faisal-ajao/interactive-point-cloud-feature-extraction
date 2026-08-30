import time
import yaml
import pyvista as pv
import numpy as np
from scipy.spatial import KDTree

# ======================================================
# Load Configuration
# ======================================================
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

input_path = config["input_path"]
merge_tolerance = config["merge_tolerance"]
num_neighbors = config["knn"]
segmentation_feature = config["segmentation_feature"]
point_size = config["point_size"]
segmented_pcd_output = config["output_path"]

# ======================================================
# Load and Clean Point Cloud
# ======================================================
point_cloud = pv.read(input_path)

point_cloud_cleaned = point_cloud.clean(merge_tol=merge_tolerance)

print(f"Number of points after cleaning: {len(point_cloud_cleaned.points)}")

# ======================================================
# Visualize Cleaned Point Cloud
# ======================================================
pv.plot(point_cloud_cleaned,
        render_points_as_spheres=True,
        point_size=point_size,
        rgb=True)

# ======================================================
# Build KD-Tree and Compute Point Neighbours
# ======================================================
kd_tree = KDTree(point_cloud_cleaned.points)

start_time = time.time()

_, neighbor_indices = kd_tree.query(
    point_cloud_cleaned.points,
    k=num_neighbors
)

point_neighbors = point_cloud_cleaned.points[neighbor_indices]

end_time = time.time()

print(f"Neighbor Computation in {end_time-start_time} seconds")

# ======================================================
# Principal Component Analysis (PCA)
# ======================================================
def compute_pca(points):
    mean = np.mean(points, axis=0)
    centered_data = points - mean
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors from largest to smallest
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]

    return sorted_eigenvalues, sorted_eigenvectors

# ======================================================
# Extract Geometric Features from PCA
# ======================================================
def extract_features(eigenvalues, eigenvectors):
    # Prevent small negative values caused by numerical precision
    eigenvalues = np.maximum(eigenvalues, 0)

    # Handle degenerate neighbourhoods with no meaningful variance
    if eigenvalues[0] <= 0:
        return np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan

    # Calculate local geometric descriptors from the eigenvalues
    planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
    linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
    omnivariance = (eigenvalues[0] * eigenvalues[1] * eigenvalues[2]) ** (1/3)

    # The eigenvector associated with the smallest eigenvalue
    # represents the estimated surface normal
    _, _, normal = eigenvectors

    # Measure how strongly the surface normal deviates from the vertical axis
    verticality = 1 - abs(normal[2])

    return (
        planarity,
        linearity,
        omnivariance,
        verticality,
    )

# ======================================================
# Initialize Feature Storage
# ======================================================
feature_dtype = {
    "names": [
        "planarity",
        "linearity",
        "omnivariance",
        "verticality",
    ],
    "formats": [
        float,
        float,
        float,
        float,
    ]
}

features = np.empty(
    len(point_cloud_cleaned.points),
    dtype=feature_dtype
)

features[:] = np.nan

num_points = len(point_cloud_cleaned.points)

# ======================================================
# Compute Features for the Entire Point Cloud
# ======================================================
start_time = time.time()

for point_index in range(num_points):
    eigenvalues, eigenvectors = compute_pca(point_neighbors[point_index])

    (
        features["planarity"][point_index],
        features["linearity"][point_index],
        features["omnivariance"][point_index],
        features["verticality"][point_index],
    ) = extract_features(
        eigenvalues,
        eigenvectors
    )

# ======================================================
# Normalize Omnivariance
# ======================================================
min_omni = np.nanmin(features["omnivariance"])
max_omni = np.nanmax(features["omnivariance"])

if max_omni > min_omni:
    features["omnivariance"] = (features["omnivariance"] - min_omni) / (max_omni - min_omni)
else:
    features["omnivariance"] = 0.0

end_time = time.time()

print(
    f"Full Point Cloud Feature Computation in "
    f"{end_time-start_time} seconds"
)

# ======================================================
# Assign Feature to Point Cloud
# ======================================================
point_cloud_cleaned[segmentation_feature] = (features[segmentation_feature])

# ======================================================
# Initialize Interactive Segmentation
# ======================================================
plotter = pv.Plotter()

initial_threshold = 0.5

mask = point_cloud_cleaned[segmentation_feature] >= initial_threshold

segmented_point_cloud = (point_cloud_cleaned.extract_points(mask).cast_to_poly_points())

segmentation_actor = plotter.add_mesh(
    segmented_point_cloud,
    scalars=segmentation_feature,
    render_points_as_spheres=True,
    point_size=point_size
)

# ======================================================
# Update Segmentation from Threshold Slider
# ======================================================
def update_slider(threshold):
    global segmented_point_cloud

    mask = (point_cloud_cleaned[segmentation_feature] >= threshold)

    segmented_point_cloud = (point_cloud_cleaned.extract_points(mask).cast_to_poly_points())

    segmentation_actor.mapper.dataset = segmented_point_cloud

# ======================================================
# Configure Feature Threshold Range
# ======================================================
feature_min = np.nanmin(point_cloud_cleaned[segmentation_feature])

feature_max = np.nanmax(point_cloud_cleaned[segmentation_feature])

plotter.add_slider_widget(
    update_slider,
    [feature_min, feature_max],
    value=initial_threshold,
    title=segmentation_feature
)

# ======================================================
# Interactive Feature-Based Segmentation
# ======================================================
plotter.show()

# ======================================================
# Export Segmented Point Cloud
# ======================================================
segmented_point_cloud.save(
    segmented_pcd_output,
    texture="RGB"
)
