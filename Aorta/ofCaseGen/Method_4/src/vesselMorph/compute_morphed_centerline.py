import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional

def compute_slice_plane(point: np.ndarray, tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two orthogonal vectors that form a plane perpendicular to the tangent.
    
    Args:
        point: Point on the centerline (3,)
        tangent: Tangent vector at that point (3,)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two orthogonal vectors defining the plane
    """
    # Normalize tangent
    tangent = tangent / np.linalg.norm(tangent)
    
    # Find first perpendicular vector
    # First try using x-axis as reference
    ref = np.array([1, 0, 0])
    v1 = np.cross(tangent, ref)
    
    # If x-axis is parallel to tangent, use y-axis
    if np.linalg.norm(v1) < 1e-6:
        ref = np.array([0, 1, 0])
        v1 = np.cross(tangent, ref)
    
    v1 = v1 / np.linalg.norm(v1)
    
    # Find second perpendicular vector
    v2 = np.cross(tangent, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    return v1, v2

def find_slice_points(point: np.ndarray, 
                     v1: np.ndarray, 
                     v2: np.ndarray, 
                     vertices: np.ndarray, 
                     search_radius: float) -> np.ndarray:
    """
    Find vertices that lie near the plane defined by point and two vectors.
    
    Args:
        point: Point on the centerline
        v1, v2: Vectors defining the plane
        vertices: All vertices of the mesh
        search_radius: Maximum distance from plane to consider
        
    Returns:
        np.ndarray: Vertices that lie in the slice
    """
    # Compute plane normal
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Calculate distance of all vertices from the plane
    distances = np.abs(np.dot(vertices - point, normal))
    
    # Find vertices within the search radius of the plane
    slice_mask = distances < search_radius
    
    return vertices[slice_mask]

def compute_morphed_centerline(vertices: np.ndarray, 
                             original_centerline: np.ndarray,
                             inlet_patch: np.ndarray, 
                             outlet_patch: np.ndarray,
                             num_points: Optional[int] = None) -> np.ndarray:
    """
    Compute the centerline of a morphed geometry by finding centroids of 
    cross-sections perpendicular to the original centerline.
    
    Args:
        vertices: Array of vertex coordinates (n, 3)
        original_centerline: Original centerline points (m, 3)
        inlet_patch: Array of inlet face indices
        outlet_patch: Array of outlet face indices
        num_points: Optional number of points for resampling
        
    Returns:
        np.ndarray: New centerline points
    """
    if num_points is None:
        num_points = len(original_centerline)
    
    # Calculate tangents along original centerline
    tangents = np.zeros_like(original_centerline)
    tangents[1:-1] = (original_centerline[2:] - original_centerline[:-2]) / 2
    tangents[0] = original_centerline[1] - original_centerline[0]
    tangents[-1] = original_centerline[-1] - original_centerline[-2]
    
    # Normalize tangents
    norms = np.linalg.norm(tangents, axis=1)
    tangents = tangents / norms[:, np.newaxis]
    
    # Calculate average vessel radius (from inlet/outlet) for search radius
    inlet_vertices = vertices[np.unique(inlet_patch)]
    outlet_vertices = vertices[np.unique(outlet_patch)]
    inlet_radius = np.mean(np.linalg.norm(inlet_vertices - np.mean(inlet_vertices, axis=0), axis=1))
    outlet_radius = np.mean(np.linalg.norm(outlet_vertices - np.mean(outlet_vertices, axis=0), axis=1))
    search_radius = max(inlet_radius, outlet_radius) * 0.2  # 20% of max radius for slice thickness
    
    # Initialize array for new centerline points
    new_centerline = np.zeros_like(original_centerline)
    
    # Process each point along the original centerline
    for i, (point, tangent) in enumerate(zip(original_centerline, tangents)):
        # Get plane vectors
        v1, v2 = compute_slice_plane(point, tangent)
        
        # Find vertices in this slice
        slice_points = find_slice_points(point, v1, v2, vertices, search_radius)
        
        if len(slice_points) > 0:
            # Calculate centroid of slice points
            new_centerline[i] = np.mean(slice_points, axis=0)
        else:
            # If no points found, use original point
            new_centerline[i] = point
    
    # Smooth the centerline using cubic spline interpolation
    if num_points != len(original_centerline):
        t = np.linspace(0, 1, len(original_centerline))
        t_new = np.linspace(0, 1, num_points)
        
        # Create splines for each coordinate
        splines = [CubicSpline(t, new_centerline[:, i]) for i in range(3)]
        
        # Generate smoothed centerline
        new_centerline = np.column_stack([spline(t_new) for spline in splines])
    
    return new_centerline