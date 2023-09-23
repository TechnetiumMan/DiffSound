import open3d as o3d
import numpy as np

# Create a sphere
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20)
o3d.io.write_triangle_mesh("sphere.ply", sphere)
# sphere.compute_vertex_normals()

# # Generate a point cloud from the sphere's vertices
# sphere_points = np.asarray(sphere.vertices)

# # Create a TetraMesh by tetrahedralizing the point cloud
# tetra_mesh, _ = o3d.geometry.TetraMesh.create_from_point_cloud(sphere_points, size=0.02)

# # Visualize the TetraMesh (optional)
# o3d.visualization.draw_geometries([tetra_mesh])

# # Optionally, you can save the TetraMesh to a .ply file
# o3d.io.write_triangle_mesh("sphere_tetra_mesh.msh", tetra_mesh)