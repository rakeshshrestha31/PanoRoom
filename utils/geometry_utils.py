import numpy as np
import open3d as o3d
import trimesh

def create_spatial_quad_polygen(quad_vertices: np.array, 
                                normal: np.array = np.array([1,0,0]), 
                                camera_center: np.array = np.array([0, 0, 0]),
                                camera_rotation: np.array = np.eye(3)):
    """create a quad polygen for spatial mesh
    """
    if camera_center is None:
        camera_center = np.array([0, 0, 0])
    if camera_rotation is None:
        camera_rotation = np.eye(3)
    quad_vertices = (quad_vertices - camera_center)
    quad_vertices = np.dot(camera_rotation, quad_vertices.T).T
    quad_triangles = []
    triangle = np.array([[0, 2, 1], [2, 0, 3]])
    quad_triangles.append(triangle)

    quad_triangles = np.concatenate(quad_triangles, axis=0)

    mesh = trimesh.Trimesh(vertices=quad_vertices,
                           faces=quad_triangles,
                           vertex_normals=np.tile(normal, (4, 1)),
                           process=False)

    centroid = np.mean(quad_vertices, axis=0)
    # print(f'centroid: {centroid}')
    normal_point = centroid + np.array(normal) * 0.5
    # print(f'normal_point: {normal_point}')

    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    # pcd_o3d.points.append(normal_point)
    # pcd_o3d.points.append(centroid)

    # return mesh, pcd_o3d
    return mesh
