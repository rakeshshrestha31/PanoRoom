from typing import Dict, List
import numpy as np
import open3d as o3d
import trimesh
import fcl

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


def heading2rotmat(heading_angle_rad):
    rotmat = np.eye(3)
    cosval = np.cos(heading_angle_rad)
    sinval = np.sin(heading_angle_rad)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat

def convert_oriented_box_to_trimesh_fmt(box: Dict, scale: float=1.0, color_to_labels: Dict = None) -> trimesh.Trimesh:
    """ convert oriented box to mesh

    Args:
        box (Dict): 'id': , 'transform': , 'size': , 
        color_to_labels (Dict, optional): each category colors. Defaults to None.

    Returns:
        trimesh.Trimesh: mesh
    """
    box_sizes = np.array(box['size'])* scale
    transform_matrix = np.array(box["transform"]).reshape(4, 4)
    transform_matrix[:3, 3] = transform_matrix[:3, 3] * scale
    box_trimesh_fmt = trimesh.creation.box(box_sizes, transform_matrix)
    if color_to_labels is not None:
        labels_lst = list(color_to_labels.values())
        colors_lst = list(color_to_labels.keys())
        color = colors_lst[labels_lst.index(box['class'])]
    else:
        color = (np.random.random(3) * 255).astype(np.uint8).tolist()
        # pass
    box_trimesh_fmt.visual.face_colors = color
    return box_trimesh_fmt

def create_oriented_bboxes(scene_bbox: List[Dict], scale: float=1.0) -> trimesh.Trimesh:
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box, scale=scale))

    mesh_list = trimesh.util.concatenate(scene.dump())
    return mesh_list

def vis_scene_mesh(room_layout_mesh: trimesh.Trimesh,
                   obj_bbox_lst: List[Dict],
                   color_to_labels: Dict = None) -> trimesh.Trimesh:

    v_object_meshes = create_oriented_bboxes(obj_bbox_lst)
    scene_mesh = trimesh.util.concatenate([room_layout_mesh, v_object_meshes])
    return scene_mesh


def check_mesh_attachment(object_mesh: trimesh.Trimesh, room_mesh: trimesh.Trimesh):
    """check if the object mesh is attached with the room, i.e. the window/door mesh is collided with the room

    use FCL to detect attachment and collision, see https://github.com/BerkeleyAutomation/python-fcl/ for ref.
    Args:
        object_mesh (o3d.geometry.TriangleMesh): object mesh
        room_walls (o3d.geometry.TriangleMesh): room walls mesh

    Returns:
        bool: True if the object mesh is in the room
    """
    room = fcl.BVHModel()
    room.beginModel(len(room_mesh.vertices), len(room_mesh.faces))
    room.addSubModel(room_mesh.vertices, room_mesh.faces)
    room.endModel()

    window = fcl.BVHModel()
    window.beginModel(len(object_mesh.vertices), len(object_mesh.faces))
    window.addSubModel(object_mesh.vertices, object_mesh.faces)
    window.endModel()

    t1 = fcl.Transform(np.array([1, 0, 0, 0]), np.array([0, 0, 0]))
    t2 = fcl.Transform(np.array([1, 0, 0, 0]), np.array([0., 0, 0]))

    o1 = fcl.CollisionObject(room, t1)
    o2 = fcl.CollisionObject(window, t2)

    request = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    result = fcl.CollisionResult()
    ret = fcl.collide(o1, o2, request, result)
    return result.is_collision


def check_mesh_distance(object_mesh: trimesh.Trimesh, room_mesh: trimesh.Trimesh):
    """caculate distance between object mesh and room mesh, i.e. the window/door mesh is adjacent to the room but is not collided with the room.

    use FCL to detect attachment and collision, see https://github.com/BerkeleyAutomation/python-fcl/ for ref.
    Args:
        object_mesh (o3d.geometry.TriangleMesh): object mesh
        room_walls (o3d.geometry.TriangleMesh): room walls mesh

    Returns:
        float: distance
    """
    room = fcl.BVHModel()
    room.beginModel(len(room_mesh.vertices), len(room_mesh.faces))
    room.addSubModel(room_mesh.vertices, room_mesh.faces)
    room.endModel()

    window = fcl.BVHModel()
    window.beginModel(len(object_mesh.vertices), len(object_mesh.faces))
    window.addSubModel(object_mesh.vertices, object_mesh.faces)
    window.endModel()

    t1 = fcl.Transform(np.array([1, 0, 0, 0]), np.array([0, 0, 0]))
    t2 = fcl.Transform(np.array([1, 0, 0, 0]), np.array([0., 0, 0]))

    o1 = fcl.CollisionObject(room, t1)
    o2 = fcl.CollisionObject(window, t2)

    request = fcl.DistanceRequest()
    result = fcl.DistanceResult()
    ret = fcl.distance(o1, o2, request, result)
    return ret


def check_bbox_in_room(bbox: Dict, room_layout_mesh: trimesh.Trimesh, layout_bbox_min: np.array, layout_bbox_max: np.array, margin_dist: float=0.05):
    """ check if the 3D bbox is in the room

    Args:
        bbox (Dict): keys:{'id': , 'transform': , 'size': ,}
        room_layout_mesh (trimesh.Trimesh): room layout mesh
        layout_bbox_min (np.array): min corner of room layout
        layout_bbox_max (np.array): max corner of room layout
        margin_dist (float, optional): margin distance. Defaults to 0.05.

    Returns:
        bool: true or false
    """
    bbox_center = np.array([bbox['transform']]).reshape(4, 4)[0:3, 3]
    # if object bbox center is outside of room layout
    if bbox_center[0] < layout_bbox_min[0] or bbox_center[0] > layout_bbox_max[0] or \
        bbox_center[1] < layout_bbox_min[1] or bbox_center[1] > layout_bbox_max[1] or \
        bbox_center[2] < layout_bbox_min[2] or bbox_center[2] > layout_bbox_max[2]:
        obj_bbox_mesh = convert_oriented_box_to_trimesh_fmt(bbox)
        if check_mesh_attachment(obj_bbox_mesh, room_layout_mesh):
            # print(f'{bbox["class"]} is attached to room ')
            return True
        else:
            min_distance = check_mesh_distance(obj_bbox_mesh, room_layout_mesh)
            if min_distance < margin_dist:
                # print(f'{bbox["class"]} is close to room, distance {min_distance} ')
                return True
        return False
    else:
        return True