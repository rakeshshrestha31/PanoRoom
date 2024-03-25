from collections import defaultdict
from copy import deepcopy
import sys
import json
from pathlib import Path
import os
import os.path as osp
from typing import List, Dict

import numpy as np
from PIL import Image
import torch

import open3d as o3d
import trimesh
from transforms3d import utils, axangles

from geometry_utils import create_spatial_quad_polygen
from scripts.preprocess_koolai_data import parse_user_output, adjust_cam_meta, parse_room_meta


SCALE = 0.001
SCALE_MAT = np.eye(4)
SCALE_MAT[range(3), range(3)] = SCALE


def getCameraT(camera_meta_dict):
    camera_position = camera_meta_dict["camera_position"]
    camera_look_at = camera_meta_dict["camera_look_at"]
    camera_up = camera_meta_dict["camera_up"]

    camera_position = [camera_position["x"], camera_position["y"], camera_position["z"]]
    camera_look_at = [camera_look_at["x"], camera_look_at["y"], camera_look_at["z"]]
    camera_up = [camera_up["x"], camera_up["y"], camera_up["z"]]

    camera_position = np.array(camera_position)
    camera_look_at = np.array(camera_look_at)
    camera_up = np.array(camera_up)

    x_look_at = camera_look_at - camera_position
    z_look_up = camera_up

    y_look = np.cross(z_look_up, x_look_at)
    camera_R = np.zeros(9).reshape(3,3)
    camera_R[:,0] = utils.normalized_vector(x_look_at)
    camera_R[:,1] = utils.normalized_vector(y_look)
    camera_R[:,2] = utils.normalized_vector(z_look_up)
    camera_T = camera_position
    T = np.eye(4)
    T[:3,:3] = camera_R
    T[:3,3] = camera_T

    return T


def get_room_lineset(room_meta_dict: Dict[str, List[Dict[str, float]]]) -> o3d.geometry.LineSet:
    lineset = o3d.geometry.LineSet()
    points = []
    for edge in room_meta_dict['edge']:
        start = edge['start']
        start = [start['x'], start['y'], start['z']]
        end = edge['end']
        end = [end['x'], end['y'], end['z']]
        points.append(start)
        points.append(end)

    points = np.array(points)
    lines = np.asarray([[2 * i, 2 * i + 1] for i in range(points.shape[0] // 2)])
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)

    return lineset


def get_room_bounds(room_meta):
    room_corners = []
    inds = []
    for point in room_meta['edge']:
        room_corners.append([point['start']['x'], point['start']['y'], point['start']['z']])
        room_corners.append([point['end']['x'], point['end']['y'], point['end']['z']])
    room_corners = np.array(room_corners)
    bbox_min = np.min(room_corners, axis=0)
    bbox_max = np.max(room_corners, axis=0)
    return bbox_min, bbox_max


def interpolate_line(p1, p2, num=30):
    t = np.expand_dims(np.linspace(0, 1, num=num, dtype=np.float32), 1)
    points = p1 * (1 - t) + t * p2
    return points


def cam3d2rad(cam3d):
    """
    Transform 3D points in camera coordinate to longitude and latitude.

    Parameters
    ----------
    cam3d: n x 3 numpy array or bdb3d dict

    Returns
    -------
    n x 2 numpy array of longitude and latitude in radiation
    first rotate left-right, then rotate up-down
    longitude: (left) -pi -- 0 --> +pi (right)
    latitude: (up) -pi/2 -- 0 --> +pi/2 (down)
    """
    backend, atan2 = (torch, torch.atan2) if isinstance(cam3d, torch.Tensor) else (np, np.arctan2)
    lon = atan2(cam3d[..., 0], cam3d[..., 2])
    lat = backend.arcsin(cam3d[..., 1] / backend.linalg.norm(cam3d, axis=-1))
    return backend.stack([lon, lat], -1)


def camrad2pix(camrad, width, height):
    """
    Transform longitude and latitude of a point to panorama pixel coordinate.

    Parameters
    ----------
    camrad: n x 2 numpy array

    Returns
    -------
    n x 2 numpy array of xy coordinate in pixel
    x: (left) 0 --> (width - 1) (right)
    y: (up) 0 --> (height - 1) (down)
    """
    if isinstance(camrad, torch.Tensor):
        campix = torch.empty_like(camrad, dtype=torch.float32)
    else:
        campix = np.empty_like(camrad, dtype=np.float32)
    campix[..., 0] = camrad[..., 0] * width / (2. * np.pi) + width / 2. + 0.5
    campix[..., 1] = camrad[..., 1] * height / np.pi + height / 2. + 0.5
    return campix


def cam3d2pix(cam3d, width, height):
    """
    Transform 3D points from camera coordinate to pixel coordinate.

    Parameters
    ----------
    cam3d: n x 3 numpy array or bdb3d dict

    Returns
    -------
    for 3D points: n x 2 numpy array of xy in pixel.
    x: (left) 0 --> width - 1 (right)
    y: (up) 0 --> height - 1 (down)
    """
    campix = camrad2pix(cam3d2rad(cam3d), width, height)
    return campix


def obbs_to_pix(bboxes, width, height, fov=None, num=300):
    cam3d = []
    for bbox in bboxes:
        T = np.array(bbox["transform"]).reshape(4, 4)
        size = np.array(bbox["size"])
        obb = o3d.geometry.OrientedBoundingBox(T[:3, 3:4], T[:3, :3], size)
        points = obb.get_box_points()
        box_lines = [
            [0, 1], [1, 7], [7, 2], [2, 0],
            [3, 6], [6, 4], [4, 5], [5, 3],
            [0, 3], [2, 5], [1, 6], [7, 4]
        ]
        for line in box_lines:
            p1 = points[line[0]]
            p2 = points[line[1]]
            cam3d.extend(interpolate_line(p1, p2, num=num).tolist())

    cam3d = np.array(cam3d)
    if fov is not None:
        fov = np.radians(fov)
        focal = width / (2 * np.tan(fov / 2))
        K = np.array([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1]
        ])
        pix = cam3d @ K.T
        pix = pix[:, :2] / pix[:, 2:]
    else:
        pix = cam3d2pix(cam3d, width, height)
    # clip
    pix = np.clip(pix, 0, [width - 1, height - 1])
    pix = pix.astype(int)

    return pix


scene_folderpath = Path("/mnt/Exp/rakesh/LRM_pano/20240229_data/3FO4K5GA97T0/")
ins_meta_file = scene_folderpath / "ins_meta.json"
structure_json_path = next(scene_folderpath.glob("*_structure/user_output.json"))
render_paths = scene_folderpath.glob("*_render/*")
render_path = next(render_paths).parents[0]
output_dir = os.path.join(str(scene_folderpath), 'panorama')
# cam_idx = 15
# camera_id_str = f"629_{cam_idx}"

with open(ins_meta_file, "r") as f:
    ins_meta = json.load(f)

mesh = o3d.geometry.TriangleMesh()
obbs = []
for bbox in ins_meta:
    T = np.array(bbox["transform"]).reshape(4, 4)
    size = np.array(bbox["size"])
    obb = trimesh.creation.box(extents=size, transform=T)
    obbs.append(obb)

obbs = trimesh.util.concatenate(obbs)
obbs.apply_transform(SCALE_MAT)
obbs.export(f'/tmp/bbox_all.ply')

meta_data_dict = parse_user_output(structure_json_path)

room_meta_dict = {}
for room_meta in meta_data_dict['room_meta']:
    lineset = get_room_lineset(room_meta)
    room_meta['lineset'] = lineset
    room_id_str = room_meta['id']
    room_meta_dict[room_id_str] = room_meta
    lineset = lineset.scale(SCALE, center=(0, 0, 0))
    o3d.io.write_line_set(f'/tmp/walls_{room_id_str}.ply', lineset)

# kool: x left, y forward, z up
R_cv_kool = np.array([
    -1, 0, 0,
    0, 0, -1,
    0, -1, 0
]).reshape((3, 3))
R_kool_cv = R_cv_kool.T

# WSU: West (left), South (back), Up coordinate system
R_cv_wsu = np.array([
    -1, 0, 0,
    0, 0, -1,
    0, -1, 0
]).reshape((3, 3))
R_wsu_cv = R_cv_wsu.T
T_wsu_cv = np.eye(4)
T_wsu_cv[:3, :3] = R_wsu_cv

camera_stat_dict = defaultdict(list)

for camera_meta_dict in meta_data_dict['camera_meta']:
    camera_id_str = camera_meta_dict['camera_id']
    room_id_str = camera_meta_dict['camera_room_id']
    room_meta = room_meta_dict[room_id_str]

    print(f'Processing camera {camera_id_str} in room {room_id_str}')

    new_cam_id_in_room = len(camera_stat_dict[room_id_str])
    room_output_dir = osp.join(output_dir, f'room_{room_id_str}')
    camera_output_dir = osp.join(room_output_dir, f'{new_cam_id_in_room}')

    target_pano_height, target_pano_width = 512, 1024

    # new camera meta data
    new_cam_meta_idct = adjust_cam_meta(raw_cam_meta_dict=camera_meta_dict,
                                        room_id=int(room_id_str),
                                        new_cam_id=new_cam_id_in_room,
                                        new_img_height=target_pano_height,
                                        new_img_width=target_pano_width)
    camera_stat_dict[room_id_str].append({new_cam_id_in_room: new_cam_meta_idct})

    T_cw = np.array(new_cam_meta_idct["camera_transform"]).reshape((4, 4))
    T_wc = np.linalg.inv(T_cw)
    T_wc[:3, 3] = T_wc[:3, 3]
    T_wc[:3, :3] = T_wc[:3, :3] @ R_wsu_cv
    T_cw = np.linalg.inv(T_wc)

    # T_wc = getCameraT(camera_meta_dict)
    # T_wc[:3, :3] = T_wc[:3, :3] @ R_kool_cv
    # T_wc[:3, 3] = T_wc[:3, 3] * SCALE
    # T_cw = np.linalg.inv(T_wc)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1).transform(T_wc)
    o3d.io.write_triangle_mesh(f'/tmp/camera_{camera_id_str}.ply', mesh)

    room_min, room_max = get_room_bounds(room_meta)
    room_obbs = []
    room_bboxes = []
    for bbox in ins_meta:
        T = np.array(bbox["transform"]).reshape(4, 4)
        center = T[:3, 3]
        if (
            room_min[0] < center[0] < room_max[0]
            and room_min[1] < center[1] < room_max[1]
            and room_min[2] < center[2] < room_max[2]
        ):
            size = np.array(bbox["size"]) * SCALE
            T[:3, 3] = T[:3, 3] * SCALE
            obb = trimesh.creation.box(extents=size, transform=T)
            room_obbs.append(obb)

            bbox = deepcopy(bbox)
            bbox["transform"] = T_cw @ T
            bbox["size"] = size
            room_bboxes.append(bbox)

    room_obbs = trimesh.util.concatenate(room_obbs)
    room_obbs.export(f'/tmp/bbox_room_{camera_id_str}.ply')

    room_obbs.apply_transform(T_wsu_cv @ T_cw)
    room_obbs.export(f'/tmp/bbox_room_cam_{camera_id_str}.ply')

    lineset = deepcopy(room_meta['lineset'])
    lineset = lineset.transform(T_wsu_cv @ T_cw)
    o3d.io.write_line_set(f'/tmp/walls_cam_{camera_id_str}.ply', lineset)

    # img_path = osp.join(render_path, camera_id_str, "cubic.jpg")
    img_path = osp.join(camera_output_dir, 'rgb.png')
    fov = None

    # img_path = osp.join(render_path, camera_id_str, "rgb_4.jpg")
    # fov = 90

    img = np.array(Image.open(img_path))
    pixs = obbs_to_pix(room_bboxes, img.shape[1], img.shape[0], fov=fov)
    img[pixs[:, 1], pixs[:, 0]] = [255, 0, 0]

    img_path = osp.join(camera_output_dir, 'rgb_debug.png')
    Image.fromarray(img).save(img_path)

    # img_path = osp.join(f'/tmp/rgb_debug_{camera_id_str}.png')
    # Image.fromarray(img).save(img_path)
    # break
