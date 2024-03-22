import os
import sys
sys.path.append('.')
sys.path.append('..')

import os.path as osp
from glob import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Dict

from PIL import Image
import trimesh
import open3d as o3d

from e2c_lib import multi_Perspec2Equirec as m_P2E
from pano_utils import vis_color_pointcloud
from meta import NYU40_LABEL_TO_ADEK_COLOR
from geometry_utils import create_spatial_quad_polygen

def read_koolai_cubemap_image(image_path:str, image_type:str='albedo')->List[np.ndarray]:
    """ read koolai cubemap image
    0:'r',
    1:'l',
    2:'u'
    3:'d',
    4:'f',
    5:'b',
    Args:
        image_path (str): cubemap image file apth

    Returns:
        numpy.ndarray: _description_
    """
    img = Image.open(image_path)
    w, h = img.size
    face_w = w // 6
    # split into 6 images
    img_list = []
    # front face
    img_list.append(img.crop((4 * face_w, 0, 5 * face_w, h)))
    # right face
    img_list.append(img.crop((0 * face_w, 0, 1 * face_w, h)))
    # back face
    img_list.append(img.crop((5 * face_w, 0, 6 * face_w, h)))
    # left face
    img_list.append(img.crop((1 * face_w, 0, 2 * face_w, h)))
    # up face
    img_list.append(img.crop((2 * face_w, 0, 3 * face_w, h)))
    # down face
    img_list.append(img.crop((3 * face_w, 0, 4 * face_w, h)))
    
    if len(img.split()) >= 3:
        img_list = np.array([np.array(img)[:,:,:3] for img in img_list])
    elif len(img.split()) == 1:
        # img_list = np.array([np.repeat(np.array(img)[:,:,np.newaxis], 3, axis=2) for img in img_list])
        if image_type == 'depth':
            img_list = np.array([np.array(img)[:,:,np.newaxis].astype(np.uint16) for img in img_list])
        elif image_type == 'semantic':
            img_list = [decode_nyu40_semantic_image(np.asarray(img)) for img in img_list]
            img_list = np.array(img_list)
        else:
            img_list = np.array([np.array(img)[:,:,np.newaxis].astype(np.uint8) for img in img_list])
    else:
        raise ValueError(f'unsupport image shape: {img.shape}')
        
    # print(f'img_list: {img_list[0].shape}')
    # plt.imshow(img_list[0])
    # plt.show()
    return img_list

def decode_nyu40_semantic_image(sem_image_nyu40:np.ndarray, visited:Dict=NYU40_LABEL_TO_ADEK_COLOR):
    height, width = sem_image_nyu40.shape[:2]

    # image_decoded = sem_image_nyu40.reshape((height, width)).astype(np.uint8)  # each pixel represent one catgory in NYU40
    image_decoded = sem_image_nyu40  # each pixel represent one catgory in NYU40

    unique_image = np.unique(image_decoded)
    # print(f'unique_image: {unique_image}')
    for semantic in unique_image:
        if visited.get(semantic) is None:
            # print(f'semantic label index {semantic} is absent!')
            visited[semantic] = np.array(
                [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)])

    image = np.zeros((height, width, 3), dtype=np.uint8)
    for semantic in unique_image:
        image[image_decoded == semantic] = visited[semantic]

    # new_image = Image.fromarray(image)
    return image
    

def cube2panorama(input_dir:str, 
                  output_dir:str, 
                  pano_width:int=1024, 
                  pano_height:int=512, 
                  convert_keys:List[str]=['albedo', 'depth', 'normal', 'instance', 'semantic']):
    """ convert cubemap images to panorama image

    Args:
        input_dir (str): input folder of koolai synthetic dataset
        output_dir (str): processed output folder
        pano_width (int, optional): _description_. Defaults to 1024.
        pano_height (int, optional): _description_. Defaults to 512.
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    convert_images_path_lst = []
    for key in convert_keys:
        convert_images_path_lst.append(osp.join(input_dir, f'{key}.png'))

    # cubemap fovs, phi and theta angles
    F_P_T_lst = [[90, 0, 0],  # front
                 [90, -90, 0], # right
                 [90, -180, 0], # back
                 [90, -270, 0], # left
                 [90, 0, 90], # up
                 [90, 0, -90]] # down
    for img_type, img_path in zip(convert_keys, convert_images_path_lst):
        faces_img_lst = read_koolai_cubemap_image(img_path, img_type)
        img_channel = faces_img_lst[0].shape[-1]
        per = m_P2E.Perspective(faces_img_lst, F_P_T_lst, channel=img_channel)
        img = per.GetEquirec(pano_height, pano_width)
        if img_type == 'depth':
            img = (img*4).astype(np.uint16)
            img = img.reshape((pano_height, pano_width))
        elif img_type == 'albedo' or img_type == 'normal':
            img = img.astype(np.uint8)
        elif img_type == 'instance':
            img = img.astype(np.uint8)
            img = img.reshape((pano_height, pano_width))
        elif img_type == 'semantic':
            # img = decode_nyu40_semantic_image(img)
            img = img.astype(np.uint8)
            # print(f'{img_type}, img shape: {img.shape}')
            # plt.imshow(img)
            # plt.show()
        img = Image.fromarray(img)
        img.save(osp.join(output_dir, osp.basename(img_path)))
    
    
def parse_user_output(structure_json_path:str)->dict:
    with open(structure_json_path, 'r') as f:
        data = json.load(f)
    
    camera_meta = None
    room_meta = None
    instance_meta = None
    for meta_data in data:
        # print(f"meta_data: {meta_data}")
        if 'camera_meta' in meta_data:
            camera_meta = meta_data['camera_meta']
            # print(f"camera_meta: {camera_meta['topview']}")
        if 'room_meta' in meta_data:
            room_meta = meta_data['room_meta']
            # print(f"room_meta: {room_meta}")
        if 'ins_meta' in meta_data:
            instance_meta = meta_data['ins_meta']
            # print(f"instance_meta: {instance_meta}")
        if 'camera_meta' not in meta_data and 'room_meta' not in meta_data and 'ins_meta' not in meta_data:
            # print(f"meta_data: {meta_data}")
            print("Unknow meta data")
    
    return dict(camera_meta=camera_meta, room_meta=room_meta, instance_meta=instance_meta)

def adjust_cam_meta(raw_cam_meta_dict:Dict, room_id:int, new_cam_id:int, new_img_height:int, new_img_width:int)->Dict:
    new_cam_meta_dict = {}
    new_cam_meta_dict["camera_id"] = new_cam_id
    new_cam_meta_dict["room_id"] = room_id
    # new_cam_meta_dict["camera_up"] = {"x": 0.0, "y": 0.0,"z": 1.0}
    new_cam_meta_dict["camera_up"] = raw_cam_meta_dict["camera_up"]
    look_at_pos = np.array([raw_cam_meta_dict["camera_look_at"]['x'],
                            raw_cam_meta_dict["camera_look_at"]['y'],
                            raw_cam_meta_dict["camera_look_at"]['z']]) * 0.001

    #  scale to meter
    raw_cam_pos = np.array([raw_cam_meta_dict["camera_position"]["x"], 
                            raw_cam_meta_dict["camera_position"]["y"], 
                            raw_cam_meta_dict["camera_position"]["z"]]) * 0.001
    look_at_dir = look_at_pos - raw_cam_pos
    look_at_dir = look_at_dir / np.linalg.norm(look_at_dir)
    new_cam_meta_dict["camera_forward"] = {"x": look_at_dir[0], "y": look_at_dir[1], "z": look_at_dir[2]}

    new_cam_meta_dict["camera_position"] = {"x": raw_cam_pos[0], "y": raw_cam_pos[1], "z": raw_cam_pos[2]}
    
    # calculate camera pose w2c
    up = np.array([raw_cam_meta_dict["camera_up"]['x'],
                  raw_cam_meta_dict["camera_up"]['y'],
                  raw_cam_meta_dict["camera_up"]['z']])
    up = up / np.linalg.norm(up)
    x = np.cross(look_at_dir, up)
    x = x / np.linalg.norm(x)
    z = np.cross(x, look_at_dir)
    z = z/np.linalg.norm(z)
    c2w = np.eye(4)
    c2w[:3, 0] = look_at_dir
    c2w[:3, 1] = x
    c2w[:3, 2] = z
    c2w[:3, 3] = raw_cam_pos
    
    axis_mat = np.array([[1, 0, 0, 0],
                         [0, 0, -1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
    w2c = np.linalg.inv(c2w)
    # w2c = axis_mat @ w2c
    new_cam_meta_dict["camera_transform"] = w2c.flatten().tolist()
    
    # convert pose from w2c to c2w
    # raw_cam_pose = np.array(raw_cam_meta_dict["camera_transform"]).reshape((4, 4))
    # new_cam_pose = np.linalg.inv(raw_cam_pose)
    # new_cam_pose[:3, 3] *= 0.001
    # new_cam_pose = new_cam_pose.flatten().tolist()
    # new_cam_meta_dict["camera_transform"] = new_cam_pose
    
    new_cam_meta_dict["image_height"] = new_img_height
    new_cam_meta_dict["image_width"] = new_img_width
    return new_cam_meta_dict

def parse_room_meta(meta_data_dict:Dict, room_id_str:str, room_output_dir:str, camera_pose_w2c:np.array=None)->Dict:
    """ parse room layout data

    Args:
        meta_data_dict (Dict): meta json file
        room_id_str (str): room id
        room_output_dir (str): output folder
        camera_pose_w2c (np.array, optional): camera pose(transform world points to caemra space). Defaults to None.

    Returns:
        Dict: room layout json data: floor, ceil
    """
    room_meta_dict = {}
    for room_meta in meta_data_dict['room_meta']:
        if room_meta['id'] == room_id_str:
            room_meta_dict['floor'] = room_meta['floor'].copy()
            room_meta_dict['ceil'] = room_meta['ceil'].copy()
            break
    # if room_id is not in the meta data
    if len(room_meta_dict) == 0:
        return room_meta_dict
    # parse floor and veil corners
    def parse_corners(corners:List[Dict[str, float]])->np.ndarray:
        corner_lst = []
        for corner in corners:
            corner_lst.append([float(corner['start']['x']), float(corner['start']['y']), float(corner['start']['z'])])
            corner_lst.append([float(corner['end']['x']), float(corner['end']['y']), float(corner['end']['z'])])
        return np.array(corner_lst)

    floor_points_lst = parse_corners(room_meta_dict['floor'])
    ceil_points_lst = parse_corners(room_meta_dict['ceil'])
    assert len(floor_points_lst) == len(ceil_points_lst)
    assert len(floor_points_lst) % 2 == 0

    quad_wall_mesh_lst = []
    for i in range(len(floor_points_lst)//2):
        floor_corner_i = floor_points_lst[i*2]
        floor_corner_j = floor_points_lst[i*2+1]
        ceil_corner_i = ceil_points_lst[i*2]
        ceil_corner_j = ceil_points_lst[i*2+1]
        # 3D coordinate for each wall
        quad_corners = np.array([floor_corner_i, ceil_corner_i, ceil_corner_j, floor_corner_j]) * 0.001
        # transform world point to camera point
        if camera_pose_w2c is not None:
            quad_corners = (camera_pose_w2c[:3, :3] @ quad_corners.T).T + camera_pose_w2c[:3, 3]
        wall_mesh = create_spatial_quad_polygen(quad_corners)
        quad_wall_mesh_lst.append(wall_mesh)

    quad_wall_mesh = trimesh.util.concatenate(quad_wall_mesh_lst)
    quad_wall_mesh.export(osp.join(room_output_dir, 'room_wall.ply'))
    return room_meta_dict
        
import time
def parse_single_scene(input_root_dir:str, output_dir:str) -> int:
    """ parse a single scene, split into multiple rooms

    Args:
        input_root_dir (str): scene folder path
        output_dir (str): output folder path

    Returns:
        int: how many rooms in the scene
    """
    rasterize_dirs = glob(osp.join(input_root_dir, '*_rasterize'))
    rgb_dirs = glob(osp.join(input_root_dir, '*_render'))
    structure_json_paths = glob(osp.join(input_root_dir, '*_structure/user_output.json'))
    if len(rasterize_dirs) == 0 or len(rgb_dirs) == 0 or len(structure_json_paths) == 0:
        print(f"WARNING: {input_root_dir} is empty")
        return 0, 0
    
    rasterize_dir = rasterize_dirs[0]
    rgb_dir = rgb_dirs[0]
    structure_json_path = structure_json_paths[0]
    
    rast_view_folder_lst = [f for f in os.listdir(rasterize_dir) if osp.isdir(osp.join(rasterize_dir, f)) and (f.split('_')[-1]).isdigit()]
    rgb_view_folder_lst = [f for f in os.listdir(rgb_dir) if osp.isdir(osp.join(rgb_dir, f)) and (f.split('_')[-1]).isdigit()]
    assert len(rast_view_folder_lst) == len(rgb_view_folder_lst), f"len(rast_view_folder_lst): {len(rast_view_folder_lst)}, len(rgb_view_folder_lst): {len(rgb_view_folder_lst)}"
    rast_view_folder_lst.sort(key=lambda x: int(x.split('_')[-1]))
    rgb_view_folder_lst.sort(key=lambda x: int(x.split('_')[-1]))
    
    
    # parse user_output.json
    meta_data_dict = parse_user_output(structure_json_path)
    
    camera_stat_dict = {}
    for rast_view_folder, rgb_view_folder in zip(rast_view_folder_lst, rgb_view_folder_lst):
        begin_tms = time.time()
        camera_id_str = rast_view_folder
        camera_meta_dict = None
        for cam_meta in meta_data_dict['camera_meta']:
            if cam_meta['camera_id'] == camera_id_str:
                camera_meta_dict = cam_meta
                break
        if camera_meta_dict is None:
            print(f"WARNING camera_id_str: {camera_id_str} not in meta_data_dict['camera_meta']")
            continue
        
        rast_view_folder = osp.join(rasterize_dir, rast_view_folder)
        rgb_view_folder = osp.join(rgb_dir, rgb_view_folder)
        assert osp.basename(rast_view_folder) == osp.basename(rgb_view_folder)
        
        if os.listdir(rast_view_folder) == 0 or os.listdir(rgb_view_folder) == 0:
            print(f"WARNING: {rast_view_folder} is empty")
            continue
        
        # output folder
        room_id_str = camera_meta_dict['camera_room_id']
        room_output_dir = osp.join(output_dir, f'room_{room_id_str}')
        os.makedirs(room_output_dir, exist_ok=True)
        if room_id_str not in camera_stat_dict:
            camera_stat_dict[room_id_str] = []
        

        new_cam_id_in_room = len(camera_stat_dict[room_id_str])
        camera_output_dir = osp.join(room_output_dir, f'{new_cam_id_in_room}')
        
        target_pano_height, target_pano_width = 512, 1024
        if True:
            # copy rgb image
            raw_rgb_img_path = osp.join(rgb_view_folder, 'cubic.jpg')
            if not os.path.exists(raw_rgb_img_path):
                print(f"WARNING: {raw_rgb_img_path} not exists!!")
                continue
            rgb_img = Image.open(raw_rgb_img_path).resize((target_pano_width, target_pano_height))
            if np.isnan(np.array(rgb_img).any()) or len(np.array(rgb_img)[np.array(rgb_img) > 0]) == 0:
                print(f"WARNING: {raw_rgb_img_path} is empty!!")
                continue
            
            os.makedirs(camera_output_dir, exist_ok=True)
            rgb_img.save(osp.join(camera_output_dir, 'rgb.png'))
        
        # convert cubemap to panorama, and copy rgb panroama
            cube2panorama(input_dir=rast_view_folder, 
                        output_dir=camera_output_dir,
                        pano_height=target_pano_height,
                        pano_width=target_pano_width, 
                        convert_keys=['albedo', 'depth', 'normal', 'instance', 'semantic'])
        # new camera meta data
        new_cam_meta_idct = adjust_cam_meta(raw_cam_meta_dict=camera_meta_dict,
                                            room_id=int(room_id_str), 
                                            new_cam_id=new_cam_id_in_room, 
                                            new_img_height=target_pano_height, 
                                            new_img_width=target_pano_width)
        camera_stat_dict[room_id_str].append({new_cam_id_in_room: new_cam_meta_idct})
        
        # generate room layout mesh in camera space
        cam_pose_w2c = np.array(new_cam_meta_idct["camera_transform"]).reshape((4, 4))
        room_layout_dict = parse_room_meta(meta_data_dict, room_id_str, camera_output_dir, camera_pose_w2c=cam_pose_w2c)
        if len(room_layout_dict) == 0:
            print(f"WARNING: room_id_str: {room_id_str} not in {structure_json_path}")

        # end_tms = time.time()
        # print(f"---------------- process scene {osp.basename(input_root_dir)} room {room_id_str} camera {new_cam_id_in_room} time: {end_tms - begin_tms} ----------------")
    print(f"---------------- process scene {osp.basename(input_root_dir)} ----------------")
    
    # save camera meta data in each room
    for k, v in camera_stat_dict.items():
        room_id_str = k
        room_output_dir = osp.join(output_dir, f'room_{room_id_str}')
        
        # room_meta_dict = {}
        room_meta_dict = parse_room_meta(meta_data_dict, room_id_str, room_output_dir)
        room_meta_dict['room_id'] = int(room_id_str)
        room_meta_dict['cameras'] = v
        save_room_meta_path = osp.join(room_output_dir, 'room_meta.json')
        json.dump(room_meta_dict, open(save_room_meta_path, 'w'), indent=4)
    
    num_rooms = len(camera_stat_dict)
    num_cameras = sum([len(v) for v in camera_stat_dict.values()])
    return num_rooms, num_cameras 
        
if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess single scene data from koolai synthetic dataset")
    parser.add_argument("--input_dir", type=str, required=True, help='input folder of koolai synthetic dataset')
    parser.add_argument("--output_dir", type=str, required=True, help='processed output folder')
    args = parser.parse_args()
    input_root_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    rasterize_dir = glob(osp.join(input_root_dir, '*_rasterize'))[0]
    rgb_dir = glob(osp.join(input_root_dir, '*_render'))[0]
    structure_json_path = glob(osp.join(input_root_dir, '*_structure/user_output.json'))[0]
    
    rast_view_folder_lst = [f for f in os.listdir(rasterize_dir) if osp.isdir(osp.join(rasterize_dir, f)) and (f.split('_')[-1]).isdigit()]
    rgb_view_folder_lst = [f for f in os.listdir(rgb_dir) if osp.isdir(osp.join(rgb_dir, f)) and (f.split('_')[-1]).isdigit()]
    assert len(rast_view_folder_lst) == len(rgb_view_folder_lst), f"len(rast_view_folder_lst): {len(rast_view_folder_lst)}, len(rgb_view_folder_lst): {len(rgb_view_folder_lst)}"
    rast_view_folder_lst.sort(key=lambda x: int(x.split('_')[-1]))
    rgb_view_folder_lst.sort(key=lambda x: int(x.split('_')[-1]))
    
    print(f' {len(rast_view_folder_lst)} views in {input_root_dir}')
    
    # parse user_output.json
    meta_data_dict = parse_user_output(structure_json_path)
    
    camera_stat_dict = {}
    for rast_view_folder, rgb_view_folder in zip(rast_view_folder_lst, rgb_view_folder_lst):
        camera_id_str = rast_view_folder
        # if camera_id_str not in meta_data_dict['camera_meta']:
        #     print(f"WARNING camera_id_str: {camera_id_str} not in meta_data_dict['camera_meta']")
        #     continue
        
        # camera_meta_dict = meta_data_dict['camera_meta'][camera_id_str]
        camera_meta_dict = None
        for cam_meta in meta_data_dict['camera_meta']:
            if cam_meta['camera_id'] == camera_id_str:
                camera_meta_dict = cam_meta
                break
        if camera_meta_dict is None:
            print(f"WARNING camera_id_str: {camera_id_str} not in meta_data_dict['camera_meta']")
            continue
        # output folder
        room_id_str = camera_meta_dict['camera_room_id']
        room_output_dir = osp.join(output_dir, f'room_{room_id_str}')
        os.makedirs(room_output_dir, exist_ok=True)
        if room_id_str not in camera_stat_dict:
            camera_stat_dict[room_id_str] = []
        
        rast_view_folder = osp.join(rasterize_dir, rast_view_folder)
        rgb_view_folder = osp.join(rgb_dir, rgb_view_folder)
        assert osp.basename(rast_view_folder) == osp.basename(rgb_view_folder)
        
        if os.listdir(rast_view_folder) == 0 or os.listdir(rgb_view_folder) == 0:
            print(f"WARNING: {rast_view_folder} is empty")
            continue
        new_cam_id_in_room = len(camera_stat_dict[room_id_str])
        camera_output_dir = osp.join(room_output_dir, f'{new_cam_id_in_room}')
        
        target_pano_height, target_pano_width = 512, 1024
        if True:
            # copy rgb image
            raw_rgb_img_path = osp.join(rgb_view_folder, 'cubic.jpg')
            rgb_img = Image.open(raw_rgb_img_path).resize((target_pano_width, target_pano_height))
            if np.isnan(np.array(rgb_img).any()) or len(np.array(rgb_img)[np.array(rgb_img) > 0]) == 0:
                print(f"WARNING: {raw_rgb_img_path} is empty!!")
                continue
            
            os.makedirs(camera_output_dir, exist_ok=True)
            rgb_img.save(osp.join(camera_output_dir, 'rgb.png'))
        
        # convert cubemap to panorama, and copy rgb panroama
            cube2panorama(input_dir=rast_view_folder, 
                        output_dir=camera_output_dir,
                        pano_height=target_pano_height,
                        pano_width=target_pano_width, 
                        convert_keys=['albedo', 'depth', 'normal', 'instance', 'semantic'])
        # new camera meta data
        new_cam_meta_idct = adjust_cam_meta(raw_cam_meta_dict=camera_meta_dict,
                                            room_id=int(room_id_str), 
                                            new_cam_id=new_cam_id_in_room, 
                                            new_img_height=target_pano_height, 
                                            new_img_width=target_pano_width)
        camera_stat_dict[room_id_str].append({new_cam_id_in_room: new_cam_meta_idct})
        
        # check depth
        if False:
            rgb_img_filepath = osp.join(camera_output_dir, 'rgb.png')
            depth_img_filepath = osp.join(camera_output_dir, 'depth.png')
            saved_color_pcl_filepath = osp.join(camera_output_dir, 'points3d.ply')
            vis_color_pointcloud(rgb_img_filepath, depth_img_filepath, saved_color_pcl_filepath, depth_scale=4000.0, normaliz=False)
        
        # generate room layout mesh in camera space
        cam_pose_w2c = np.array(new_cam_meta_idct["camera_transform"]).reshape((4, 4))
        cam_pos = np.array([new_cam_meta_idct["camera_position"]["x"], 
                            new_cam_meta_idct["camera_position"]["y"], 
                            new_cam_meta_idct["camera_position"]["z"]]) 
        parse_room_meta(meta_data_dict, room_id_str, camera_output_dir, camera_pose_w2c=cam_pose_w2c)
        if False:
            # transform point cloud from camera space to world space
            pcd = o3d.io.read_point_cloud(saved_color_pcl_filepath)
            pcd.transform(np.linalg.inv(cam_pose_w2c))
            o3d.io.write_point_cloud(osp.join(camera_output_dir, 'w_points3d.ply'), pcd)
        print(f"---------------- process room {room_id_str} camera {new_cam_id_in_room} ----------------")
    
    # save camera meta data in each room
    for k, v in camera_stat_dict.items():
        room_id_str = k
        room_output_dir = osp.join(output_dir, f'room_{room_id_str}')
        
        # room_meta_dict = {}
        room_meta_dict = parse_room_meta(meta_data_dict, room_id_str, room_output_dir)
        room_meta_dict['room_id'] = int(room_id_str)
        room_meta_dict['cameras'] = v
        save_room_meta_path = osp.join(room_output_dir, 'room_meta.json')
        json.dump(room_meta_dict, open(save_room_meta_path, 'w'), indent=4)