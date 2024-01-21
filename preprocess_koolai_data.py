import os
import os.path as osp
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PIL import Image
from typing import List, Dict
from e2c_lib import multi_Perspec2Equirec as m_P2E
from pano_utils import vis_color_pointcloud

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
        else:
            img_list = np.array([np.array(img)[:,:,np.newaxis].astype(np.uint8) for img in img_list])
    else:
        raise ValueError(f'unsupport image shape: {img.shape}')
        
    # print(f'img_list: {img_list[0].shape}')
    # plt.imshow(img_list[0])
    # plt.show()
    return img_list

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
        elif img_type == 'instance' or img_type == 'semantic':
            img = img.astype(np.uint8)
            img = img.reshape((pano_height, pano_width))
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
    new_cam_meta_dict["camera_up"] = {"x": 0.0, "y": 0.0,"z": 1.0}
    new_cam_meta_dict["camera_forward"] = {"x": 0.0, "y": 1.0,"z": 0.0}
    
    raw_cam_pos_dict = raw_cam_meta_dict["camera_position"]
    #  scale to meter
    raw_cam_pos = np.array([raw_cam_pos_dict["x"], raw_cam_pos_dict["y"], raw_cam_pos_dict["z"]]) * 0.001
    new_cam_meta_dict["camera_position"] = {"x": raw_cam_pos[0], "y": raw_cam_pos[1], "z": raw_cam_pos[2]}
    
    # convert pose from w2c to c2w
    raw_cam_pose = np.array(raw_cam_meta_dict["camera_transform"]).reshape((4, 4))
    new_cam_pose = np.linalg.inv(raw_cam_pose)
    new_cam_pose[:3, 3] *= 0.001
    new_cam_pose = new_cam_pose.flatten().tolist()
    new_cam_meta_dict["camera_transform"] = new_cam_pose
    
    new_cam_meta_dict["image_height"] = new_img_height
    new_cam_meta_dict["image_width"] = new_img_width
    return new_cam_meta_dict
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help='input folder of koolai synthetic dataset')
    parser.add_argument("--output_dir", type=str, required=True, help='processed output folder')
    args = parser.parse_args()
    input_root_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    rasterize_dir = osp.join(input_root_dir, 'rasterize')
    rgb_dir = osp.join(input_root_dir, 'render')
    structure_json_path = osp.join(input_root_dir, 'structure/user_output.json')
    
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
        if camera_id_str not in meta_data_dict['camera_meta']:
            print(f"WARNING camera_id_str: {camera_id_str} not in meta_data_dict['camera_meta']")
            continue
        
        camera_meta_dict = meta_data_dict['camera_meta'][camera_id_str]
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
        os.makedirs(camera_output_dir, exist_ok=True)
        
        target_pano_height, target_pano_width = 512, 1024
        # copy rgb image
        raw_rgb_img_path = osp.join(rgb_view_folder, 'cubic.jpg')
        rgb_img = Image.open(raw_rgb_img_path).resize((target_pano_width, target_pano_height))
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
        rgb_img_filepath = osp.join(camera_output_dir, 'rgb.png')
        depth_img_filepath = osp.join(camera_output_dir, 'depth.png')
        saved_color_pcl_filepath = osp.join(camera_output_dir, 'points3d.ply')
        vis_color_pointcloud(rgb_img_filepath, depth_img_filepath, saved_color_pcl_filepath, depth_scale=4000.0, normaliz=False)
        
        print(f"---------------- process room {room_id_str} camera {new_cam_id_in_room} ----------------")
    
    # save camera meta data in each room
    for k, v in camera_stat_dict.items():
        room_id_str = k
        room_meta_dict = {}
        room_meta_dict['room_id'] = int(room_id_str)
        room_meta_dict['cameras'] =v
        for room_meta in meta_data_dict['room_meta']:
            if room_meta['room_id'] == int(room_id_str):
                room_meta_dict['room_type'] = room_meta['room_type']
                room_meta_dict['room_area'] = room_meta['room_area']
                break
        room_output_dir = osp.join(output_dir, f'room_{room_id_str}')
        save_room_meta_path = osp.join(room_output_dir, 'room_meta.json')
        json.dump(room_meta_dict, open(save_room_meta_path, 'w'), indent=4)