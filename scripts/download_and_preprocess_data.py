import os
import argparse
import numpy as np
import json
from tqdm import tqdm

from preprocess_koolai_data import parse_single_scene

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--koolai_csv_filepath', type=str, 
                        default='/mnt/nas_3dv/hdd1/datasets/datasets/KoolAI/20240223/koolai_render_data_20240223.csv', 
                        help='The csv file contains all download info of the data')
    parser.add_argument('--koolai_room_ins_folderpath', type=str,
                        default='/mnt/nas_3dv/hdd1/datasets/datasets/KoolAI/20240223/koolai_room_ins_data_20240223/',
                        help='The directory contains room_meta info corresponding to the downloaded files')
    parser.add_argument('--output_dir', type=str, 
                        default='/mnt/nas_3dv/hdd1/datasets/datasets/KoolAI/20240223', 
                        help='The directory to save the downloaded files')
    args = parser.parse_args()
    csv_file = args.koolai_csv_filepath
    room_meta_folder = args.koolai_room_ins_folderpath
    
    dataset_stat_dict = {}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # downlaod the data
    download_exe = 'download_data.py'
    cmd = f'python {download_exe} --koolai_csv_filepath {csv_file} --koolai_room_ins_folderpath {room_meta_folder} --output_dir {args.output_dir}'
    os.system(cmd)
    
    # process the data
    process_exe = 'preprocess_koolai_data.py'
    scene_folders_lst = [f for f in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, f))]
    scene_folders_lst.sort()
    
    total_rooms_num = 0
    total_cams_num = 0
    total_scenes_num = len(scene_folders_lst)
    for scene_folder in tqdm(scene_folders_lst):
        scene_folderpath = os.path.join(args.output_dir, scene_folder)
        # cmd = f'python {process_exe} --input_dir {scene_folderpath} --output_dir {os.path.join(scene_folderpath, "panorama")}'
        # os.system(cmd)
        save_folderpath = os.path.join(scene_folderpath, 'panorama')
        if not os.path.exists(save_folderpath):
            num_rooms, num_cams = parse_single_scene(input_root_dir=scene_folderpath, output_dir=save_folderpath)
        else:
            print(f'Panorama folder already exists: {save_folderpath}')
            room_folders = [os.path.join(save_folderpath, f) for f in os.listdir(save_folderpath) if os.path.isdir(os.path.join(save_folderpath, f))]
            num_rooms = len(room_folders)
            num_cams = sum([len(next(os.walk(room_folder))[1]) for room_folder in room_folders])
        
        dataset_stat_dict[scene_folder] = {'num_rooms': num_rooms, 'num_cams': num_cams} 
        total_rooms_num += num_rooms
        total_cams_num += num_cams

    print(f'Processed {total_scenes_num} scenes, {total_rooms_num} rooms, {total_cams_num} cameras')
    dataset_stat_dict['total_scenes'] = total_scenes_num
    dataset_stat_dict['total_rooms'] = total_rooms_num
    
    dataset_stat_filepath = os.path.join(args.output_dir, 'dataset_stat.txt')
    with open(dataset_stat_filepath, 'w') as f:
        json.dump(dataset_stat_dict, f, indent=4)