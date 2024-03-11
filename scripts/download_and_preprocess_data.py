import os
import argparse
import numpy as np
from tqdm import tqdm

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
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # downlaod the data
    download_exe = 'download_data.py'
    cmd = f'python {download_exe} --koolai_csv_filepath {csv_file} --koolai_room_ins_folderpath {room_meta_folder} --output_dir {args.output_dir}'
    os.system(cmd)
    
    # process the data
    process_exe = 'preprocess_koolai_data.py'
    scene_folders_lst = [f for f in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, f))]
    for scene_folder in tqdm(scene_folders_lst):
        scene_folderpath = os.path.join(args.output_dir, scene_folder)
        cmd = f'python {process_exe} --input_dir {scene_folderpath} --output_dir {os.path.join(scene_folderpath, "panorama")}'
        os.system(cmd)
        exit(0)