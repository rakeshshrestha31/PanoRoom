# find out all broken links in the csv file

import csv
import requests
import os
import os.path as osp
from glob import glob
import pandas as pd

if __name__ == "__main__":


    # if len(sys.argv) < 2:
    #     print("Usage: {} <csv_file>".format(sys.argv[0]))
    #     sys.exit(1)

    # csv_file = sys.argv[1]
    # with open(csv_file, "r") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         url = row[0]
    #         try:
    #             response = requests.head(url)
    #             if response.status_code != 200:
    #                 print("Broken link: {}".format(url))
    #         except requests.exceptions.RequestException as e:
    #             print("Error: {}".format(e))
    #             continue
    
    input_folderpath = '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240313_data/data'
    
    input_scene_folders = [f for f in os.listdir(input_folderpath) if os.path.isdir(osp.join(input_folderpath, f))]
    
    for scen_folder in input_scene_folders:
        scene_root_dir = osp.join(input_folderpath, scen_folder)
        rasterize_dirs = glob(osp.join(scene_root_dir, '*_rasterize'))
        rgb_dirs = glob(osp.join(scene_root_dir, '*_render'))
        structure_json_paths = glob(osp.join(scene_root_dir, '*_structure/user_output.json'))
        if len(rasterize_dirs) == 0 or len(rgb_dirs) == 0 or len(structure_json_paths) == 0:
            print(f"WARNING: {scene_root_dir} is empty")
            corresp_cvs_filepath = glob(osp.join(osp.dirname(input_folderpath), '*.csv'))
            if len(corresp_cvs_filepath) > 0:
                corresp_cvs_filepath = corresp_cvs_filepath[0]
                # Load the content of the CSV file
                content = pd.read_csv(corresp_cvs_filepath, delimiter=',')

                # Parse the content and download the data
                for result, wid in zip(content['result'], content['world_id']):
                    result = result.replace('null', 'None')
                    # Load the JSON data
                    data = eval(result)
                    
                    task_id = data['sceneTaskId']
                    task_index = data['index']
                    scene_name = data['sceneId']
                    camera_num = data['cameraCount']
                    if scene_name == scen_folder:
                        print(f"Found the corresponding scene: {data}")
                    