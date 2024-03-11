import os
import pandas as pd
import requests
import shutil

def download_from_csv(csv_filename:str, meta_folderpath:str, output_dir:str)->int:
    # Load the content of the CSV file
    content = pd.read_csv(csv_filename, delimiter=',')
    print(f'Content: {content}')
    
    # output_dir = os.path.dirname(csv_filename)
    meta_folders = [f for f in os.listdir(meta_folderpath) if f.isdigit()]
    # Parse the content and download the data
    for result, wid in zip(content['result'], content['world_id']):
        result = result.replace('null', 'None')
        # print(f'result: {result}, world_id: {wid}')
        # Load the JSON data
        data = eval(result)
        print(f'data: {data}, world_id: {wid}')
        assert str(wid) in meta_folders, f'world_id {wid} not in {meta_folderpath}'
        
        room_meta_filepath = os.path.join(meta_folderpath, str(wid), 'room_meta.json')
        ins_meta_filepath = os.path.join(meta_folderpath, str(wid), 'ins_meta.json')
        
        task_id = data['sceneTaskId']
        task_index = data['index']
        scene_name = data['sceneId']
        camera_num = data['cameraCount']
        print(f'scene: {scene_name}, camera_num: {camera_num}')
        output_folder = os.path.join(output_dir, scene_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # copy the meta files
        shutil.copy(room_meta_filepath, os.path.join(output_folder, 'room_meta.json'))
        shutil.copy(ins_meta_filepath, os.path.join(output_folder, 'ins_meta.json'))
        
        # Download the files
        zip_files = []
        unzip_dirs = []
        for key, value in data.items():
            if key.endswith('Size'):
                file_key = key[:-4]
                url = data[file_key]
                response = requests.get(url)

                # Save the downloaded file
                save_path = os.path.join(output_folder, f'{wid}_{file_key}.zip')
                zip_files.append(save_path)
                unzip_dirs.append(os.path.join(output_folder, f'{wid}_{file_key}'))
                with open(save_path, 'wb') as output_file:
                    output_file.write(response.content)
        
        # unzip the files
        for zip_file, out_dir in zip(zip_files, unzip_dirs):
            os.system(f'unzip {zip_file} -d {out_dir}')
            os.system(f'rm {zip_file}')
    return len(content)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--koolai_csv_filepath', type=str, 
                        default='/mnt/nas_3dv/hdd1/datasets/KoolAI/20240206_data/koolai_render_data_20240206.csv', 
                        help='The filename of the data')
    parser.add_argument('--koolai_room_ins_folderpath', type=str, 
                        default='/mnt/nas_3dv/hdd1/datasets/KoolAI/20240206_data/koolai_room_ins_data_20240206/', help='The directory to save the downloaded files')
    parser.add_argument('--output_dir', type=str, default='', help='The directory to save the downloaded files')
    args = parser.parse_args()
    
    scene_num = download_from_csv(args.koolai_csv_filepath, meta_folderpath=args.koolai_room_ins_folderpath, output_dir=args.output_dir)
    print(f'Downloaded {scene_num} scenes from {args.koolai_csv_filepath} to {args.output_dir}')
    
    