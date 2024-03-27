import os
import os.path as osp
import shutil

if __name__ == '__main__':
    # preprocessed data folder
    input_dirs = [
        # '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240304_data/data/',
        #          '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240310_data/data/',
        #          '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240312_data/data/',
        #          '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240313_data/data/',
                 '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240229_data/data/',]
    
    for input_dir in input_dirs:
        scene_folders_lst = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
        scene_folders_lst.sort()
        
        for scene_folder in scene_folders_lst:
            scene_folderpath = osp.join(input_dir, scene_folder)
            # preprocessed panorama folder
            pano_folder = osp.join(scene_folderpath, 'panorama')
            if not osp.exists(pano_folder):
                continue
            
            rooms_folder_lst = [f for f in os.listdir(pano_folder) if osp.isdir(osp.join(pano_folder, f)) if f.startswith('room_')]
            for room_folder in rooms_folder_lst:
                room_folderpath = osp.join(pano_folder, room_folder)
                rgb_folder = osp.join(room_folderpath, 'rgb')
                depth_folder = osp.join(room_folderpath, 'depth')
                semantic_folder = osp.join(room_folderpath, 'semantic')
                instance_folder = osp.join(room_folderpath, 'instance')
                normal_folder = osp.join(room_folderpath, 'normal')
                albedo_folder = osp.join(room_folderpath, 'albedo')
                
                os.makedirs(rgb_folder, exist_ok=True)
                os.makedirs(depth_folder, exist_ok=True)
                os.makedirs(semantic_folder, exist_ok=True)
                os.makedirs(instance_folder, exist_ok=True)
                os.makedirs(normal_folder, exist_ok=True)
                os.makedirs(albedo_folder, exist_ok=True)
                
                camera_folders = [f for f in os.listdir(room_folderpath) if osp.isdir(osp.join(room_folderpath, f)) if f.isdigit()]
                camera_folders.sort(key=lambda x: int(x))
                
                for camera_folder in camera_folders:
                    camera_folderpath = osp.join(room_folderpath, camera_folder)
                    rgb_img_path = osp.join(camera_folderpath, 'rgb.png')
                    depth_img_path = osp.join(camera_folderpath, 'depth.png')
                    semantic_img_path = osp.join(camera_folderpath, 'semantic.png')
                    instance_img_path = osp.join(camera_folderpath, 'instance.png')
                    normal_img_path = osp.join(camera_folderpath, 'normal.png')
                    albedo_img_path = osp.join(camera_folderpath, 'albedo.png')
                    
                    shutil.move(rgb_img_path, osp.join(rgb_folder, f'{camera_folder}.png'))
                    shutil.move(depth_img_path, osp.join(depth_folder, f'{camera_folder}.png'))
                    shutil.move(semantic_img_path, osp.join(semantic_folder, f'{camera_folder}.png'))
                    shutil.move(instance_img_path, osp.join(instance_folder, f'{camera_folder}.png'))
                    shutil.move(normal_img_path, osp.join(normal_folder, f'{camera_folder}.png'))
                    shutil.move(albedo_img_path, osp.join(albedo_folder, f'{camera_folder}.png'))
                    
                    print(f'Moved images from {camera_folderpath} to {rgb_folder}')
                    # delete the camera folder
                    shutil.rmtree(camera_folderpath)
