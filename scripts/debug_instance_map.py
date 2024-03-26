from argparse import ArgumentParser
from pathlib import Path
import json
import random

from PIL import Image
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser(description="Preprocess single scene data from koolai synthetic dataset")
    parser.add_argument("--input_dir", type=str, required=True, help='input folder of koolai synthetic dataset')
    args = parser.parse_args()
    input_root_dir = Path(args.input_dir)
    scene_folderpath = (input_root_dir / ".." / ".." / "..").resolve()
    room_meta_path = input_root_dir / ".." / "room_meta.json"
    camera_id = input_root_dir.name
    rasterize_paths = next(scene_folderpath.glob("*_rasterize/"))

    with open(room_meta_path, "r") as f:
        room_meta = json.load(f)

    camera_meta = room_meta["cameras"][int(camera_id)][camera_id]
    camera_id_str = camera_meta["camera_old_room_id"]

    panorama_image = input_root_dir / "instance.png"
    cubemap_image = rasterize_paths / camera_id_str / "instance.png"
    print(panorama_image, cubemap_image)

    panorama_image = np.array(Image.open(panorama_image))
    cubemap_image = np.array(Image.open(cubemap_image))

    unique_image = np.concatenate([np.unique(panorama_image), np.unique(cubemap_image)])
    unique_image = np.unique(unique_image)
    print(f'unique_image: {unique_image}')
    visited = {}
    for semantic in unique_image:
        if visited.get(semantic) is None:
            # print(f'semantic label index {semantic} is absent!')
            visited[semantic] = np.array(
                [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)])

    image1 = np.zeros((*panorama_image.shape[:2], 3), dtype=np.uint8)
    for semantic in unique_image:
        image1[panorama_image == semantic] = visited[semantic]

    image2 = np.zeros((*cubemap_image.shape[:2], 3), dtype=np.uint8)
    for semantic in unique_image:
        image2[cubemap_image == semantic] = visited[semantic]

    Image.fromarray(image1).save("/tmp/instance_panorama.png")
    Image.fromarray(image2).save("/tmp/instance_cubemap.png")
