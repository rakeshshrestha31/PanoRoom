import os
import math
import open3d as o3d
import numpy as np

def manual_traverse_ply(ply:o3d.geometry.PointCloud, win_width:int=1920, win_height:int=1080):
    """Manual look-around traversal of point clouds

    Args:
        look_around_hegiht (float, optional): The height of the viewing angle when looking around (horizontal height). Defaults to 1.0.
    """
    def identity_T(vis):
        # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        vis.get_render_option().background_color = [0,0,0]
        vis.get_render_option().light_on = True

        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # world2cam = np.array([[0.99488194, -0.04935506, -0.0881703, -0.00310839],
        #                       [-0.10085076, -0.43105671, -0.8966712,  0.0118423],
        #                       [ 0.00624886,  0.90097402, -0.43382803,       0.1722927],
        #                       [0 ,       0 ,       0 ,         1]])
        world2cam = np.array([[1, 0, 0, -0.143],
                                [0, 0, -1, 0.014],
                                [ 0.,  1, 0,  -0.05],
                                [0 ,       0 ,       0 ,         1]])
        camera_params.extrinsic = world2cam
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
        vis.update_renderer()
        # o3d.io.write_pinhole_camera_parameters('viewpoint.json', camera_params)

    def capture_image(vis):
        vis.get_render_option().point_size = 2
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        print(
            f'camera params: intrinsic:\n {camera_params.intrinsic} \n extrinsic:\n {camera_params.extrinsic}')

        image = vis.capture_screen_float_buffer()
        depth = vis.capture_depth_float_buffer()
        vis.poll_events()
        o3d.io.write_image("output.png", image, 9)
        vis.update_renderer()
        return False

    print("   Press 'D' to detect apriltag")
    print("   Press ' ' to arrive the origin of world coordinate system")

    key_to_callback = {}
    key_to_callback[ord("D")] = capture_image
    key_to_callback[ord(" ")] = identity_T
    o3d.visualization.draw_geometries_with_key_callbacks(
        [ply], key_to_callback, width=win_width, height=win_height)

def traverse_ply(ply:o3d.geometry.PointCloud, win_width:int=1920, win_height:int=1080, trajectory=None):
    """Automatic look-around traversal of point clouds

    Args:
        look_around_hegiht (float, optional): The height of the viewing angle when looking around (horizontal height). Defaults to 1.0.
        detect_angle_inv (float, optional): Angular interval for automatic detection of calibration plates when looking around. Defaults to 10.0.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=win_width, height=win_height)
    vis.get_render_option().point_size = 5
    vis.get_render_option().light_on = True
    vis.get_render_option().background_color = [0,0,0]
    vis.get_view_control().change_field_of_view(step=90.0)
    # vis.get_render_option().mesh_shade_option = MeshShadeOption.SmoothShade
    # vis.get_render_option().light_ambient_color =  [ 0.6, 0.6, 0.6 ]
    vis.get_render_option().load_from_json("/home/ziqianbai/Toolkits/Open3D/examples/TestData/renderoption.json")
    vis.add_geometry(ply)

    camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    cam_intrinsic = camera_params.intrinsic.intrinsic_matrix
    print('camera_intrinsic: {}'.format(cam_intrinsic))

    world2cam = np.eye(4)
    if trajectory is None:
        world2cam = np.array(
            [[-9.65168910e-01, -2.61627549e-01,  0.0,  2.77999750e+00],
            [-5.80929859e-17,  2.14310549e-16, -1.0000000,  0.00000000e+00],
            [ 2.61627549e-01, -9.65168910e-01, -2.22044605e-16, -2.77999750e+00],
            [0, 0, 0, 1.0]])
        origin = world2cam[:3, 3]
        ply.translate(origin)
        world2cam[:3, 3] = [0,0,0]

        camera_params.extrinsic = world2cam
        euler_ang = np.array([0., 0., 0.])

        detect_angle_inv = 10.0
        detect_iter = int(360/detect_angle_inv)
        for i in range(360):
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
            cam_trans = np.eye(4)
            euler_ang[2] += math.pi * 2 / 360
            cam_trans[2, 3] = 1
            cam_trans[:3, :3] = eulerAnglesToRotationMatrix(euler_ang)
            camera_params.extrinsic = world2cam @ cam_trans
            vis.poll_events()
            vis.update_renderer()
            if i % detect_iter == 0:
                vis.capture_screen_image("output_{}.png".format(i))
                # image = vis.capture_screen_float_buffer()
                # depth = vis.capture_depth_float_buffer()
                # o3d.io.write_image("output_{}.png".format(i), image, 9)
    else:
        cnt = 0
        for idx in range(len(trajectory)):
            world2cam = trajectory[idx]
            # if idx  == 834:
            #     print(f'T_w_c: {world2cam}')
            #     origin = world2cam[:3, 3]
            #     self.ply.translate(origin)
            #     vis.clear_geometries()
            #     vis.add_geometry(self.ply)

            print(f' world2cam:\n {world2cam[:3,3]}')
            camera_params.extrinsic = world2cam

            vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
            vis.poll_events()
            vis.update_renderer()
            if idx % 3 == 0:
                vis.capture_screen_image("output_{}.png".format(cnt))
                cnt += 1

    vis.destroy_window()

if __name__=="__main__":
    test_ply_path = '/Users/fc/Desktop/papers/2024-ECCV-ctrlroom/_0/points3d_n.ply'
    o3d_ply = o3d.io.read_point_cloud(test_ply_path)
    o3d_ply.estimate_normals()
    manual_traverse_ply(o3d_ply, win_width=1920, win_height=1080)
