# import numpy as np
# import pptk

# from distutils.sysconfig import get_python_lib
# print(get_python_lib())

# path_to_point_cloud = './kitti/training/velodyne_reduced/000000.bin'

# point_cloud_data = np.fromfile(path_to_point_cloud, '<f4')  # little-endian float32
# point_cloud_data = np.reshape(point_cloud_data, (-1, 4))    # x, y, z, r
# pptk.viewer(point_cloud_data[:, :3])













# """ Remove points outside the image coordinates

#     This script is from https://github.com/qianguih/voxelnet/blob/master/data/crop.py
# """

# import numpy as np
# # from scipy.misc import imread
# import imageio
# import os


# CAM = 2


# def load_velodyne_points(filename):
#     points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
#     # points = points[:, :3]  # exclude luminance
#     return points


# def load_calib(calib_dir):
#     # P2 * R0_rect * Tr_velo_to_cam * y
#     lines = open(calib_dir).readlines()
#     lines = [line.split()[1:] for line in lines][:-1]
#     #
#     P = np.array(lines[CAM]).reshape(3, 4)
#     #
#     Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
#     Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
#     #
#     R_cam_to_rect = np.eye(4)
#     R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)
#     #
#     P = P.astype('float32')
#     Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
#     R_cam_to_rect = R_cam_to_rect.astype('float32')
#     return P, Tr_velo_to_cam, R_cam_to_rect


# def prepare_velo_points(pts3d_raw):
#     '''Replaces the reflectance value by 1, and tranposes the array, so
#         points can be directly multiplied by the camera projection matrix'''
#     pts3d = pts3d_raw
#     # Reflectance > 0
#     indices = pts3d[:, 3] > 0
#     pts3d = pts3d[indices, :]
#     pts3d[:, 3] = 1
#     return pts3d.transpose(), indices


# def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
#     '''Project 3D points into 2D image. Expects pts3d as a 4xN
#         numpy array. Returns the 2D projection of the points that
#         are in front of the camera only an the corresponding 3D points.'''
#     # 3D points in camera reference frame.
#     pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
#     # Before projecting, keep only points with z>0
#     # (points that are in fronto of the camera).
#     idx = (pts3d_cam[2, :] >= 0)
#     pts2d_cam = Prect.dot(pts3d_cam[:, idx])
#     return pts3d[:, idx], pts2d_cam / pts2d_cam[2, :], idx


# def align_img_and_pc(img_dir, pc_dir, calib_dir):
#     img = imageio.imread(img_dir)
#     # img = imread(img_dir)
#     pts = load_velodyne_points(pc_dir)
#     P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_dir)

#     pts3d, indices = prepare_velo_points(pts)
#     # pts3d_ori = pts3d.copy()
#     reflectances = pts[indices, 3]
#     pts3d, pts2d_normed, idx = project_velo_points_in_img(pts3d, Tr_velo_to_cam, R_cam_to_rect, P)
#     # print reflectances.shape, idx.shape
#     reflectances = reflectances[idx]
#     # print reflectances.shape, pts3d.shape, pts2d_normed.shape
#     # assert reflectances.shape[0] == pts3d.shape[1] == pts2d_normed.shape[1]

#     rows, cols = img.shape[:2]

#     points = []
#     for i in range(pts2d_normed.shape[1]):
#         c = int(np.round(pts2d_normed[0, i]))
#         r = int(np.round(pts2d_normed[1, i]))
#         if c < cols and r < rows and r > 0 and c > 0:
#             color = img[r, c, :]
#             point = [pts3d[0, i], pts3d[1, i], pts3d[2, i], reflectances[i], color[0], color[1], color[2],
#                      pts2d_normed[0, i], pts2d_normed[1, i]]
#             points.append(point)

#     points = np.array(points)
#     return points


# # update the following directories

# # path to data_object_image_2/training/image_2
# IMG_ROOT = './kitti/training/image_2/'
# # path to data_object_velodyne/training/velodyne
# PC_ROOT = './kitti/training/velodyne/'
# # path to data_object_calib/training/calib
# CALIB_ROOT = '/home/dtc/Data/KITTI/data_object_calib/training/calib/'

# # path to the folder for saving cropped point clouds
# SAVE_ROOT = '/home/dtc/Data/KITTI/'

# for frame in range(0, 3):

#     print('--- processing {0:06d}'.format(frame))

#     img_dir = os.path.join(IMG_ROOT,  '{0:06d}.png'.format(frame))
#     pc_dir = os.path.join(PC_ROOT, '{0:06d}.bin'.format(frame))
#     calib_dir = os.path.join(CALIB_ROOT, '{0:06d}.txt'.format(frame))

#     points = align_img_and_pc(img_dir, pc_dir, calib_dir)

#     output_name = os.path.join(SAVE_ROOT, '{0:06d}.bin'.format(frame))
#     points[:, :4].astype('float32').tofile(output_name)



""" Draw labeled objects in images

    Author: Huanle Zhang
    Website: www.huanlezhang.com
"""

import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import os
import pptk
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


MARKER_COLOR = {
    'Car': [1, 0, 0],               # red
    'DontCare': [0, 0, 0],          # black
    'Pedestrian': [0, 0, 1],        # blue
    'Van': [1, 1, 0],               # yellow
    'Cyclist': [1, 0, 1],           # magenta
    'Truck': [0, 1, 1],             # cyan
    'Misc': [0.5, 0, 0],            # maroon
    'Tram': [0, 0.5, 0],            # green
    'Person_sitting': [0, 0, 0.5]}  # navy

# image border width
BOX_BORDER_WIDTH = 5

# point size
POINT_SIZE = 0.005


def show_object_in_image(img_filename, label_filename):
    img = mping.imread(img_filename)
    with open(label_filename) as f_label:
        lines = f_label.readlines()
        for line in lines:
            line = line.strip('\n').split()
            left_pixel, top_pixel, right_pixel, bottom_pixel = [int(float(line[i])) for i in range(4, 8)]
            box_border_color = MARKER_COLOR[line[0]]
            for i in range(BOX_BORDER_WIDTH):
                img[top_pixel+i, left_pixel:right_pixel, :] = box_border_color
                img[bottom_pixel-i, left_pixel:right_pixel, :] = box_border_color
                img[top_pixel:bottom_pixel, left_pixel+i, :] = box_border_color
                img[top_pixel:bottom_pixel, right_pixel-i, :] = box_border_color
    plt.imshow(img)
    plt.show()


def show_object_in_point_cloud(point_cloud_filename, label_filename, calib_filename):
    pc_data = np.fromfile(point_cloud_filename, '<f4')  # little-endian float32
    pc_data = np.reshape(pc_data, (-1, 4))
    pc_color = np.ones((len(pc_data), 3))
    calib = load_kitti_calib(calib_filename)
    with open(label_filename) as f_label:
        lines = f_label.readlines()
        for line in lines:
            line = line.strip('\n').split()
            point_color = MARKER_COLOR[line[0]]
            _, box3d_corner = camera_coordinate_to_point_cloud(line[8:15], calib['Tr_velo_to_cam'])
            for i, v in enumerate(pc_data):
                if point_in_cube(v[:3], box3d_corner) is True:
                    pc_color[i, :] = point_color

    v = pptk.viewer(pc_data[:, :3], pc_color)
    v.set(point_size=POINT_SIZE)


def point_in_cube(point, cube):
    z_min = np.amin(cube[:, 2], 0)
    z_max = np.amax(cube[:, 2], 0)

    if point[2] > z_max or point[2] < z_min:
        return False

    point = Point(point[:2])
    polygon = Polygon(cube[:4, :2])

    return polygon.contains(point)


def load_kitti_calib(calib_file):
    """
    This script is copied from https://github.com/AI-liu/Complex-YOLO
    """
    with open(calib_file) as f_calib:
        lines = f_calib.readlines()

    P0 = np.array(lines[0].strip('\n').split()[1:], dtype=np.float32)
    P1 = np.array(lines[1].strip('\n').split()[1:], dtype=np.float32)
    P2 = np.array(lines[2].strip('\n').split()[1:], dtype=np.float32)
    P3 = np.array(lines[3].strip('\n').split()[1:], dtype=np.float32)
    R0_rect = np.array(lines[4].strip('\n').split()[1:], dtype=np.float32)
    Tr_velo_to_cam = np.array(lines[5].strip('\n').split()[1:], dtype=np.float32)
    Tr_imu_to_velo = np.array(lines[6].strip('\n').split()[1:], dtype=np.float32)

    return {'P0': P0, 'P1': P1, 'P2': P2, 'P3': P3, 'R0_rect': R0_rect,
            'Tr_velo_to_cam': Tr_velo_to_cam.reshape(3, 4),
            'Tr_imu_to_velo': Tr_imu_to_velo}


def camera_coordinate_to_point_cloud(box3d, Tr):
    """
    This script is copied from https://github.com/AI-liu/Complex-YOLO
    """
    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2 * np.pi + angle
        return angle

    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    # t_lidar: the x, y coordinator of the center of the object
    # box3d_corner: the 8 corners
    return t_lidar, box3d_corner.astype(np.float32)


if __name__ == '__main__':

    # updates
    # IMG_DIR = '/home/dtc/Data/KITTI/data_object_image_2/training/image_2'
    # LABEL_DIR = '/home/dtc/Data/KITTI/data_object_label_2/training/label_2'
    # POINT_CLOUD_DIR = '/home/dtc/Data/KITTI/save'
    # CALIB_DIR = '/home/dtc/Data/KITTI/data_object_calib/training/calib'

    # path to data_object_image_2/training/image_2
    IMG_DIR = './kitti/training/image_2/'
    # path to data_object_velodyne/training/velodyne
    POINT_CLOUD_DIR = './kitti/training/velodyne_reduced/'
    # path to data_object_calib/training/calib
    CALIB_DIR= './kitti/training/calib/'
    LABEL_DIR = './output/3dobj_kitti/20231213-172100/'
    
    
    # id for viewing
    file_id = 53

    img_filename = os.path.join(IMG_DIR, '{0:06d}.png'.format(file_id))
    label_filename = os.path.join(LABEL_DIR, '{0:06d}.txt'.format(file_id))
    pc_filename = os.path.join(POINT_CLOUD_DIR, '{0:06d}.bin'.format(file_id))
    calib_filename = os.path.join(CALIB_DIR, '{0:06d}.txt'.format(file_id))

    # show object in image
    # show_object_in_image(img_filename, label_filename)

    # show object in point cloud
    show_object_in_point_cloud(pc_filename, label_filename, calib_filename)