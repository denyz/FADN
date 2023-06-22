import random

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import os
from PIL import Image
from tqdm import tqdm
from cv2 import cv2

from mmseg.apis import inference_segmentor, init_segmentor
import pcdet_utils.calibration_kitti as calibration_kitti

# for yolo
import cv2 as cv

TRAINING_PATH = "../detector/data/kitti/training/"
TESTING_PATH = "../detector/data/kitti/testing/"
TWO_CAMERAS = True
model=torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local',device='0')


class Painter:
    def __init__(self, seg_net_index):
        self.root_split_path = TRAINING_PATH
        self.save_path = TRAINING_PATH + "painted_lidar/"

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.seg_net_index = seg_net_index
        self.model = None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path + 'velodyne/' + ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_score(self, idx, left):
        ''' idx : index string
            left : string indicates left/right camera
        return:
            a tensor H  * W * 4(deeplab)/5(deeplabv3plus), for each pixel we have 4/5 scorer that sums to 1
        '''
        output_reassign_softmax = None
        if self.seg_net_index == 0:
            filename = self.root_split_path + left + ('%s.png' % idx)
            input_image = Image.open(filename)
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')

            with torch.no_grad():
                output = self.model(input_batch)['out'][0]

            output_permute = output.permute(1, 2, 0)#(21，370，1224）to（370，1224，21）
            output_probability, output_predictions = output_permute.max(2)

            other_object_mask = ~((output_predictions == 0) | (output_predictions == 2) | (output_predictions == 7) | (
                        output_predictions == 15))
            detect_object_mask = ~other_object_mask
            sf = torch.nn.Softmax(dim=2)

            output_reassign = torch.zeros(output_permute.size(0), output_permute.size(1), 4)

            output_reassign[:, :, 0] = detect_object_mask * output_permute[:, :, 0] + other_object_mask * output_probability
            output_reassign[:, :, 1] = output_permute[:, :, 2]  # bicycle
            output_reassign[:, :, 2] = output_permute[:, :, 7]  # car
            output_reassign[:, :, 3] = output_permute[:, :, 15]  # person

            output_reassign_softmax = sf(output_reassign).cpu().numpy()

        elif self.seg_net_index == 1:
            filename = self.root_split_path + left + ('%s.png' % idx)

            result = inference_segmentor(self.model, filename)

            # ID：person 11, rider 12, vehicle 13/14/15/16, bike 17/18
            # cityscapes car 13，truck 14,bus 15,train 16
            output_permute = torch.tensor(result[0]).permute(1, 2, 0)  # H, W, 19


            output_reassign = torch.zeros(output_permute.size(0), output_permute.size(1), 5)
            output_reassign[:, :, 0], _ = torch.max(output_permute[:, :, :11], dim=2)  # background
            output_reassign[:, :, 1], _ = torch.max(output_permute[:, :, [17, 18]], dim=2)  # bicycle
            output_reassign[:, :, 2], _ = torch.max(output_permute[:, :, [13, 14, 15, 16]], dim=2)  # car
            output_reassign[:, :, 3] = output_permute[:, :, 11]  # person
            output_reassign[:, :, 4] = output_permute[:, :, 12]  # rider
            sf = torch.nn.Softmax(dim=2)
            output_reassign_softmax = sf(output_reassign).cpu().numpy()

        elif self.seg_net_index == 2:
            filename = self.root_split_path + "score_hma/" + left + ('%s.npy' % idx)
            output_reassign_softmax = np.load(filename)

        # person in output_reassign_softmax[:,:,3]
        return output_reassign_softmax

    def scores_rgb(self, idx, left):
        filename = self.root_split_path + left + ('%s.png' % idx)
        input_image = Image.open(filename)

        r, g, b = input_image.split()
        r_numpy = np.array(r)
        g_numpy = np.array(g)
        b_numpy = np.array(b)

        scores_from_cam = np.zeros([input_image.height, input_image.width, 4], np.float16)

        # sum_scores_from_cam=scores_from_cam[:,:,1]+scores_from_cam[:,:,2]+scores_from_cam[:,:,3]+scores_from_cam[:,:,4]

        # sum = sum_scores_from_cam >= 0.7
        # scores_from_cam[:, :, 0] = 0
        # scores_from_cam[:, :, 1] = sum*(r_numpy/256)
        # scores_from_cam[:, :, 2] = sum*(g_numpy/256)
        # scores_from_cam[:, :, 3] = sum*(b_numpy/256)
        # scores_from_cam[:, :, 4] = 0

        if 1:

            scores_from_cam[:, :, 0] = 0
            scores_from_cam[:, :, 1] = r_numpy / 256
            scores_from_cam[:, :, 2] = g_numpy / 256
            scores_from_cam[:, :, 3] = b_numpy / 256
            # scores_from_cam[:, :, 4] = 0
        else:

            scores_from_cam[:, :, :] = 0
            scores_from_cam[:, :, 1] = (r_numpy / 256) * detectionMask
            scores_from_cam[:, :, 2] = (g_numpy / 256) * detectionMask
            scores_from_cam[:, :, 3] = (b_numpy / 256) * detectionMask

        tmp = scores_from_cam[:,:,1:4] * 256
        imageshow = Image.fromarray(tmp.astype('uint8')).convert('RGB')
        #imageshow.show()

        return scores_from_cam


    def paint_global_RGB(self, idx, left):
        ''' idx : index string
                   left : string indicates left/right camera
               return:
                   a tensor H  * W * 4, for each pixel we have 4/5 scorer that sums to 1
               '''

        filename = self.root_split_path + left + ('%s.png' % idx)
        input_image = Image.open(filename)
        imgArray = np.array(input_image)  # PIL to numpy

        # Inference
        results = model(input_image, size=640)  # includes NMS


        img = np.zeros([input_image.height, input_image.width, 4], np.float16)

        # YOLO-V5，0：person，  1：bicycle，3：motocycle，  2：car    5：bus，6：train，7：truck

        img[:, :, :] = 0
        img[:, :, 1:] = imgArray[:, :, :] / 255  # create_cyclist

        return img

    def paint_DetEllipse_RGB(self, idx, left):
        ''' idx : index string
            left : string indicates left/right camera
        return:
            a tensor H  * W * 4, for each pixel we have 4/5 scorer that sums to 1
        '''

        filename = self.root_split_path + left + ('%s.png' % idx)
        input_image = Image.open(filename)
        imgArray = np.array(input_image)#PIL to numpy
        # Inference
        results = model(input_image, size=640)  # includes NMS


        imageout = np.zeros([input_image.height, input_image.width, 4], np.float16)
        mask = np.zeros([input_image.height, input_image.width, 3], np.int8)



        for i in range(len(results.pandas().xyxy[0])):
            objectName = results.pandas().xyxy[0].name[i]
            if objectName == 'Cyclist': #'bicycle' or objectName == 'motocycle':
                # 归为单车类
                bicycleBoxs = results.pandas().xyxy[0].iloc[i, :]

                centerPoint_x = (bicycleBoxs.xmax + bicycleBoxs.xmin) / 2
                centerPoint_y = (bicycleBoxs.ymax + bicycleBoxs.ymin) / 2

                Ellipse_a = (bicycleBoxs.xmax - bicycleBoxs.xmin)/2
                Ellipse_b = (bicycleBoxs.ymax - bicycleBoxs.ymin)/2

                # 在mask上用椭圆画出目标位置
                cv2.ellipse(mask, (int(centerPoint_x),int(centerPoint_y)), (int(Ellipse_a),int(Ellipse_b)), 0, 0, 360, (1, 1, 1), -1)


            elif objectName == 'Car':#or objectName == 'bus' or objectName == 'train' or objectName == 'truck':
                # 归为car类
                carBoxs = results.pandas().xyxy[0].iloc[i, :]

                centerPoint_x = (carBoxs.xmax + carBoxs.xmin) / 2
                centerPoint_y = (carBoxs.ymax + carBoxs.ymin) / 2

                Ellipse_a = (carBoxs.xmax - carBoxs.xmin)/2
                Ellipse_b = (carBoxs.ymax - carBoxs.ymin)/2

                cv2.ellipse(mask, (int(centerPoint_x),int(centerPoint_y)), (int(Ellipse_a),int(Ellipse_b)), 0, 0, 360, (1, 1, 1), -1)



            elif objectName == 'Pedestrian': #:'person':
                # pedestrian
                personBoxs = results.pandas().xyxy[0].iloc[i, :]

                centerPoint_x = (personBoxs.xmax + personBoxs.xmin) / 2
                centerPoint_y = (personBoxs.ymax + personBoxs.ymin) / 2

                Ellipse_a = (personBoxs.xmax - personBoxs.xmin)/2
                Ellipse_b = (personBoxs.ymax - personBoxs.ymin)/2


                cv2.ellipse(mask, (int(centerPoint_x),int(centerPoint_y)), (int(Ellipse_a),int(Ellipse_b)), 0, 0, 360, (1, 1, 1), -1)

        tmp = mask * imgArray
        imageshow = Image.fromarray(tmp.astype('uint8')).convert('RGB')

        imgArray = imgArray/255
        imageout[:,:,1:] = mask * imgArray

        return imageout

    def paint_DetBBox_RGB(self, idx, left):  #使用矩形框的RGB值赋值
        ''' idx : index string
            left : string indicates left/right camera
        return:
            a tensor H  * W * 4, for each pixel we have 4/5 scorer that sums to 1
        '''

        filename = self.root_split_path + left + ('%s.png' % idx)
        input_image = Image.open(filename)
        imgArray = np.array(input_image)#PIL to numpy
        # Inference
        results = model(input_image, size=640)  # includes NMS


        img = np.zeros([input_image.height, input_image.width, 3], np.uint8) #np.float16)

        img[:, :, :] = 0

        for i in range(len(results.pandas().xyxy[0])):
            objectName = results.pandas().xyxy[0].name[i]
            if objectName == 'Cyclist': #'bicycle' or objectName == 'motocycle':
                # 归为单车类
                bicycleBoxs = results.pandas().xyxy[0].iloc[i, :]
                # confidence
                # img[int(bicycleBoxs.ymin):int(bicycleBoxs.ymax), int(bicycleBoxs.xmin):int(bicycleBoxs.xmax), 1] = bicycleBoxs.confidence
                img[int(bicycleBoxs.ymin):int(bicycleBoxs.ymax), int(bicycleBoxs.xmin):int(bicycleBoxs.xmax), :] = \
                    imgArray[int(bicycleBoxs.ymin):int(bicycleBoxs.ymax), int(bicycleBoxs.xmin):int(bicycleBoxs.xmax),:]

            elif objectName == 'Car':#or objectName == 'bus' or objectName == 'train' or objectName == 'truck':
                # 归为car类
                carBoxs = results.pandas().xyxy[0].iloc[i, :]
                # confidence
                # img[int(carBoxs.ymin):int(carBoxs.ymax), int(carBoxs.xmin):int(carBoxs.xmax), 2] = 1#carBoxs.confidence
                img[int(carBoxs.ymin):int(carBoxs.ymax), int(carBoxs.xmin):int(carBoxs.xmax), :] = \
                    imgArray[int(carBoxs.ymin):int(carBoxs.ymax), int(carBoxs.xmin):int(carBoxs.xmax), :]

            elif objectName == 'Pedestrian': #:'person':
                # 归为pedestrian类
                personBoxs = results.pandas().xyxy[0].iloc[i, :]
                # confidence
                # img[int(personBoxs.ymin):int(personBoxs.ymax), int(personBoxs.xmin):int(personBoxs.xmax), 3] = personBoxs.confidence
                img[int(personBoxs.ymin):int(personBoxs.ymax), int(personBoxs.xmin):int(personBoxs.xmax), :] = \
                    imgArray[int(personBoxs.ymin):int(personBoxs.ymax), int(personBoxs.xmin):int(personBoxs.xmax),:]

        #FOR show
        imageshow = Image.fromarray(img.astype('uint8')).convert('RGB')
        #imageshow.show()

        imgout = np.zeros([input_image.height, input_image.width, 4], np.float16)
        imgout[:,:,1:] = img[:,:,:]/255

        return imgout #img

    def paint_DetBBox_Score(self, idx, left):
        ''' idx : index string
            left : string indicates left/right camera
        return:
            a tensor H  * W * 4, for each pixel we have 4/5 scorer that sums to 1
        '''

        filename = self.root_split_path + left + ('%s.png' % idx)
        input_image = Image.open(filename)
        imgArray = np.array(input_image)#PIL to numpy
        # Inference
        results = model(input_image, size=640)  # includes NMS

        img = np.zeros([input_image.height, input_image.width, 4], np.float16)

        img[:, :, :] = 0

        for i in range(len(results.pandas().xyxy[0])):  # i为目标框索引
            objectName = results.pandas().xyxy[0].name[i]
            if objectName == 'bicycle' or objectName == 'motocycle':

                bicycleBoxs = results.pandas().xyxy[0].iloc[i, :]
                # confidence
                img[int(bicycleBoxs.ymin):int(bicycleBoxs.ymax), int(bicycleBoxs.xmin):int(bicycleBoxs.xmax), 1] = bicycleBoxs.confidence #试用不同类别的置信度score来增广点云

            elif objectName == 'car' or objectName == 'bus' or objectName == 'train' or objectName == 'truck':

                carBoxs = results.pandas().xyxy[0].iloc[i, :]
                # confidence
                img[int(carBoxs.ymin):int(carBoxs.ymax), int(carBoxs.xmin):int(carBoxs.xmax), 2] = carBoxs.confidence

            elif objectName == 'person':

                personBoxs = results.pandas().xyxy[0].iloc[i, :]
                # confidence
                img[int(personBoxs.ymin):int(personBoxs.ymax), int(personBoxs.xmin):int(personBoxs.xmax), 3] = personBoxs.confidence

        return img

    def paint_DetBBox_Mask(self, idx, left):
        ''' idx : index string
            left : string indicates left/right camera
        return:
            a tensor H  * W * 4, for each pixel we have 4/5 scorer that sums to 1
        '''

        filename = self.root_split_path + left + ('%s.png' % idx)
        input_image = Image.open(filename)
        imgArray = np.array(input_image)#PIL to numpy
        # Inference
        results = model(input_image, size=640)  # includes NMS

        img = np.zeros([input_image.height, input_image.width, 4], np.float16)

        img[:, :, :] = 0

        for i in range(len(results.pandas().xyxy[0])):
            objectName = results.pandas().xyxy[0].name[i]
            if objectName == 'Cyclist': #'bicycle' or objectName == 'motocycle':
                # 归为单车类
                bicycleBoxs = results.pandas().xyxy[0].iloc[i, :]
                # confidence
                img[int(bicycleBoxs.ymin):int(bicycleBoxs.ymax), int(bicycleBoxs.xmin):int(bicycleBoxs.xmax), 1] = 0.6

            elif objectName == 'Car': #'car' or objectName == 'bus' or objectName == 'train' or objectName == 'truck':
                # 归为car类
                carBoxs = results.pandas().xyxy[0].iloc[i, :]
                # confidence
                img[int(carBoxs.ymin):int(carBoxs.ymax), int(carBoxs.xmin):int(carBoxs.xmax), 2] = 0.7

            elif objectName == 'Pedestrian': #'person':
                # 归为pedestrian类
                personBoxs = results.pandas().xyxy[0].iloc[i, :]
                # confidence
                img[int(personBoxs.ymin):int(personBoxs.ymax), int(personBoxs.xmin):int(personBoxs.xmax), 3] = 0.8

        return img

    def get_calib(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        return calibration_kitti.Calibration(calib_file)

    def get_calib_fromfile(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        calib = calibration_kitti.get_calib_from_file(calib_file)
        calib['P2'] = np.concatenate([calib['P2'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['P3'] = np.concatenate([calib['P3'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['R0_rect'] = np.zeros([4, 4], dtype=calib['R0'].dtype)
        calib['R0_rect'][3, 3] = 1.
        calib['R0_rect'][:3, :3] = calib['R0']
        calib['Tr_velo2cam'] = np.concatenate([calib['Tr_velo2cam'], np.array([[0., 0., 0., 1.]])], axis=0)
        return calib

    def create_cyclist(self, augmented_lidar):
        if self.seg_net_index == 3:
            bike_idx = np.where(augmented_lidar[:, 5] >= 0.2)[0]  # 0, 1(bike), 2, 3(person)
            bike_points = augmented_lidar[bike_idx]
            cyclist_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
            for i in range(bike_idx.shape[0]):
                cyclist_mask = (np.linalg.norm(augmented_lidar[:, :3] - bike_points[i, :3], axis=1) < 1) & (
                        np.argmax(augmented_lidar[:, -4:], axis=1) == 3)
                if np.sum(cyclist_mask) > 0:
                    cyclist_mask_total |= cyclist_mask
                else:
                    augmented_lidar[bike_idx[i], 4], augmented_lidar[bike_idx[i], 5] = augmented_lidar[
                                                                                           bike_idx[i], 5], 0
            augmented_lidar[cyclist_mask_total, 7], augmented_lidar[cyclist_mask_total, 5] = 0, augmented_lidar[
                cyclist_mask_total, 7]

            return  augmented_lidar

        if self.seg_net_index == 0:
            bike_idx = np.where(augmented_lidar[:, 5] >= 0.2)[0]  # 0, 1(bike), 2, 3(person)
            bike_points = augmented_lidar[bike_idx]
            cyclist_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
            for i in range(bike_idx.shape[0]):
                cyclist_mask = (np.linalg.norm(augmented_lidar[:, :3] - bike_points[i, :3], axis=1) < 1) & (
                            np.argmax(augmented_lidar[:, -4:], axis=1) == 3)
                if np.sum(cyclist_mask) > 0:
                    cyclist_mask_total |= cyclist_mask
                else:
                    augmented_lidar[bike_idx[i], 4], augmented_lidar[bike_idx[i], 5] = augmented_lidar[
                                                                                           bike_idx[i], 5], 0
            augmented_lidar[cyclist_mask_total, 7], augmented_lidar[cyclist_mask_total, 5] = 0, augmented_lidar[
                cyclist_mask_total, 7]
            return augmented_lidar
        if self.seg_net_index == 1 or 2:
            rider_idx = np.where(augmented_lidar[:, 8] >= 0.3)[0]  # 0, 1(bike), 2, 3(person), 4(rider)
            rider_points = augmented_lidar[rider_idx]
            bike_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
            bike_total = (np.argmax(augmented_lidar[:, -5:], axis=1) == 1)
            for i in range(rider_idx.shape[0]):
                bike_mask = (np.linalg.norm(augmented_lidar[:, :3] - rider_points[i, :3], axis=1) < 1) & bike_total
                bike_mask_total |= bike_mask
            augmented_lidar[bike_mask_total, 8] = augmented_lidar[bike_mask_total, 5]
            augmented_lidar[bike_total ^ bike_mask_total, 4] = augmented_lidar[bike_total ^ bike_mask_total, 5]
            return augmented_lidar[:, [0, 1, 2, 3, 4, 8, 6, 7]]


    def cam_to_lidar(self, pointcloud, projection_mats):
        """
        Takes in lidar in velo coords, returns lidar points in camera coords

        :param pointcloud: (n_points, 4) np.array (x,y,z,r) in velodyne coordinates
        :return lidar_cam_coords: (n_points, 4) np.array (x,y,z,r) in camera coordinates

        """

        lidar_velo_coords = copy.deepcopy(pointcloud)
        reflectances = copy.deepcopy(lidar_velo_coords[:, -1])  # copy reflectances column
        lidar_velo_coords[:, -1] = 1  # for multiplying with homogeneous matrix
        lidar_cam_coords = projection_mats['Tr_velo2cam'].dot(lidar_velo_coords.transpose())
        lidar_cam_coords = lidar_cam_coords.transpose()
        lidar_cam_coords[:, -1] = reflectances

        return lidar_cam_coords

    def augment_lidar_RGB(self, class_scores_r, class_scores_l, lidar_raw, projection_mats):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        # lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        # TODO: Project lidar points onto left and right segmentation maps. How to use projection_mats?
        ##########return lidar_cam_coords: lidar_cam_coords (x,y,z,r) in camera coordinates,
        lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)

        lidar_cam_coords[:, -1] = 1  # homogenous coords for projection,
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?   #points_projected_on_mask_r表示点云在mask上的坐标，以图像左上角为原点的坐标系
        # 其中，lidar_cam_coords  = Tr_velo2cam * lidar_velo_coords.transpose()
        points_projected_on_mask_r = projection_mats['P3'].dot(
            projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_r = points_projected_on_mask_r.transpose()
        points_projected_on_mask_r = points_projected_on_mask_r / (points_projected_on_mask_r[:, 2].reshape(-1, 1))

        true_where_x_on_img_r = (0 < points_projected_on_mask_r[:, 0]) & (
                    points_projected_on_mask_r[:, 0] < class_scores_r.shape[1])  # x in img coords is cols of img
        true_where_y_on_img_r = (0 < points_projected_on_mask_r[:, 1]) & (
                    points_projected_on_mask_r[:, 1] < class_scores_r.shape[0])
        true_where_point_on_img_r = true_where_x_on_img_r & true_where_y_on_img_r

        points_projected_on_mask_r = points_projected_on_mask_r[
            true_where_point_on_img_r]  # filter out points that don't project to image
        points_projected_on_mask_r = np.floor(points_projected_on_mask_r).astype(
            int)  # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_r = points_projected_on_mask_r[:,
                                     :2]

        # left相机
        lidar_cam_coords[:, -1] = 1  # homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_l = projection_mats['P2'].dot(
            projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_l = points_projected_on_mask_l.transpose()
        points_projected_on_mask_l = points_projected_on_mask_l / (points_projected_on_mask_l[:, 2].reshape(-1, 1))

        true_where_x_on_img_l = (0 < points_projected_on_mask_l[:, 0]) & (
                    points_projected_on_mask_l[:, 0] < class_scores_l.shape[1])  # x in img coords is cols of img
        true_where_y_on_img_l = (0 < points_projected_on_mask_l[:, 1]) & (
                    points_projected_on_mask_l[:, 1] < class_scores_l.shape[0])
        true_where_point_on_img_l = true_where_x_on_img_l & true_where_y_on_img_l

        points_projected_on_mask_l = points_projected_on_mask_l[
            true_where_point_on_img_l]
        points_projected_on_mask_l = np.floor(points_projected_on_mask_l).astype(
            int)
        points_projected_on_mask_l = points_projected_on_mask_l[:,
                                     :2]  # drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        true_where_point_on_both_img = true_where_point_on_img_l & true_where_point_on_img_r
        true_where_point_on_img = true_where_point_on_img_l | true_where_point_on_img_r

        # indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        # socre dimesion: point_scores.shape[2] TODO!!!!
        point_scores_r = class_scores_r[points_projected_on_mask_r[:, 1], points_projected_on_mask_r[:, 0]].reshape(-1,
                                                                                                                    class_scores_r.shape[
                                                                                                                        2])
        point_scores_l = class_scores_l[points_projected_on_mask_l[:, 1], points_projected_on_mask_l[:, 0]].reshape(-1,
                                                                                                                    class_scores_l.shape[
                                                                                                                        2])
        # augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores), axis=1)
        augmented_lidar = np.concatenate((lidar_raw, np.zeros((lidar_raw.shape[0], class_scores_r.shape[2]))), axis=1)
        augmented_lidar[true_where_point_on_img_r, -class_scores_r.shape[2]:] += point_scores_r
        augmented_lidar[true_where_point_on_img_l, -class_scores_l.shape[2]:] += point_scores_l
        augmented_lidar[true_where_point_on_both_img, -class_scores_r.shape[2]:] = 0.5 * augmented_lidar[
                                                                                         true_where_point_on_both_img,
                                                                                         -class_scores_r.shape[2]:]
        augmented_lidar = augmented_lidar[true_where_point_on_img]

        augmented_lidar = augmented_lidar[:, 0:8]

        return augmented_lidar

    def augment_lidar_class_scores_both(self, class_scores_r, class_scores_l, lidar_raw, projection_mats, create_rider):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        # lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        # TODO: Project lidar points onto left and right segmentation maps. How to use projection_mats?
        lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)

        # right相机
        lidar_cam_coords[:, -1] = 1  # homogenous coords for projection,
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?   #points_projected_on_mask_r表示点云在mask上的二维坐标，以图像左上角为原点的坐标系
        # 其中，lidar_cam_coords  = Tr_velo2cam * lidar_velo_coords.transpose()
        points_projected_on_mask_r = projection_mats['P3'].dot(
            projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_r = points_projected_on_mask_r.transpose()
        points_projected_on_mask_r = points_projected_on_mask_r / (points_projected_on_mask_r[:, 2].reshape(-1, 1))

        true_where_x_on_img_r = (0 < points_projected_on_mask_r[:, 0]) & (
                    points_projected_on_mask_r[:, 0] < class_scores_r.shape[1])  # x in img coords is cols of img
        true_where_y_on_img_r = (0 < points_projected_on_mask_r[:, 1]) & (
                    points_projected_on_mask_r[:, 1] < class_scores_r.shape[0])  # y in img coords is rows of img
        true_where_point_on_img_r = true_where_x_on_img_r & true_where_y_on_img_r

        points_projected_on_mask_r = points_projected_on_mask_r[
            true_where_point_on_img_r]  # filter out points that don't project to image，
        points_projected_on_mask_r = np.floor(points_projected_on_mask_r).astype(
            int)  # using floor so you don't end up indexing num_rows+1th row or col，
        points_projected_on_mask_r = points_projected_on_mask_r[:,
                                     :2]

        # left
        lidar_cam_coords[:, -1] = 1  # homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_l = projection_mats['P2'].dot(
            projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_l = points_projected_on_mask_l.transpose()
        points_projected_on_mask_l = points_projected_on_mask_l / (points_projected_on_mask_l[:, 2].reshape(-1, 1))

        true_where_x_on_img_l = (0 < points_projected_on_mask_l[:, 0]) & (
                    points_projected_on_mask_l[:, 0] < class_scores_l.shape[1])  # x in img coords is cols of img
        true_where_y_on_img_l = (0 < points_projected_on_mask_l[:, 1]) & (
                    points_projected_on_mask_l[:, 1] < class_scores_l.shape[0])
        true_where_point_on_img_l = true_where_x_on_img_l & true_where_y_on_img_l

        points_projected_on_mask_l = points_projected_on_mask_l[
            true_where_point_on_img_l]  # filter out points that don't project to image，
        points_projected_on_mask_l = np.floor(points_projected_on_mask_l).astype(
            int)  # using floor so you don't end up indexing num_rows+1th row or col，
        points_projected_on_mask_l = points_projected_on_mask_l[:,
                                     :2]  # drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        true_where_point_on_both_img = true_where_point_on_img_l & true_where_point_on_img_r
        true_where_point_on_img = true_where_point_on_img_l | true_where_point_on_img_r

        # indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        # socre dimesion: point_scores.shape[2] TODO!!!!
        point_scores_r = class_scores_r[points_projected_on_mask_r[:, 1], points_projected_on_mask_r[:, 0]].reshape(-1,
                                                                                                                    class_scores_r.shape[
                                                                                                                        2])
        point_scores_l = class_scores_l[points_projected_on_mask_l[:, 1], points_projected_on_mask_l[:, 0]].reshape(-1,
                                                                                                                    class_scores_l.shape[
                                                                                                                        2])
        # augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores), axis=1)
        augmented_lidar = np.concatenate((lidar_raw, np.zeros((lidar_raw.shape[0], class_scores_r.shape[2]))), axis=1)
        augmented_lidar[true_where_point_on_img_r, -class_scores_r.shape[2]:] += point_scores_r
        augmented_lidar[true_where_point_on_img_l, -class_scores_l.shape[2]:] += point_scores_l


        augmented_lidar[true_where_point_on_both_img, -class_scores_r.shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_img,-class_scores_r.shape[2]:]

        augmented_lidar = augmented_lidar[true_where_point_on_img]

        if create_rider == 'True':
            augmented_lidar = self.create_cyclist(augmented_lidar)

        return augmented_lidar


    def pointCloudInterpolation(self, dataIn, dis_thred):
        dataOut = dataIn[:, 0:3]
        dataOut = dataOut[dataOut[:, 0] > 0]

        dataInNum = len(dataOut) - 1
        for i in range(dataInNum):
            for j in range(dataInNum):
                if i + j < dataInNum and dataOut[i, 2] > -1.56 and dataOut[i, 0] > 0:
                    dist = math.sqrt(
                        (dataOut[i, 0] - dataOut[i + j, 0]) ** 2 + (dataOut[i, 1] - dataOut[i + j, 1]) ** 2 + (
                                dataOut[i, 2] - dataOut[i + j, 2]) ** 2)
                    if (dist > 0.01) and (dist < dis_thred):
                        pointPos = np.array([((dataOut[i, 0] + dataOut[i + j, 0]) / 2,
                                              (dataOut[i, 1] + dataOut[i + j, 1]) / 2,
                                              (dataOut[i, 2] + dataOut[i + j, 2]) / 2)])
                        dataOut = np.concatenate((dataOut, pointPos), axis=0)
                else:
                    continue
        return dataOut

    def RandomSphInterp(self, dataIn, radius):
        N = 10  # 随机插值点数量
        dataOut = dataIn[:, :]

        dataOut = dataOut[dataOut[:, 0] > 0]  #
        dataInNum = len(dataOut) - 1
        npzero = np.zeros(N, dtype=float)
        for i in range(dataInNum):
            sita = np.random.uniform(low=0, high=math.pi, size=N)
            phi = np.random.uniform(low=0, high=2 * math.pi, size=N)

            x = radius * np.sin(sita) * np.cos(phi)
            y = radius * np.sin(sita) * np.sin(phi)
            z = radius * np.cos(sita)

            pointPos = np.array([dataOut[i, 0] + x,
                                 dataOut[i, 1] + y,
                                 dataOut[i, 2] + z])
            tmp = dataOut[i, 3:].reshape(dataOut[i, 3:].shape[0], 1)
            DeltaPos = np.concatenate((pointPos, tmp.repeat(N, axis=1)), axis=0)
            dataOut = np.concatenate((dataOut, DeltaPos.transpose()), axis=0)

        return dataOut

    def pointCloudInterpolation2(self, dataIn, dis_thred):
        dataOut = dataIn[:, 0:3]
        dataRet = dataOut
        dataOut = dataOut[dataOut[:, 0] > 0]
        dataOut = dataOut[dataOut[:, 2] > -1.56]
        dataInNum = len(dataOut) - 1
        tmp = []
        if dataInNum > 0:
            for i in range(dataInNum):

                oneRow = dataOut[i, :]
                oneRow = oneRow.reshape(1, -1)
                dataRow = oneRow.repeat(dataInNum + 1, axis=0)

                tmp = abs(dataRow + (-1) * dataOut)
                tmp = tmp ** 2
                rowSum = tmp[:, 0] + tmp[:, 1] + tmp[:, 2]
                dist = np.sqrt(rowSum)

                middlePix = (dataRow + dataOut) / 2

                dist = dist.reshape(-1, 1)
                pointPos = middlePix[dist[:, 0] < dis_thred, :]


                tmp = np.concatenate((tmp, pointPos), axis=0)

                dataRet = np.concatenate((dataRet, tmp), axis=0)

        return dataRet

    def genDelta_PC(self, points):

        points_other = points[points[:, 4] > 0]
        points_bicycle = points[points[:, 5] > 0]
        points_car = points[points[:, 6] > 0]
        points_person = points[points[:, 7] > 0]  # 过滤出人和骑单车的人类别

        tmp = np.concatenate((points_bicycle, points_other), axis=0)
        tmp = np.concatenate((tmp, points_car), axis=0)
        delta_points = np.concatenate((tmp, points_person), axis=0)

        # delta_pc = self.pointCloudInterpolation(delta_points[:,0:3], 0.3)
        # delta_pc = np.zeros([0,delta_points.shape[1]])
        delta_pc = self.RandomSphInterp(delta_points, 0.1)

        return delta_pc

    def runDetInstruc(self):
        num_image = 7481 #tain3712 + val3769 = 7481， testSet 7518
        for idx in tqdm(range(num_image)):
            #idx = 8

            sample_idx = "%06d" % idx
            points = self.get_lidar(sample_idx) #115384 points

            fileName = TRAINING_PATH + (
                    "PointCloud_csv_original/%06d.csv" % idx)
            np.savetxt(fileName, points, fmt='%10.7f', delimiter=',')

            result_from_cam = self.paint_DetEllipse_RGB(sample_idx, "image_2/")
            result_from_cam_r = self.paint_DetEllipse_RGB(sample_idx, "image_3/")

            # get calibration data
            calib_fromfile = self.get_calib_fromfile(sample_idx)


            # points = self.pointCloudInterpolation(points, 0.3)
            # fileName = TRAINING_PATH + ("PointCloud_csv_aug/%06d.csv" % idx)
            # np.savetxt(fileName, points, fmt='%10.7f', delimiter=',')

            # paint the point clouds
            # points = self.augment_lidar_class_scores_both(scores_from_cam_r, scores_from_cam, points, calib_fromfile)
            points = self.augment_lidar_class_scores_both(result_from_cam_r, result_from_cam, points, calib_fromfile, 'False') #33896 points

            fileName = TRAINING_PATH + ("PointCloud_csv_original_painted/%06d.csv" % idx)
            np.savetxt(fileName, points, fmt='%10.7f', delimiter=',')

            delta_pc = self.genDelta_PC(points)
                    #z = np.zeros(5)
                    #z = z.reshape(1, 5)
                    #z = z.repeat(len(delta_pc), axis=0)
                    #delta_pc = np.concatenate((delta_pc, z), axis=1)
            points = np.concatenate((points, delta_pc), axis=0)

            fileName = TRAINING_PATH + ("PointCloud_csv_dense/%06d.csv" % idx)
            #np.savetxt(fileName, points, fmt='%10.7f', delimiter=',')


            # yolov5
            result_from_cam = self.paint_DetEllipse_RGB(sample_idx, "image_2/")
            result_from_cam_r = self.paint_DetEllipse_RGB(sample_idx, "image_3/")



            # get calibration data
            #calib_fromfile = self.get_calib_fromfile(sample_idx)
            points = points[:,0:4]

            points = self.augment_lidar_class_scores_both(result_from_cam_r, result_from_cam, points, calib_fromfile,'False')  # 33896个点

            # 保存成csv文件
            fileName = TRAINING_PATH + ("PointCloud_csv_dense_painted/%06d.csv" % idx)
            np.savetxt(fileName, points, fmt='%10.7f', delimiter=',')


            np.save(self.save_path + ("%06d.npy" % idx), points)



if __name__ == '__main__':
    painter = Painter(SEG_NET)
    painter.runDetInstruc()