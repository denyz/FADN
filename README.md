# FADN
This repository contains an implementation of a convolution neural network for 3D object detection in the fusion of lidar point clouds and camera images. As shown in the figure below, the model architecture consists of three major components: frustum-aware generator, decorator, and fusion-based detection network.
![image](https://github.com/denyz/FADN/assets/18696187/1f4fb9e6-e055-4276-a3d4-da1eab007513)

__Note: It is built upon pytorch-geometric and provides usage with the KITTI and nuScenes dataset._
<br>

# Result
Results of our FADN model for 3D object detection on both the [KITTI](https://www.cvlibs.net/datasets/kitti/) and the [nuScenes](https://www.nuscenes.org/) dataset. 
### 3D Object Detection in KITTI and nuScenes
|  KITTI Model  |     mAP       |    Car    |   Pedestrian     | Cyclist  |      
|---------------|---------------|-----------|------------------|----------|
| FAD PV-RCNN   |      66.56    |   83.88   |      48.47       |   69.42  |
|    FADN       |      71.49    |   86.33   |      50.82       |   77.33  |

<br>

|  nuScenes Model  |     mAP       |    Car    |   Ped.  |    Bus   |   Bar. |     TC |   Tru. |  Tra.  | Moto   | C.V    | Bic. |         
|------------------|---------------|-----------|---------|----------|--------|--------|--------|--------|--------|--------|------|
| FAD PointPillars |     35.7      |   72.7    |   66.9  |    31.8  |    44.6|  40.6  |  25.8  |  29.2  |  32.0  |   7.3  | 5.9  |
|      FADN        |     72.09     |   89.1    |   89.9  |    73.2  |   79.6 |  89.5  |  62.1  |  65.3  |  77.5  |   36.1 | 58.6 |



![image](https://github.com/denyz/FADN/assets/18696187/a431a2a8-7faa-46b6-b649-85e3cae15443)


# Prerequisites
- OS: Ubuntu 20.04 LTS
- CUDA: 11.3
- cuDNN: 8
- pytorch 1.10.0

# Preparation
Inside the project folder create a "FADN/data" folder and within this folder, create a "data/output" subfolder. The trained models and evaluations will be stored in that folder. Depending on the desired dataset, create the following additional subfolders inside the "data" folder:
```
datasets/kitti/
datasets/nuscenes/
```
In a second step follow the instructions of the KITTI and nuScenes websites to download and store the datasets in the created subfolders.

Finally, clone this repository into the project folder using the command:

```
git clone https://github.com/denyz/FADN.git
```

<details>
<summary>If you use the KITTI dataset, your folder structure should now look like this: </summary>

```
|  
+---FADN/  
|   |  
|   +---data/  
|   |   |  
|   |   +---kitti/  
|   |   |   |
|   |   |   +---ImageSets/
|   |   |   |	|   +---train.txt
|   |   |   |	|   +---val.txt
|   |   |   |	|   +---test.txt
|   |   |   +---gt_database/
|   |   |   +---trainning/
|   |   |   |	|   +---calib/
|   |   |   |   |   +---image_2
|   |   |   |   |   +---image_3
|   |   |   |   |   +---label_2
|   |   |   |   |   +---planes
|   |   |   |	|   +---velodyne
|   |   |   |	|   +---decorated_lidar
|   |   |   +---testing/
|   |   |   |   |   +---calib/
|   |   |   |   |   +---image_2
|   |   |   |   |   +---image_3
|   |   |   |   |   +---planes
|   |   |   |   |   +---velodyne
|   |   |   |
|   |   |   +---kitti_infos_test.pkl
|   |   |   +---kitti_infos_train.pkl
|   |   |   +---kitti_infos_trainval.pkl
|   |   |
|   +---tools/  
| 
.
.
.
+---...
```
</details>
<br>

## Install
FADN is a LiDAR-camera fusion 3D detection framework. It supports many popular datasets like KITTI and nuscenes. To install the FADN please first install its requirements.txt. And as we modify some parts of the OpenPCDet LIB to support the decorated KITTI dataset. To install it, run the following commands.

```
$ python setup.py develop
```

##  Usage
The overall pipeline is divided into three major steps. 

- Creation of a decorated dataset from the raw KITTI or nuScenes dataset
- Creation and training of a model based on the created decorated dataset
- Evaluation of the trained model

The settings of all three steps are defined in a unified configuration file, which must consequently be created first.
### 1. Create a decorated dataset
the decorated dataset needs to be created by converting the lidar point clouds of the raw datasets to a decorated data structure. This will generate the decorated_lidar folder in the dataset. To do this, execute the following command inside the docker container: 
```
$ cd FADN
$ python tools/decorating.py
```
```
usage:          decorating.py [--dataset] [--config]

arguments:
    --dataset   Path to the raw (RadarScenes/nuScenes) dataset.
    --config  Parameters to the created decorated dataset.
```

Create the KITTI PKL
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/decorate_kitti_dataset.yaml
```

### 2. Create and train a model
Next step, you can use the created decorated dataset to train a model. To do this, run the following command: 
```
$ python -m pcdet.datasets.kitti.decorate_kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/decorate_kitti_dataset.yaml
$ cd tools
```

```
usage:   train.py [--data] [--results] [--config]
$ python tools/train.py --cfg_file cfgs/kitti_models/FADN_decorated.yaml
```

### 3. Evaluate a KITTI trained model 
Finally, you can evaluate a trained model using the following command in **kittiEval**:
```
usage:   ./eval_detection_3d_offline [gt_dir] [result_dir]
```
The evaluation metrics include :    
- Overlap on image (AP)
- Oriented overlap on image (AOS)
- Overlap on ground-plane (AP)
- Overlap in 3D (AP)

Within the provided "results" folder a new "data" and "plot" folder is created, in which the evaluation results are saved.

