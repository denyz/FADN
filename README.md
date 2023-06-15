# FADN
This repository contains an implementation of a convolution neural network for 3D object detection in the fusion of lidar point clouds and camera images As shown in the figure below, the model architecture consists of three major components: Frustum-Aware Generator, decorator, and fussion-based detection network.
![image](https://github.com/denyz/FADN/assets/18696187/1f4fb9e6-e055-4276-a3d4-da1eab007513)

It is built upon pytorch-geometric and provides usage with the KITTI and nuScenes dataset.


# Result

Results of our FADN model for 3D object detection on both the [KITTI](https://www.cvlibs.net/datasets/kitti/) and the [nuScenes](https://www.nuscenes.org/) dataset. 
### 3D Object Detection in KITTI and nuScenes
|  KITTI Model  |     mAP       |    Car    |   Pedestrian     | Cyclist  |      
|---------------|---------------|-----------|------------------|----------|
| FAD PV-RCNN   |      66.56    |   83.88   |      48.47       |   69.42  |
|    FADN       |      71.49    |   86.33   |      50.82       |   77.33  |

<br>

|  nuScenes Model  |     mAP       |    Car    |   Ped.  |    Bus   |   Bar. |     TC |   Tru. |  Tra.  | Moto   | C.V    | Bic.           
|------------------|---------------|-----------|---------|----------|--------|--------|--------|--------|--------|--------|------|
| FAD PointPillars |     35.7      |   72.7    |    66.9 |    31.8  |    44.6|  40.6  |  25.8  |  29.2  |  32.0  |   7.3  | 5.9  |
|      FADN        |      72.09    |   89.1    |   89.9  |    73.2  |   79.6 |   89.5 |  62.1  |   65.3 |   77.5 |   36.1 | 58.6 |



![image](https://github.com/denyz/FADN/assets/18696187/a431a2a8-7faa-46b6-b649-85e3cae15443)


# Prerequisites
- OS: Ubuntu 20.04 LTS
- CUDA: 11.3
- cuDNN: 8
- pytorch 1.10.0

# Preparation
Inside the project folder create a "data" folder and within this folder, create a "results" subfolder. The trained models and evaluations will be stored in that folder. Depending on the desired dataset, create the following additional sub folders inside the "data" folder:

datasets/radarscenes/raw
datasets/nuscenes
In a second step follow the instructions of the nuScenes and/or RadarScenes websites to download and store the datasets in the created sub folders.

Finally, clone this repository into the project folder using the command:

```
git clone https://github.com/denyz/FADN.git
```

Create the PKL
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

<details>
<summary>If you use the KITTI dataset, your folder structure should now look like this: </summary>

```
|  
+---detector/  
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

##  Usage
The overall pipeline is divided into three major steps. 

- Creation of a graph-dataset from the raw RadarScenes or nuScenes dataset
- Creation and training of a model based on the created graph-dataset
- Evaluation of the trained model

The settings of all three steps are defined in a unified configuration file, which must consequently be created first.
### 1. Create a configuration file 
The configuration file contains three sections with relevant settings for the corresponding steps (dataset creation, training, evaluation). It can be created based on the provided [configuration description](/configurations/configuration_description.yml) and [configuration template](/configurations/configuration_template.yml).
<br />

### 2. Create a graph-dataset
Next, the graph-dataset needs to be created by converting the radar point clouds of the raw datasets to a graph data structure. To do this, execute the following command inside the docker container: 
```
python3 src/gnnradarobjectdetection/create_dataset.py --dataset ${path_to_raw_dataset_folder}$ --config ${path_to_config_file}$
```

```
usage:          create_dataset.py [--dataset] [--config]

arguments:
    --dataset   Path to the raw (RadarScenes/nuScenes) dataset
    --config    Path to the created configuration.yml file
```

The created graph-dataset is saved in the automatically created folder "{path_to_dataset}/processed". After creating the graph-dataset, this folder may be renamed.
<br />

### 3. Create and train a model
In a next step, you can use the created graph-dataset to train a model. To do this, run the following command inside the docker container: 
```
python3 src/gnnradarobjectdetection/train.py --data ${path_to_graph_dataset_folder}$ --results ${path_to_results_folder}$ --config ${path_to_config_file}$
```
```
usage:             train.py [--data] [--results] [--config]

arguments:
    --data         Path to the created graph-dataset
    --results      Path to the created "results" folder
    --config       Path to the created configuration.yml file
```

Within the provided "results" folder, a new "model" folder is automatically created, in which the trained model is saved.
<br />


