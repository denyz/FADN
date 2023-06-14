# FADN
This repository contains an implementation of a convolution neural network for 3D object detection in the fusion of lidar point clouds and camera images As shown in the figure below, the model architecture consists of three major components: Frustum-Aware Generator, decorator, and fussion-based detection network.
![image](https://github.com/denyz/FADN/assets/18696187/1f4fb9e6-e055-4276-a3d4-da1eab007513)

It is built upon pytorch-geometric and provides usage with the KITTI and nuScenes dataset.


# Result

Hi! I'm your first Markdown file in **StackEdit**. If you want to learn about StackEdit, you can read me. If you want to play with Markdown, you can edit me. Once you have finished with me, you can create new files by opening the **file explorer** on the left corner of the navigation bar.
### 3D Object Detection and Semantic Segmentation (on KITTI)
| Model    | Invariance             | mAP      | F1       | Checkpoint                                                              |
|----------|------------------------|----------|----------|-------------------------------------------------------------------------|
| RadarGNN | None                   | 19.4     | 68.1     | [Link](https://zenodo.org/record/7822037/files/model_01.zip?download=1) |
| RadarGNN | Translation            | **56.5** | **77.1** | [Link](https://zenodo.org/record/7822037/files/model_02.zip?download=1) |
| RadarGNN | Translation & Rotation | 19.6     | 76.5     | [Link](https://zenodo.org/record/7822037/files/model_03.zip?download=1) |


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



StackEdit stores your files in your browser, which means all your files are automatically saved locally and are accessible **offline!**

## Create files and folders

The file explorer is accessible using the button in left corner of the navigation bar. You can create a new file by clicking the **New file** button in the file explorer. You can also create folders by clicking the **New folder** button.

## Switch to another file

All your files and folders are presented as a tree in the file explorer. You can switch from one to another by clicking a file in the tree.

## Rename a file

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

## Delete a file

You can delete the current file by clicking the **Remove** button in the file explorer. The file will be moved into the **Trash** folder and automatically deleted after 7 days of inactivity.

## Export a file

You can export the current file by clicking **Export to disk** in the menu. You can choose to export the file as plain Markdown, as HTML using a Handlebars template or as a PDF.


# Synchronization

Synchronization is one of the biggest features of StackEdit. It enables you to synchronize any file in your workspace with other files stored in your **Google Drive**, your **Dropbox** and your **GitHub** accounts. This allows you to keep writing on other devices, collaborate with people you share the file with, integrate easily into your workflow... The synchronization mechanism takes place every minute in the background, downloading, merging, and uploading file modifications.

There are two types of synchronization and they can complement each other:

- The workspace synchronization will sync all your files, folders and settings automatically. This will allow you to fetch your workspace on any other device.
	> To start syncing your workspace, just sign in with Google in the menu.

- The file synchronization will keep one file of the workspace synced with one or multiple files in **Google Drive**, **Dropbox** or **GitHub**.
	> Before starting to sync files, you must link an account in the **Synchronize** sub-menu.

## Open a file

You can open a file from **Google Drive**, **Dropbox** or **GitHub** by opening the **Synchronize** sub-menu and clicking **Open from**. Once opened in the workspace, any modification in the file will be automatically synced.

## Save a file

You can save any file of the workspace to **Google Drive**, **Dropbox** or **GitHub** by opening the **Synchronize** sub-menu and clicking **Save on**. Even if a file in the workspace is already synced, you can save it to another location. StackEdit can sync one file with multiple locations and accounts.

## Synchronize a file

Once your file is linked to a synchronized location, StackEdit will periodically synchronize it by downloading/uploading any modification. A merge will be performed if necessary and conflicts will be resolved.

If you just have modified your file and you want to force syncing, click the **Synchronize now** button in the navigation bar.

> **Note:** The **Synchronize now** button is disabled if you have no file to synchronize.

## Manage file synchronization

Since one file can be synced with multiple locations, you can list and manage synchronized locations by clicking **File synchronization** in the **Synchronize** sub-menu. This allows you to list and remove synchronized locations that are linked to your file.


# Publication

Publishing in StackEdit makes it simple for you to publish online your files. Once you're happy with a file, you can publish it to different hosting platforms like **Blogger**, **Dropbox**, **Gist**, **GitHub**, **Google Drive**, **WordPress** and **Zendesk**. With [Handlebars templates](http://handlebarsjs.com/), you have full control over what you export.

> Before starting to publish, you must link an account in the **Publish** sub-menu.

## Publish a File

You can publish your file by opening the **Publish** sub-menu and by clicking **Publish to**. For some locations, you can choose between the following formats:

- Markdown: publish the Markdown text on a website that can interpret it (**GitHub** for instance),
- HTML: publish the file converted to HTML via a Handlebars template (on a blog for example).

## Update a publication

After publishing, StackEdit keeps your file linked to that publication which makes it easy for you to re-publish it. Once you have modified your file and you want to update your publication, click on the **Publish now** button in the navigation bar.

> **Note:** The **Publish now** button is disabled if your file has not been published yet.

## Manage file publication

Since one file can be published to multiple locations, you can list and manage publish locations by clicking **File publication** in the **Publish** sub-menu. This allows you to list and remove publication locations that are linked to your file.


# Markdown extensions

StackEdit extends the standard Markdown syntax by adding extra **Markdown extensions**, providing you with some nice features.

> **ProTip:** You can disable any **Markdown extension** in the **File properties** dialog.


## SmartyPants

SmartyPants converts ASCII punctuation characters into "smart" typographic punctuation HTML entities. For example:

|                |ASCII                          |HTML                         |
|----------------|-------------------------------|-----------------------------|
|Single backticks|`'Isn't this fun?'`            |'Isn't this fun?'            |
|Quotes          |`"Isn't this fun?"`            |"Isn't this fun?"            |
|Dashes          |`-- is en-dash, --- is em-dash`|-- is en-dash, --- is em-dash|


## KaTeX

You can render LaTeX mathematical expressions using [KaTeX](https://khan.github.io/KaTeX/):

The *Gamma function* satisfying $\Gamma(n) = (n-1)!\quad\forall n\in\mathbb N$ is via the Euler integral

$$
\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.
$$

> You can find more information about **LaTeX** mathematical expressions [here](http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference).


## UML diagrams

You can render UML diagrams using [Mermaid](https://mermaidjs.github.io/). For example, this will produce a sequence diagram:

```mermaid
sequenceDiagram
Alice ->> Bob: Hello Bob, how are you?
Bob-->>John: How about you John?
Bob--x Alice: I am good thanks!
Bob-x John: I am good thanks!
Note right of John: Bob thinks a long<br/>long time, so long<br/>that the text does<br/>not fit on a row.

Bob-->Alice: Checking with John...
Alice->John: Yes... John, how are you?
```

And this will produce a flow chart:

```mermaid
graph LR
A[Square Rect] -- Link text --> B((Circle))
A --> C(Round Rect)
B --> D{Rhombus}
C --> D
```
