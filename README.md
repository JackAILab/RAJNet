# A Novel End-to-end 3D-Residual and Attention Enhancement Joint Few-Shot Learning Model for Huntington Diagnosis 

<hr />

> Diagnosis of Huntington's disease (HD) is of great significance for the treatment of HD, and many methods have been 
proposed to solve this problem. However, most of the existing diagnostic models are expensive, and few studies consider end-to-end video-based diagnosis. In order to fill this research gap, we developed a few-shot learning model based on a 3D residual and attention enhancement joint learning model. The proposed model is expected to serve as a suitable baseline in future HD video-based diagnostic studies. 
***

## Installation

See [INSTALL.md](https://github.com/JackAILab/RAJNet/blob/main/INSTALL.md) for the installation of dependencies required to run RAJNet.


### Data preparation

We obtained the video data of HC and HD groups from the hospital. First, the videos are extracted at a rate of one frame per second, and the extracted images are converted into grayscale images. Second, we take all the image data obtained every 20 frames as individual data.

You can make your own h5f file according to [h5fMake.py](https://github.com/JackAILab/RAJNet/blob/main/h5fMake.py). At the same time, in order to facilitate operation, we have produced h5f files for you to run through. In order to protect privacy, the [data](https://github.com/JackAILab/RAJNet/blob/main/Data) we provide only take a small number of samples.


> tips: In the code, in order to write the h5f file, I encoded the name of each patient, and the encoding rules were designed by myself. Therefore, when running this project, it is necessary to design the file path, patient name, and encoding rules in order for the project to run normally.


## How to run

1. Prepare H5f Data

    At runtime, you can choose to use your own h5f data: please replace the path with your data in [dataset_h5f.py](https://github.com/JackAILab/RAJNet/blob/main/dataset_h5f.py) path. If you want to use the data we provide, please also make sure the path is correct.


2. Execute train.py

    ```
    python train.py
    ```
    You can also remove the training part and test directly.


## Contact

Should you have any question, please contact jiehuihuang1107@163.com

