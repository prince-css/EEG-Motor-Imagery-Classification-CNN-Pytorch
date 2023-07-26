# EEG-Motor-Imagery-Classification-CNN-Pytorch

Date: July, 2023


**Based on the Paper:**

[A Novel Approach of Decoding EEG Four-class Motor Imagery Tasks via Scout ESI and CNN](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta)

The method in this repository is EEG source imaging (ESI) + Fourier Transform for joint time-frequency analysis + Convolutional Neural Networks (CNNs). The raw data has been processed using the Matlab Toolkit Brainstorm. The technique of ESI uses a boundary element method (BEM) and weighted minimum norm estimation (WMNE) to solve the EEG forward and inverse problems, respectively. Ten scouts are then created within the motor cortex to select the region of interest (ROI). Features were extracted from the time series of scouts using a Morlet wavelet approach. Lastly, CNN is employed for classifying MI tasks.

<img src="https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/blob/main/64-channel-sharbrough-1.jpg" alt="64_channels" width="400" height="400">


<h3>Filtering Data:</h3>


<img src="https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/blob/main/viz/filter0.png" alt="filter0" width="400" height="400">

<h3>time-frequency analysis:</h3>


<img src="https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/blob/main/viz/map0.png" alt="filter0" width="400" height="400">


<h3>Model Summary:</h3>

```
        Layer (type)               Output Shape         Param #

            Conv2d-1           [-1, 32, 32, 20]             320
           Dropout-2           [-1, 32, 32, 20]               0
            Conv2d-3           [-1, 32, 32, 20]           9,248
       BatchNorm2d-4           [-1, 32, 32, 20]              64
            Conv2d-5           [-1, 64, 32, 20]          36,928
           Dropout-6           [-1, 64, 32, 20]               0
            Conv2d-7            [-1, 64, 14, 8]          36,928
       BatchNorm2d-8            [-1, 64, 14, 8]             128
           Dropout-9            [-1, 64, 14, 8]               0
           Conv2d-10            [-1, 64, 14, 8]          36,928
      BatchNorm2d-11            [-1, 64, 14, 8]             128
           Conv2d-12           [-1, 128, 14, 8]         147,584
          Dropout-13           [-1, 128, 14, 8]               0
          Flatten-14                 [-1, 3584]               0
           Linear-15                  [-1, 512]       1,835,520
      BatchNorm2d-16            [-1, 512, 1, 1]           1,024
          Dropout-17                  [-1, 512]               0
           Linear-18                    [-1, 4]           2,052
	   
Total params: 2,106,852
Trainable params: 2,106,852
Non-trainable params: 0
```

<h3>Results:</h3>

<img src="https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/blob/main/viz/global_acc.png" alt="global_acc" width="400" height="400">

<img src="https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/blob/main/viz/acc_all_tasks.png" alt="all_tasks_acc" width="400" height="400">


<h3>Confusion matrix of R5 scout:</h3>

<img src="https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/blob/main/viz/confusion_matrix_R5.png" alt="cm_r5" width="400" height="400">

