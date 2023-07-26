# EEG-Motor-Imagery-Classification-CNN-Pytorch

Date: July, 2023


Based on the Paper:

A Novel Approach of Decoding EEG Four-class Motor Imagery Tasks via Scout ESI and CNN

The method in this repository is EEG source imaging (ESI) + Fourier Transform for joint time-frequency analysis + Convolutional Neural Networks (CNNs). The raw data has been processed using the Matlab Toolkit Brainstorm. 


![64-channel-sharbrough-1](https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/assets/63596657/da58e004-d073-41fa-86c9-9305b267682e)

##Filtering Data:


![filter0](https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/assets/63596657/5bf1f281-217e-4e08-b283-bcf52e995ff3)

##time-frequency analysis:
![map0](https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/assets/63596657/43591d57-5d24-41bc-bb20-f9d24dd7df9e)



## Model Summary:

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
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
================================================================
Total params: 2,106,852
Trainable params: 2,106,852
Non-trainable params: 0

##Results:

![global_acc](https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/assets/63596657/8218bf3e-f0a4-4e20-b057-94a834b78d47)

![acc_all_tasks](https://github.com/prince-css/EEG-Motor-Imagery-Classification-CNN-Pytorch/assets/6359665![confusion_matrix_R5](https://github.com/prince-css/EEG-

##Confusion matrix of R5 scout:


Motor-Imagery-Classification-CNN-Pytorch/assets/63596657/bd10ec8d-fe26-4fa6-902a-7a1279c3091c)
7/eccfd83e-e467-416a-98be-e37751c9ff3c)

