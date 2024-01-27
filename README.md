# DGCN and UNET application on X-ray threat object detection in bags
training and evaluating dual graph CNN and Unet architecture based mode on single class dataset using image segmentaiton. This is a Digital Image processing postgrad semester project and achieved 78% accuracy using a small dataset uning 10 epoches, 256x256 resolution, batch size 4 and trainned on 80-20 rule.

In first i extracted the architecure provided in this git and loaded in my model (in training code):
https://github.com/lxtGH/GALD-DGCNet

Dataset link:
https://drive.google.com/drive/folders/1nTKKt3XpAbqxCPXukSMsnH3KRvbRPIXE?usp=drive_link

UNET model file link:
https://drive.google.com/file/d/1ncLLqFaXScpad40vLxlusgKGAIINClKt/view?usp=sharing

DGCN architecure has 70 million plus parameters and 360+ layers

then I used Unet standard architecture for training in same dataset and generated confusion matrix, predicion accuracy on the 20% evaluation file test my training code, image preprocessing, model settings and efficiency of selected loss functions and optimizer.

DGCN architecture is extensive model training, requires high-end system to train even small datasets like i used a 200mb dataset yet it still required better cpu and memory. you can use the training code i wrote for the DGCN model and run the same evaluation on it.
LOSS FUNCT: Binary Cross Entropy loss for binary segmentation
OPTIMIZER: ADAM
