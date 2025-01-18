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

![unet-architectures](https://github.com/user-attachments/assets/aec13297-7dfb-4430-b24b-465e27d88478)

**Results**

**Qualitative Results:**

The segmentation results demonstrate the model's ability to identify and segment objects such as guns and razor blades with significant accuracy.
Comparison of predicted segmentation maps and ground truth highlights the model's performance on fine object details, though some edge discrepancies are observed.

![Picture2](https://github.com/user-attachments/assets/92d612b8-5e74-4b3a-b9a1-47e5870bcba4)
![Picture1](https://github.com/user-attachments/assets/19bb5327-f2ae-41e7-8a86-778a185697cf)


**Quantitative Results:**

The confusion matrix indicates an overall accuracy of 78.08% for the segmentation task.
The model achieves high precision in segmenting the background class but struggles slightly with smaller or less distinct object details.
Pixel-level statistics:
True Background: 983,975 pixels
False Background: 281,460 pixels
True Objects: 23,478 pixels
False Objects: 1,327 pixels
![Picture3](https://github.com/user-attachments/assets/0ad1c432-945b-4084-8a22-a410e28b7c95)

**Analysis:**

While the model shows strong performance in identifying large and clear objects, there is room for improvement in detecting smaller or overlapping objects.
Future enhancements could involve fine-tuning hyperparameters or experimenting with advanced loss functions to balance precision and recall.


