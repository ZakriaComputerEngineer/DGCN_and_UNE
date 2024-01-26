# DGCN_and_UNE
training and evaluating dual graph CNN and Unet architecture based mode on single class dataset using image segmentaiton. This is a Digital Image processing postgrad semester project and achieved 78% accuracy using a small dataset uning 10 epoches, 256x256 resolution, batch size 4 and trainned on 80-20 rule.

In first i extracted the architecure provided in this git and loaded in my model (in training code):
https://github.com/lxtGH/GALD-DGCNet

the architecure has 70 million plus parameters and 360+ layers

then I used Unet standard architecture for training in same dataset and generated confusion matrix, predicion accuracy on the 20% evaluation file.
