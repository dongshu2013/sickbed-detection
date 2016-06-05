###Steps to run the project:

####Preparation:
1. Check data folder to make sure the rgb-image-train/negative and rgb-image-train/positive are not empty. These two folder are used to store training images.
2. Check folder model exists otherwise run '__mkdir model__' to create one. This folder is to store trained models.
3. Check folder results exists otherwise run '__mkdir result__' to create one. This folder is to store classification results. Under result, create three subfolders: predict, rgb-hard-negative and rgb-low-accuracy.

####Run the program:
1. Run 'python train.py' to train the model. The model will be cached in the model folder
2. Run 'python predict.py' to do the detection. The result contains five things.
	1. __result/predict__. These are predicted images with both correct bounding box(white) and the bounding box generated(black). 
	2. __result/accuracy.csv__. This file contains the accuracy details for each test image.
	3. __result/accuracy.png__. This file contains the histogram of accuracy results.
	4. __result/rgb-low-accuracy__. These are images with accuracy lower than 50%.
	5. __result/rgb-hard-negative__. These are false positive windows generated during detection.

####Apply Hard Negative Learning
The program will automatically generate false positive windows and store them in result/rgb-hard-negative. If you want to apply hard negative learning, put all these images into data/rgb-image-train/negative and redo the training and predicting process.