# Horse_Human_Flask_restful
This repository contains scripts for training a horse-human classifier and deploying it using Flask_Restfull


Datset and model weights can be downloaded from->   Utility Folder
Dataset:
	Dataset Name: Horse vs Human
	
	Training Size: 1027 images
	Validation Size: 256 images
	
	Directory structure: Horse-and-Human------------> train------------>Horse
					     \   		\
					      \ 		 \---------> Human
	                                       \
					        \---------> Vlidation---------> horse
								     \
								       \---------> Human

Training.py:
	Training:
	Model has been trained on Tensorflow Framework:

	Architecture has 64 Layers of type:
	1. Input Layer 
	2. Conv2D Layer
	3. Batch Normalisation Layer
	4. MaxPooling2D Layer
	5. Dropout Layer
	6.. Flatten Layer
	7. Lambda Layer
	8. Dense Layer
	9. Output Layer
	
	Loss Function: Binary Cross Entropy
	
	Optimizer: RMS
	
	Epochs: 200

	Output Model format: '.hdf5'
	
	Classes:
		1. Augment class: Used to do in-memory augmentation
		2. Net class : Model class
		3. Train class: Performs training and data visualization
		4. Callback Classs: To check for desired metric

Prediction.py:

	Evaluation: Evaluates Test images
	Deployement framework : FLask_restful
	
	Pre_Pocess class:
	Contains methods for pre-processing input image
	
	Predict class:
	Contains methos for prediction the ouput
	
	Predictions_Api:
	Wrapper class for production integrated with Flask_restful
	
	Note: Production can be further enhance by rendering a html page in get() function and a predict button calling post() 
	Function in Predictions_Api class
	
	We can also deploy the model using Tensorflow serving on edge

