# Horse_Human_Flask_restful
This repository contains scripts for training a horse-human classifier and deploying it using Flask_Restfull


Datset can be downloaded from->   kaggle datasets download -d sanikamal/horses-or-humans-dataset

Training .py:
	Training:
	Model has been trained on Tensorflow Framework:

	Architecture:
	1. Input Layer 
	2. Conv2D Layer
	3. Batch Normalisation Layer
	4. MaxPooling2D Layer
	5. Dropout Layer
	6. Conv2D Layer
	7. Batch Normalisation Layer
	8. MaxPooling2D Layer
	9. Dropout Layer
	10.Conv2D Layer
	11. Batch Normalisation Layer
	12. MaxPooling2D Layer
	13. Flatter Layer
	14. Dropout Layer
	15. Dense Layer
	16. Batch Norm Layer
	17. Dense Layer
	18. Output Layer

	Output Model format: '.hdf5'

Prediction.py:

	Evaluation: Evaluates Test images
	Deployement framework : FLask_restful
	
	Pre_pocess class:
	Contains methods for pre-processing input image
	
	predict class:
	Contains methos for prediction the ouput

