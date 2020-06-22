{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install flask\n",
    "!pip install flask_restful\n",
    "\n",
    "from flask import Flask,request,jsonify\n",
    "from flask_restful import Resource,Api\n",
    "import tensorflow as tf\n",
    "import cv2\n",
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "import io\n",
    "import base64\n",
    "import numpy as np\n",
    "\n",
    "app=Flask(__name__)\n",
    "api=Api(app)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Pre_Process():\n",
    "    \n",
    "    \"\"\" Pre-Processing Class\"\"\"\n",
    "    \n",
    "    def pipeline(self,image):\n",
    "    \n",
    "        img=cv2.imread(image)\n",
    "        img=self.check_size(img)\n",
    "        img=self.check_channels(img)\n",
    "        img=self.convert_array(img)\n",
    "        img=self.expand_dims(img)\n",
    "        img=self.caster(img)\n",
    "        #print(len(glob.glob(os.path.join(datadir,\"*\",\"*\" ,\"*.png\"))))\n",
    "        \n",
    "        return img\n",
    "\n",
    "     \n",
    "    def check_channels(self,img):\n",
    "        \n",
    "        if img.shape[-1]==3:\n",
    "            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)\n",
    "            return img\n",
    "        elif img.shape[-1]==1:\n",
    "            img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)\n",
    "            return img\n",
    "        else:\n",
    "            raise ValueError(\"{} channel is not supported\".format(img.shape[-1]))\n",
    "            \n",
    "            \n",
    "    def check_size(self,img):\n",
    "        \n",
    "        if img.shape[0:2]==(300,300):\n",
    "            return img\n",
    "\n",
    "        else:\n",
    "            img=cv2.resize(img,(300,300),interpolation=cv2.INTER_NEAREST)\n",
    "            return img\n",
    "    \n",
    "    def check_format(self,img):\n",
    "        \n",
    "        chck=(os.path.splitext(os.path.basename(img))[1])\n",
    "        if chck == '.png'or chck=='.jpg':\n",
    "            pass\n",
    "        else:\n",
    "            raise ValueError(\"{} is not supported\".format(chck))\n",
    "    def convert_array(self,img):\n",
    "        img=np.asarray(img)\n",
    "        return img\n",
    "\n",
    "    def expand_dims(sef,img):\n",
    "        if img.ndim==3:\n",
    "            img=np.expand_dims(img,axis=0)\n",
    "            return img\n",
    "        else:\n",
    "            raise ValueError(\"Input dim {} does not match expected dim {}\".format(img.ndim,3))\n",
    "\n",
    "    def caster(self,img):\n",
    "        img=tf.cast(img,tf.int64)\n",
    "        return img\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Predictions():\n",
    "    \n",
    "    def __init__(self):\n",
    "        \n",
    "        \"\"\" Prediction Class\"\"\"\n",
    "        \n",
    "        self.model=tf.keras.models.load_model('final_model_train1.hdf5')\n",
    "        \n",
    "        \n",
    "    def pipeline(self, data):\n",
    "        \n",
    "        self.pre_cls=Pre_Process()\n",
    "        out_im=self.pre_cls.pipeline(data)\n",
    "        return out_im\n",
    "    \n",
    "    def pred(self,image):\n",
    "        \n",
    "        img=self.pipeline(image)\n",
    "        predict_score=self.model.predict(img,steps=1)\n",
    "        return (predict_score.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Predictions_Api(Resource):\n",
    "    def __init__(self):\n",
    "        \n",
    "        self.predict_initiate=Predictions()\n",
    "        \n",
    "    def get(self,img_path):\n",
    "        \n",
    "        try:\n",
    "            prediction=self.predict_initiate.pred(img_path)\n",
    "            predict_class='Horse' if float(''.join(map(str,prediction[0])))>0 else 'Human'\n",
    "        \n",
    "            return jsonify({\"Output\":predict_class}) \n",
    "        \n",
    "        except Exception:\n",
    "            return jsonify(status_code='400', msg='Bad Request'), 400\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "api.add_resource(Predictions_Api,'/pred/<string:img_path>')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__==\"__main__\":\n",
    "    app.run(host='0.0.0.0',port='8080')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
