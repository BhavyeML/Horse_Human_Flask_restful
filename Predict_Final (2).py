#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install flask')
get_ipython().system('pip install flask_restful')

from flask import Flask,request,jsonify
from flask_restful import Resource,Api
import tensorflow as tf
import cv2
import cv2
import matplotlib.pyplot as plt
import os
import io
import base64
import numpy as np

app=Flask(__name__)
api=Api(app)


# In[ ]:


class Pre_Process():
    
    """ Pre-Processing Class"""
    
    def pipeline(self,image):
    
        img=cv2.imread(image)
        img=self.check_size(img)
        img=self.check_channels(img)
        img=self.convert_array(img)
        img=self.expand_dims(img)
        img=self.caster(img)
        #print(len(glob.glob(os.path.join(datadir,"*","*" ,"*.png"))))
        
        return img

     
    def check_channels(self,img):
        
        if img.shape[-1]==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img
        elif img.shape[-1]==1:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            return img
        else:
            raise ValueError("{} channel is not supported".format(img.shape[-1]))
            
            
    def check_size(self,img):
        
        if img.shape[0:2]==(300,300):
            return img

        else:
            img=cv2.resize(img,(300,300),interpolation=cv2.INTER_NEAREST)
            return img
    
    def check_format(self,img):
        
        chck=(os.path.splitext(os.path.basename(img))[1])
        if chck == '.png'or chck=='.jpg':
            pass
        else:
            raise ValueError("{} is not supported".format(chck))
    def convert_array(self,img):
        img=np.asarray(img)
        return img

    def expand_dims(sef,img):
        if img.ndim==3:
            img=np.expand_dims(img,axis=0)
            return img
        else:
            raise ValueError("Input dim {} does not match expected dim {}".format(img.ndim,3))

    def caster(self,img):
        img=tf.cast(img,tf.int64)
        return img
        


# In[ ]:


class Predictions():
    
    def __init__(self):
        
        """ Prediction Class"""
        
        self.model=tf.keras.models.load_model('final_model_train1.hdf5')
        
        
    def pipeline(self, data):
        
        self.pre_cls=Pre_Process()
        out_im=self.pre_cls.pipeline(data)
        return out_im
    
    def pred(self,image):
        
        img=self.pipeline(image)
        predict_score=self.model.predict(img,steps=1)
        return (predict_score.tolist())


# In[ ]:


class Predictions_Api(Resource):
    def __init__(self):
        
        self.predict_initiate=Predictions()
        
    def get(self,img_path):
        
        try:
            prediction=self.predict_initiate.pred(img_path)
            predict_class='Horse' if float(''.join(map(str,prediction[0])))>0 else 'Human'
        
            return jsonify({"Output":predict_class}) 
        
        except Exception:
            return jsonify(status_code='400', msg='Bad Request'), 400
        


# In[ ]:


api.add_resource(Predictions_Api,'/pred/<string:img_path>')


# In[ ]:


if __name__=="__main__":
    app.run(host='0.0.0.0',port='8080')


# In[ ]:




