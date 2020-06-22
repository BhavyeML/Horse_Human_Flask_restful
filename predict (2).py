#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install flask')
get_ipython().system('pip install flask_restful')

from flask import Flask,request
from flask_restful import Resource,Api
import tensorflow as tf
import cv2
import cv2
import matplotlib.pyplot as plt
import os
import io
import base64

app=Flask(__name__)
api=Api(app)


# In[ ]:


class pre_process():
    
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
    

class predictions(Resource, pre_process):
    
    def __init__(self):
        
        self.model=tf.keras.load_model('C:/Users\Mridu\Flask\final_model_train1.hdf5')
        
        
    def pipeline(self, data):
        
        out_im=pre_process.pipeline(self,data)
        return out_im
    
    def pred(self,image):
        
        img=self.pipeline(image)
        predict_score=self.model.predict(img)
        return (predict_score.tolist())
    
    def get(self,img_path):
        
        try:
            self.img_path =request.form['data']
        except Exception:
            return jsonify(status_code='400', msg='Bad Request'), 400
        
        img_path = base64.b64decode(data)

        img = io.BytesIO(img_path)
        precition=self.pred(img)
        predict_class='a' if float(''.join(map(str,prediction[0])))>0 else 'b'
        
        return jsonify({"output": predict_class})
        


# In[ ]:


api.add_resource(predictions,'/<string:text>')


# In[ ]:


if __name__=="__main__":
    app.run(host='0.0.0.0',port='8080')


# In[ ]:





# In[ ]:




