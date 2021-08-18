import streamlit as st
from skimage import color
import time
import numpy as np
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image as pil_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing.image import img_to_array

st.title("Image Denoising and Colouring")
with st.form(key='form1'):
  img=st.file_uploader('Please upload a RGB/Greyscale image', type=['jpg','png'])
  method=st.selectbox('please select method',('Denoising','Colouring','Denosiing and colouring'))
  submit = st.form_submit_button(label ='submit this form')

  
def Classifer(img):
  #Classifer
  classifier = tf.keras.models.load_model(r"C:\streamlit\classifier_90_.model-20210818T171404Z-001\classifier_90_.model",custom_objects=None,
                                   compile=True)
  
  class_names = {0: 'Gaussian',1:'Salt and Pepper',2:'Speckle'} 
  size=(256,256)
  #image = Image.open(img)
  image = ImageOps.fit(img, size, Image.ANTIALIAS)
  image = np.asarray(image)
  image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
  resized = cv2.resize(image, (256, 256))
  try:
    image =  np.expand_dims(resized, axis=0)/255
  except:
    noise_type = 'Model only supports s&p,speckile,gaussian noise'
  predictions = classifier.predict(image)
  score = tf.nn.softmax(predictions[0])
  noise_type = class_names[np.argmax(score)]
  return noise_type

def denoising(img,noise_type):
  #denoising
  gaussian = tf.keras.models.load_model(r"C:\Users\chana\Downloads\denoising_models-20210817T175729Z-001\denoising_models\gaussian\gaussian_autoencoder.model",
                                   custom_objects=None,
                                   compile=True)
  salt_pepper = tf.keras.models.load_model("/content/drive/MyDrive/denoising_models/s&p/s&p.model",
                                   custom_objects=None,
                                   compile=True)
  speckle = tf.keras.models.load_model("/content/drive/MyDrive/denoising_models/speckle/speckle_autoencoder.model",
                                   custom_objects=None,
                                   compile=True)
  size=(256,256)
 #image = Image.open(img)
  image = ImageOps.fit(img, size, Image.ANTIALIAS)
  image = np.asarray(image)
  image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
  resized = cv2.resize(image, (256, 256))
  image =  np.expand_dims(resized, axis=0)/255
  print(image.shape)
  if noise_type == 'Gaussian':
    denoised = gaussian.predict(image)
    denoised_img = denoised.reshape(256,256,3)
  elif noise_type == 'Salt and Pepper':
    denoised = salt_pepper.predict(image)
    denoised_img = denoised.reshape(256,256,3)
  elif noise_type == 'Speckle':
    denoised = speckle.predict(image)
    denoised_img = denoised.reshape(256,256,3)
  output = denoised
  out = Image.fromarray((output[0] * 255).astype(np.uint8))
  return output,out

def colouring(img):
  colour = tf.keras.models.load_model("/content/drive/MyDrive/coloured_final/coloured.model",
                                   custom_objects=None,
                                   compile=True)
  size=(256,256)
  image = ImageOps.grayscale(img)
  image = np.array(image) 
  resized = cv2.resize(image, (256, 256))
  image =  np.expand_dims(resized, axis=0)/255
  coloured = colour.predict(image)
  coloured = coloured.reshape(256,256,3)
  return coloured

if submit:
  image = Image.open(img)
  st.image(image, width=250)
  if method == 'Denoising':
    noise_type = Classifer(image)
    st.write(noise_type)
    if noise_type != 'Gaussian' and noise_type != 'Speckle' and noise_type != 'Salt and Pepper':
      st.write('Model only supports salt and pepper,speckile,gaussian noise')
    else:
      denoised,out = denoising(image,noise_type)
      st.image(denoised, width=250)

  if method == 'Colouring':
    coloured = colouring(image)
    st.image(coloured, width=250)

  if method == 'Denosiing and colouring':
    noise_type = Classifer(image)
    if noise_type != 'Gaussian' and noise_type != 'Speckle' and noise_type != 'Salt and Pepper':
      st.write(noise_type)
    else:
      denoised,out = denoising(image,noise_type)
    st.write(noise_type)
    st.image(denoised, width=250)
    coloured = colouring(out)
    st.image(coloured, width=250)




