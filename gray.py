
# coding: utf-8

# In[2]:


import os
from PIL import Image

def Image2GRAY(path):
    imgs = os.listdir(path)
    i = 0
    for img in imgs:
        if i%10==0:
            im = Image.open(path+'/'+img).convert('L')
            im = im.resize((180,80),Image.ANTIALIAS)
            im.save('data/test'+img)
        else:
            im = Image.open(path+'/'+img).convert('L')
            im = im.resize((180,80),Image.ANTIALIAS)
            im.save('data/train'+img) 
        i+=1

if __name__=='__main__':
    path = 'data/tem'
    Image2GRAY(path)

