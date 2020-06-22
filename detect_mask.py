# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:16:26 2020

@author: 33668
"""
import torch.nn as nn
from torch import load, device, cuda
from model import Flatten
from torchvision.transforms import ToTensor,Compose, Resize, Normalize, ToPILImage
import cv2
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils import detect_face
from PIL import Image, ImageDraw
import numpy as np


index_to_class = {0: 'with_mask', 1:'without_mask'}
vid_capture = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

dvc = device('cpu')
# if cuda.is_available():
#     dvc = device('cuda')
mtcnn = MTCNN(keep_all=True, device=dvc)   
# channel_1 = 16
# channel_2 = 8
# channel_3 = 4
# model = nn.Sequential(
#         nn.Conv2d(3, channel_1,(7, 7),padding = 3),
#         nn.ReLU(),
#         nn.Dropout2d(p=0.25),
#         nn.Conv2d(channel_1, channel_2,(5, 5),padding = 2),
#         nn.ReLU(),
#         nn.MaxPool2d((2,2)), 
#         nn.Conv2d(channel_2, channel_3,(3, 3),padding = 1),
#         nn.ReLU(),
#         nn.Dropout2d(p=0.25),
#         Flatten(),
#         nn.Linear(channel_3 * 64 * 64//4, 2)
#     )   
channel_0 = 32
channel_1 = 16
channel_2 = 8
channel_3 = 4
model = nn.Sequential(
        nn.Conv2d(3, channel_0,(7, 7),padding = 3),
        nn.ReLU(),
        nn.Dropout2d(p=0.25),
        nn.Conv2d(channel_0, channel_1,(5, 5),padding = 2),
        nn.ReLU(),
        nn.MaxPool2d((2,2)), 
        nn.Conv2d(channel_1, channel_2,(3, 3),padding = 1),
        nn.ReLU(),
        nn.Dropout2d(p=0.25),
        nn.Conv2d(channel_2, channel_3,(3, 3),padding = 1),
        nn.ReLU(),
        nn.MaxPool2d((2,2)), 
        Flatten(),
        nn.Linear(channel_3 * 64 * 64//16, 2)
    )  
model.load_state_dict(load("model_bis0.92.pth", map_location = dvc), strict=False)

def predict_image(image):
    to_pil = ToPILImage()
    image = to_pil(image).convert('RGB')
    test_transforms  = Compose(
            [Resize([64, 64]),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])])
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    # input = Variable(image_tensor)
    input = image_tensor.to(dvc)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index
ctr = 50
margin = 40
frame_width = int(vid_capture.get(3))
frame_height = int(vid_capture.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
while(True):
      # Capture each frame of webcam video
      ret,frame = vid_capture.read()
      PIL_frame = Image.fromarray(np.uint8(frame)).convert('RGB')
      PIL_frame = Image.fromarray(frame.astype('uint8'), 'RGB')
      
      # Detect faces
      boxes, _ = mtcnn.detect(PIL_frame)
    
      # Draw faces
      frame_draw = PIL_frame.copy()
      draw = ImageDraw.Draw(frame_draw)
      if not(boxes is None):
          for box in boxes:
                box[0] -= margin
                box[1] -= margin
                box[2] += margin
                box[3] += margin
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)
                # extract face
                face = detect_face.extract_face(frame_draw, box, 32, 0)
                draw.text((box[0],box[1]), index_to_class[predict_image(face)])
      
      cv2.imshow("My cam video", np.array(frame_draw.convert('RGB')))
      out.write(np.array(frame_draw.convert('RGB')))
      # frame_draw
      # Close and break the loop after pressing "x" key
     
      if cv2.waitKey(1) &0XFF == ord('x'):
           cv2.destroyAllWindows() 
           vid_capture.release()
           out.release()
           break
     