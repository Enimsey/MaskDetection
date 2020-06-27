# Mask Detection
Using CNNs with Pytorch to detect whether a person is wearing a mask<br />
Final model: nn.Sequential(
 - nn.Conv2d(3, channel_0,(7, 7),padding = 3),
 - nn.ReLU(),
 - nn.Dropout2d(p=0.25),
 - nn.Conv2d(channel_0, channel_1,(5, 5),padding = 2),
 - nn.ReLU(),
 - nn.MaxPool2d((2,2)), 
 - nn.Conv2d(channel_1, channel_2,(3, 3),padding = 1),
 - nn.ReLU(),
 - nn.Dropout2d(p=0.25),
 - nn.Conv2d(channel_2, channel_3,(3, 3),padding = 1),
 - nn.ReLU(),
 - nn.MaxPool2d((2,2)),
 - nn.Linear(channel_3 * 64 * 64//16, 2)
 <br />
With : channel_0 = 32, channel_1 = 16, channel_2 = 8, channel_3 = 4<br />
I reached up to 96% of accuracy on the test set.  It can be improved with more training epochs and a broader and more diverse dataset.<br />

# Dataset
I am using a subset of the Real World Masked Face Dataset: https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset

# Face Detection
For face detection/ cropping, i am using the excellent facenet-pytorch library (check dependencies).

# HOWTO
To test the mask detection, run the detect_mask.py file. You can change the device used to cuda or cpu. It will use the already trained model, in the repository. <br />
To train a new model with a different structure,  you should open the model.py file, change the architecture, and run, it will train and save the model for you. <br />
To apply it later, change the name of the model in the detect_mask.py file and copy the structure you used in model.py for a successful load of the model.<br />

# Requirements:
facenet_pytorch, torch, torchvision, cv2, PIL
