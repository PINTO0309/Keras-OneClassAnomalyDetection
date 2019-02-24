# Keras-OneClassAnomalyDetection
Learning Deep Features for One-Class Classification (AnomalyDetection).  
Corresponds RaspberryPi3.  
Convert to Tensorflow, ONNX, Caffe, PyTorch, Tensorflow Lite.  
  
**[Jan 19, 2019] First Release. It corresponds to RaspberryPi3.**  
**[Jan 20, 2019] I have started work to make it compatible with OpenVINO.**  
**[Feb 15, 2019] Support for OpenVINO. [x86_64 only. CPU/GPU(Intel HD Graphics)]**  
**[Feb 24, 2019] Support for Tensorflow Lite. [for RaspberryPi3]**

# Introduction
This repository was inspired by **[Image abnormality detection using deep learning ーPapers and implementationー - Qiita - shinmura0](https://qiita.com/shinmura0/items/cfb51f66b2d172f2403b)**, **[Image inspection machine for people trying hard - Qiita - shinmura0](https://qiita.com/shinmura0/items/7f4298b75d6b788bba80)** and was created.    
I would like to express my deepest gratitude for having pleasantly accepted his skill, consideration and article quotation.  
His articles that were supposed to be used practically, not limited to logic alone, are wonderful.  
However, I don't have the skills to read papers, nor do I have skills to read mathematical expressions.  
I only want to verify the effectiveness of his wonderful article content in a practical range.  
To be honest, I am not engaged in the work of making a program.  

# Environment (example)
1. Ubuntu 16.04 (GPU = Geforce GTX 1070)
2. CUDA 9.0 + cuDNN 7.2
3. LattePanda Alpha (GPU = Intel HD Graphics 615)
4. RaspberryPi3 (CPU = Coretex-A53)
5. Python 3.5
6. Tensorflow-gpu 1.12.0 (pip install) or Tensorflow 1.11.0 (self-build wheel) or Tensorflow Lite 1.11.0 (self-build wheel)
7. Keras 2.2.4
8. PyTorch 1.0.0
9. torchvision 0.2.1
10. Caffe
11. numpy 1.15.3
12. matplotlib 3.0.1
13. PIL 5.3.0
14. OpenCV 4.0.1-openvino
15. sklearn 0.20.0
16. OpenVINO R5 2018.5.445

# Translating shinmura0's article
## 0. Table of contents
1. **[Introduction](#1-introduction)**
2. **[Overview](#2-overview)**
3. **[Preparing data](#3-preparing-data)**
4. **[Preparing the model](#4-preparing-the-model)**
5. **[Learning phase](#5-learning-phase)**
6. **[Test phase](#6-test-phase)**
7. **[Implementation by Keras](#7-implementation-by-keras)**  
  7-1. **[Load data](#7-1-load-data)**  
  7-2. **[Data resizing](#7-2-data-resizing)**  
  7-3. **[Model building and learning](#7-3-model-building-and-learning)**  
8. **[Result](#8-result)**  
  8-1. **[Look at the distribution](#8-1-look-at-the-distribution)**  
  8-2. **[Abnormality detection performance](#8-2-abnormality-detection-performance)**  
  8-3. **[Relationship between images and abnormal scores](#8-3-relationship-between-images-and-abnormal-scores)**  
9. **[Visualization by Keras](#9-visualization-by-keras)**  
  9-1. **[Grad-CAM](#9-1-grad-cam)**  
  9-2. **[Results](#9-2-results)**  
10. **[Implementation by RaspberryPi](#10-implementation-by-raspberrypi)**  
  10-1. **[Environment](#10-1-environment)**  
  10-2. **[How to use](#10-2-how-to-use)**  
  10-3. **[Learning by original data set](#10-3learning-by-original-data-set)**  
  10-4. **[Acquisition of learning image data](#10-4acquisition-of-learning-image-data)**  
  10-5. **[Resizing the image for learning](#10-5resizing-the-image-for-learning)**  
  10-6. **[Data Augmentation](#10-6data-augmentation)**  
  10-7. **[Generation of reference data](#10-7generation-of-reference-data)**  
  10-8. **[Training](#10-8training)**  
  10-9. **[Execution of inference with RaspberryPi](#10-9execution-of-inference-with-raspberrypi)**  
  10-10. **[Results](#10-10results)**  
11. **[Acceleration of LOF](#11-acceleration-of-lof)**
12. **[Structure of the model](#12-structure-of-the-model)**
13. **[Model Convert](#13-model-convert)**  
  13-1. **[MMdnn](#13-1-mmdnn)**  
  13-2. **[Keras -> Tensorflow -> OpenVINO](#13-2-keras---tensorflow---openvino)**  
  13-3. **[Keras -> ONNX -> OpenVINO](#13-3-keras---onnx---openvino)**  
  13-4. **[Keras -> Caffe -> OpenVINO](#13-4-keras---caffe---openvino)**  
  13-5. **[Keras -> PyTorch](#13-5-keras---pytorch)**  
  13-6. **[Keras -> Tensorflow -> Tensorflow Lite](#13-6-keras---tensorflow---tensorflow-lite)**
14. **[Issue](#14-issue)**  

## 1. Introduction
There are many methods such as methods using **"[Implemented ALOCC for detecting anomalies by deep learning (GAN) - Qiia - kzkadc](https://qiita.com/kzkadc/items/334c3d85c2acab38f105)"** and methods using **"[Detection of Video Anomalies Using Convolutional Autoencoders and One-Class Support Vector Machines (AutoEncoder)](http://cbic2017.org/papers/cbic-paper-49.pdf)"** for image anomaly detection using deep learning.  
Here is an article on detecting abnormality of images using "Variational Autoencoder".  
**[Image abnormality detection using Variational Autoencoder (Variational Autoencoder) - Qiita - shinmura0](https://qiita.com/shinmura0/items/811d01384e20bfd1e035)**  

The method to be introduced this time is to detect abnormality by devising the loss function using normal convolution neural network(CNN).  
  
「Learning Deep Features for One-Class Classification」 (Subsequent abbreviations, DOC)  
**arxiv：　https://arxiv.org/abs/1801.05365**  
  
![01](media/01.png)  
  
In conclusion, it was found that **this method has good anomaly detection accuracy** and visualization of abnormal spots is also possible.  

## 2. Overview
This paper states that it achieved state-of-the-art at the time of publication.  
In the figure below, we learned under various conditions using normal CNN and visualized the output from the convolution layer with t-SNE.  
  
![02](media/02.png)
  
- Figure (b): Alexnet's learned model with Normal and Abnormal distributed
- Figure (c): Diagram learned with Normal vs Abnormal
- Figure (e): Proposed method (DOC)

I think that it is not only me that thinking that abnormality can be detected also in figure (b).  
However, it is somewhat inferior to figure (e).  
  
In the thesis, it finally uses "k neighborhood method" in (e) to detect abnormality.  
As a learning method, view the images that you want to detect abnormality at the same time, completely different kinds of images, and narrow down the range of the images for which you want to detect anomalies.  

## 3. Preparing data
For learning, prepare the following data.  

|Dataset name|Contents|Concrete example|Number of classes|
|:--|:--|:--|:-:|
|Target data|Image you want to detect abnormality|Product etc.|1|
|Reference data|A data set not related to the above|ImageNet and CIFAR-10|10 or 1,000 or more|

## 4. Preparing the model
![03](media/03.png)  

- The deep learning model **`g`** prepares a learned model.
- In the paper, **`g`** uses Alexnet and VGG16. **`h`** is 1,000 nodes for ImageNet, 10 nodes for CIFAR-10.
- During learning, **`g`** and h of Reference Network (R) and Secondary Network (S) are shared.
- Also, during learning, the weights are fixed except for the last four layers.

## 5. Learning phase

- First, using the reference data, let **`R`** compute the loss function <img src="https://latex.codecogs.com/gif.latex?l_D" />.
- Next, using the target data, let **`S`** calculate the loss function <img src="https://latex.codecogs.com/gif.latex?l_C" />.
- Finally, let 's learn **`R`** and **`S`** at the same time by <img src="https://latex.codecogs.com/gif.latex?l_D" /> and <img src="https://latex.codecogs.com/gif.latex?l_C" />.
  
Total Loss <img src="https://latex.codecogs.com/gif.latex?L" /> is defined by the following formula.  
  
<img src="https://latex.codecogs.com/gif.latex?{L=l_D&plus;\lambda&space;l_C&space;}" />  
  
<img src="https://latex.codecogs.com/gif.latex?l_D" /> uses the cross entropy used in normal classification problems.  
Also in the paper <img src="https://latex.codecogs.com/gif.latex?\lambda=0.1" />.  
  
The most important compact loss <img src="https://latex.codecogs.com/gif.latex?l_C" /> is calculated as follows.  
Let <img src="https://latex.codecogs.com/gif.latex?n" /> be the batch size and let <img src="https://latex.codecogs.com/gif.latex?x_i\in&space;R^k" /> be the output (k dimension) from **`g`**. Then define <img src="https://latex.codecogs.com/gif.latex?z_i"/> as follows.  
  
<img src="https://latex.codecogs.com/gif.latex?{z_i&space;=&space;x_i&space;-&space;m_i}" />
<img src="https://latex.codecogs.com/gif.latex?m_i&space;=&space;\frac{1}{n-1}\sum_{j\not=i}x_j" />
  
<img src="https://latex.codecogs.com/gif.latex?m_i" /> is the average value of the output except <img src="https://latex.codecogs.com/gif.latex?x_i" /> in the batch. At this time, <img src="https://latex.codecogs.com/gif.latex?l_C" /> is defined as follows.  
  
<img src="https://latex.codecogs.com/gif.latex?{l_{C}=\frac{1}{nk}\sum_{i=1}^nz_i^Tz_i&space;}" />

<hr />

**＜Annotation＞**  
As an image, (Strictly speaking it is different) <img src="https://latex.codecogs.com/gif.latex?l_C" /> can be regarded as the variance of the output within the batch.  
When assembling <img src="https://latex.codecogs.com/gif.latex?l_C" /> code, it is troublesome to write "average value other than <img src="https://latex.codecogs.com/gif.latex?x_i" />", I used the following formula in the appendix of the paper.  
  
<img src="https://latex.codecogs.com/gif.latex?l_{C}=\frac{1}{nk}\sum_{i=1}^n\frac{n^2\sigma^2_i}{(n-1)^2}" />  
<img src="https://latex.codecogs.com/gif.latex?\sigma^2_i=[x_i-m]^T[x_i-m]" />  
  
However, <img src="https://latex.codecogs.com/gif.latex?m" /> is the average value of the output within the batch.  
  
<hr />
  
And at the time of learning, I will let you learn so that <img src="https://latex.codecogs.com/gif.latex?l_C" />, which is the variance of the output, also decreases with cross entropy <img src="https://latex.codecogs.com/gif.latex?l_D" />.  
The learning rate seems to be <img src="https://latex.codecogs.com/gif.latex?5\times10^{-5}" />, and the weight decay is set to 0.00005.  

## 6. Test phase
![04](media/04.png)  

- Remove **`h`** from the model.
- First, bring in the image from the learning data of the target data, put it in **`g`**, and obtain the distribution.
- Next, put the image you want to test in **`g`** and get the distribution.
- Finally, abnormality detection is performed by using the k-nearest neighbor method in "Distribution of image of training data" and "Distribution of test image".

## 7. Implementation by Keras
The learned model uses lightweight MobileNetV2.  

### 7-1. Load data
Data use this time is Fashion-MNIST.  
And I distributed the data as follows.  
  
||Number<br>of<br>data|Number<br>of<br>classes|Remarks|
|:--|--:|--:|:--|
|Reference data|6,000|8|Excluding sneakers and boots|
|Target data|6,000|1|sneakers|
|Test data（Normal）|1,000|1|sneakers|
|Test data（Abnormal）|1,000|1|boots|

  
**Logic 1 : [data_loader.py](data_loader.py)**  
<br>

### 7-2. Data resizing
In MobileNetv2, the minimum input size is <img src="https://latex.codecogs.com/gif.latex?(96\times96\times3)" />.  
Therefore, Fashion-MNIST <img src="https://latex.codecogs.com/gif.latex?(28\times28\times1)" /> can not be used as it is.  
So I will resize the data.  
  
**Logic 2 : [data_resizer.py](data_resizer.py)**  
<br>
  
The figure is as follows.  
![05](media/05.png)  
The left figure is original data <img src="https://latex.codecogs.com/gif.latex?(28\times28\times1)" />, the right figure is after resizing <img src="https://latex.codecogs.com/gif.latex?(96\times96\times3)" />.  

### 7-3. Model building and learning
During learning, the weight is fixed for the second half of the convolution layer.  
I will explain a part of the code here.  
Using Keras, building a model was easy, but building the following loss function was extremely difficult.  
```python
def original_loss(y_true, y_pred):
    lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]) / ((batchsize-1)**2)
    return lc
```
And the part to be careful is the following part.  
```python
#target data
#Get loss while learning
lc.append(model_t.train_on_batch(batch_target, np.zeros((batchsize, feature_out))))
            
#reference data
#Get loss while learning
ld.append(model_r.train_on_batch(batch_ref, batch_y))
```
**`model_t.train_on_batch`** gives a dummy zero matrix because any teacher data can be used.  
**`np.zeros((batchsize, feature_out))`**  
  
In addition, because it was very difficult to use Keras to simultaneously learn <img src="https://latex.codecogs.com/gif.latex?l_D" /> and <img src="https://latex.codecogs.com/gif.latex?l_C" />, I tried a method to let the machine learn <img src="https://latex.codecogs.com/gif.latex?l_D" /> after learning with <img src="https://latex.codecogs.com/gif.latex?l_C" />.  
Loss functions and simultaneous learning may be easily done with Pytorch.  
  
**Logic 3 : [train.py](train.py)**  
<br>
  
## 8. Result
### 8-1. Look at the distribution
Before looking at the performance of anomaly detection, visualize the distribution with t-sne.  
The figure below shows the image of the test data <img src="https://latex.codecogs.com/gif.latex?96\times96\times3" /> as it is visualized.  
![06](media/06.png)  
Even if I use input data as it is, sneakers and boots are separated considerably.  
However, some seem to be mixed.  
  
Next, the figure below visualizes the test data output (1280 dimension) with t-sne using CNN (MobileNetV2) learned with DOC.  
![07](media/07.png)  
It is well separated as in the previous figure.  
What I want to emphasize here, CNN is that **it only learns the image of sneakers (normal items)**.  
Nonetheless, it is surprising that sneakers and boots are well separated. It is just abnormality detection.  
Thanks to DOC learning about metastasis, it succeeded because CNN had learned the place to see the image beforehand.  
  
I will post the transition of loss function during learning.  
![08](media/08.png)  
![09](media/09.png)  

### 8-2. Abnormality detection performance
Next let's detect abnormality with **`g`** output. In the paper the k-neighbor method was used, but this implementation uses LOF.  
  
**Logic 4 : [calc_ROC.py](calc_ROC.py)**  
<br>
  
The ROC curve is as follows.  
![10](media/10.png)  
  
AUC has surprised value of **`0.90`**.  
By the way, the overall accuracy is **`about 83%`**.  
Compared with previous results, it is as follows.  
  
*VAE = Variational Autoencoder  
*Measurement speed is measured by Google Colaboratory's GPU  
*Visualization of DOC is explained in the next section

||Performance<br>(AUC)|Inference speed<br>(millisec/1 image)|Visualization<br>accuracy|
|:--|--:|--:|:-:|
|VAE(Small window)|0.58|**0.80**|×|
|VAE+Irregularization(Small window)|0.67|4.3|**◯**|
|**DOC(MobileNetV2)**|**0.90**|140|△|

DOC was a victory over VAE in performance, but at decision speed it is slow to use LOF.  
By the way, it was 370 millisec / 1 image when it was DOC + VGG16.  
MobileNetV2 is fast.  
Also, inferior accuracy "VAE + irregularity" was invented for complex images like screw threads.  
So, for complex images, the performance may be "VAE + irregularity > DOC".  
  
### 8-3. Relationship between images and abnormal scores
Next, let's look at the relationship between boots (abnormal items) images and abnormal scores.  
The larger the abnormality score, the more likely it is that it is different from sneakers (normal items).  
  
First of all, it is the image of the boots where the anomaly score was large, that is judged not to resemble sneakers altogether.  
![11](media/11.png)  
Sure, it does not look like a sneaker at all.  
  
Next, it is an image of an image with a small abnormality score, that is, boots judged to be very similar to sneakers.  
![12](media/12.png)  
It looks like a high-cut sneaker overall.  
Even if humans make this judgment, they may erroneously judge.  
Intuitively, the greater the abnormality score due to DOC, the more likely it is that it deviates from normal products.  

## 9. Visualization by Keras
I also tried visualization with Grad-CAM.  
It is also important to visualize where abnormality is.  
### 9-1. Grad-CAM
Grad-CAM is often used for CNN classification problems.  
When used in a classification problem, it shows the part that became the basis of that classification.  
For details, please see the following article.  
**[Visualization of deep learning gaze region - Qiita - bele_m](https://qiita.com/bele_m/items/a7bb15313e2a52d68865)**  
**[With Keras, Grad-CAM, a model I made by myself - Qiita - haru1977](https://qiita.com/haru1977/items/45269d790a0ad62604b3)**  
  
This time, I tried using Grad-CAM directly in DOC.  
### 9-2. Results
First of all, I tried using Grad-CAM in boots images where the abnormal score was large.  
![13](media/13.png)  
The vertical part of the heel part and the boots is red, and parts which are not totally similar to the sneakers are visualized.  
Next, I will try using Grad-CAM on boots with small anomaly scores.  
![14](media/14.png)  
The high cut part is red, and parts that are not similar to sneakers are visualized.  
On the whole, whether visualization succeeds or not is like feeling like 50%, there seems to be room for improvement, such as adding a Fully Connected Layer.  
The problem is that the processing time takes about 5 seconds / image (Colaboratory-GPU) time and it can not be used in real time.  
It seems necessary to select the scene to use.  

## 10. Implementation by RaspberryPi
- Cost is $100 or less (conventional products have over $9,000)
- Absolute detection accuracy is the highest peak (state-of-the-art at the time of publication)
- Compact (RaspberryPi and Web Camera only)
- Despite using deep learning at RaspberryPI it is fast (5 FPS)
- Application areas
  - Visual inspection of industrial products
  - Appearance inspection of bridges by drone **[here](http://www.tobu.co.jp/file/pdf/0f5fc3eafc73a1e18b62c9008c0385b4/newsletter_180413.pdf?date=20180413133524)**
  - Surveillance camera **[here](https://jpn.nec.com/press/201812/20181211_02.html)**  

![001](media/001.gif)
### 10-1. Environment
1. Numpy 1.15.4
2. scikit-learn 0.19.2+
3. Keras 2.2.4
4. OpenCV 4.0.1-openvino
### 10-2. How to use
If you want to try it immediately, please try Face Detection.

1.Execute below.  
```bash
$ sudo apt-get install -y python-pip python3-pip python3-scipy libhdf5-dev libatlas-base-dev
$ sudo -H pip3 install protobuf==3.6.0
$ sudo pip3 uninstall tensorflow
$ wget -O tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-1.11.0-cp35-cp35m-linux_armv7l_jemalloc.whl
$ sudo pip3 install tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl
$ rm tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl
$ sudo -H pip3 install scikit-learn==0.20.2
$ git clone --recursive https://github.com/PINTO0309/Keras-OneClassAnomalyDetection.git
$ cd Keras-OneClassAnomalyDetection
$ git submodule init
$ git submodule update
```
2.Connect USB Camera to RaspberryPi.  
3.Execute below.  
```bash
$ cd OneClassAnomalyDetection-RaspberryPi3/DOC
$ python3 main.py
```
4.When the real-time image of USB camera is drawn, please press **`s`** key.  
5.When the score is displayed, an abnormality test is started.

**[Note]**
- **Since abnormality test runs RaspberryPi fully, it will freeze with thermal runaway after about 5 minutes.**
- **When operating for a long time, please operate while cooling.**
- **If "human face" appears, the score falls. (Normal)**
- **If "human face" is not appeared, the score will rise. (Abnormality)**
- **Learning was executed with [CelebA](http://aidiary.hatenablog.com/entry/20180311/1520762446).**  
  
**＜Test of Corei7 CPU Only 320x240 / 11 FPS＞**  
![15](media/15.gif)  
**Youtube : https://youtu.be/p8BDwhF7y8w**  
  
**＜Test of Core m3 CPU Only + OpenVINO + 320x240 / 180 FPS＞**  
![31](media/31.gif)  
  
**＜Test of Core m3 CPU Only + OpenVINO + 640x480/ 70 FPS＞**  
![32](media/32.gif)  
  
**＜Test of Intel HD Graphics 615 + OpenVINO + 320x240 / 130 FPS＞**  
![33](media/33.gif)  
  
**＜Test of Intel HD Graphics 615 + OpenVINO + 640x480 / 70 FPS＞**  
![34](media/34.gif)  
  
### 10-3.Learning by original data set
For those who want to training models themselves, the technical contents are described below.  
The overall flow is as shown in the figure below.  
![16](media/16.png)  
Since the calculation load is not high, it is possible to complete all processes with RaspberryPi.  
However, in order to download CIFAR10, it needs to be processed by a terminal connected to the network.  
### 10-4.Acquisition of learning image data
First, you take pictures for learning.  
Please take a picture along the following notes.  
1. Delete the model folder from the downloaded DOC folder (booting will be faster).
2. Connect USB camera and execute DOC/main.py.
3. When the image of the USB Camera is displayed, press the **`p`** key to take a picture.
4. By shooting about 10 images, sufficient accuracy can be obtained.
### 10-5.Resizing the image for learning
After taking a picture, upload the DOC/pictures folder to Google Drive.  
And from here you will process it on Google Colaboratory.  
Please refer to **[The story that it was easy to mount on Google Drive at Colaboratory - Qiita - k_uekado](https://qiita.com/k_uekado/items/45b76f9a6f920bf0f786)** on how to mount Google Drive on Google Colaboratory.  
  
In order to let you learn on MobileNetV2, resize the image with the following code.  
```python
import cv2
import numpy as np
import os
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img

img_path = 'pictures/'
NO = 1

def resize(x):
    x_out = []

    for i in range(len(x)):
        img = cv2.resize(x[i],dsize=(96,96))
        x_out.append(img)

    return np.array(x_out)

x = []

while True:
    if not os.path.exists(img_path + str(NO) + ".jpg"):
        break
    img = Image.open(img_path + str(NO) + ".jpg")
    img = image.img_to_array(img)
    x.append(img)
    NO += 1

x_train = resize(x)
```
Before resizing on the left figure. <img src="https://latex.codecogs.com/gif.latex?(288\times288\times3)" />  
After resizing the figure on the right. <img src="https://latex.codecogs.com/gif.latex?(96\times96\times3)" />  
![17](media/17.png)  
### 10-6.Data Augmentation
The number of various data is as follows.  

|Contents|Number|Number<br>of<br>Classes|Note|
|:--|--:|--:|:--|
|Reference data|6,000|10|CIFAR10|
|Target data|6,000|1|Image of nut|

Infrate the captured image (target data) with Data Augmentation.  
```python
from keras.preprocessing.image import ImageDataGenerator

X_train = []
aug_num = 6000 # Number of DataAug
NO = 1

datagen = ImageDataGenerator(
           rotation_range=10,
           width_shift_range=0.2,
           height_shift_range=0.2,
           fill_mode="constant",
           cval=180,
           horizontal_flip=True,
           vertical_flip=True)

for d in datagen.flow(x_train, batch_size=1):
    X_train.append(d[0])
    # Because datagen.flow loops infinitely,
    # it gets out of the loop if it gets the required number of sheets.
    if (NO % aug_num) == 0:
        print("finish")
        break
    NO += 1

X_train = np.array(X_train)
X_train /= 255
```
Data Augmentation as follows.  
![18](media/18.png)  
Points are as follows.  
When moving the image in parallel, I specified the color to fill in the blanks.  
It is necessary to adjust this as necessary.  
```python
fill_mode="constant",
cval=180,
```
This time I want to detect color differences, so I am using regular Data Augmentation.  
However, if you want to detect only the shape of the object, it may be better to run **[PCA Data Augmentation](https://qiita.com/shinmura0/items/f818bcbb92d5fff5279d#data-augmentation)**.  
### 10-7.Generation of reference data
In this case I will use color image, use CIFAR10 image as reference data.  
```python
from keras.datasets import cifar10
from keras.utils import to_categorical

# dataset
(x_ref, y_ref), (x_test, y_test) = cifar10.load_data()
x_ref = x_ref.astype('float32') / 255

#6,000 randomly extracted from ref data
number = np.random.choice(np.arange(0,x_ref.shape[0]),6000,replace=False)

x, y = [], []

x_ref_shape = x_ref.shape

for i in number:
    temp = x_ref[i]
    x.append(temp.reshape((x_ref_shape[1:])))
    y.append(y_ref[i])

x_ref = np.array(x)
y_ref = to_categorical(y)

X_ref = resize(x_ref)
```
It will be resized as follows.  
Before resizing on the left figure. <img src="https://latex.codecogs.com/gif.latex?(32\times32\times3)" />  
After resizing the figure on the right. <img src="https://latex.codecogs.com/gif.latex?(96\times96\times3)" />  
![19](media/19.png)  
### 10-8.Training
Train according to **[Model building and learning](#7-3-model-building-and-learning)**.  
Save the model with the code below.  
A "model" folder is created under the execution path.  
```python
train_num = 1000# number of training data

model_path = "model/" 
if not os.path.exists(model_path):
    os.mkdir(model_path)

train = model.predict(X_train)

# model save
model_json = model.to_json()
open(model_path + 'model.json', 'w').write(model_json)
model.save_weights(model_path + 'weights.h5')
np.savetxt(model_path + "train.csv",train[:train_num],delimiter=",")
```
The "model" folder contains three files "model.json", "weights.h5" and "train.csv".  
**`train_num = 1000 # number of training data`** is an acceleration parameter of LOF.  
### 10-9.Execution of inference with RaspberryPi
1. copy the "model" folder right under the "DOC" folder of RaspberryPi.  
2. Execute **`main.py`**.  
```bash
$ cd OneClassAnomalyDetection-RaspberryPi3/DOC
$ python3 main.py
```
The **`threshold`** defined in **`main.py`** is the threshold.  
Beyond this the score will be in the red.  
Anomaly score is moving average of 10 inference.  
### 10-10.Results
The result of verification with nut image is shown.  
Pictures of normal products are as follows.  
![20](media/20.jpeg)  
  
Normal product A (score 1.3) → Abnormal product (rust, score 1.6)  
![21](media/21.gif)  
  
Normal product A (score 1.4) → Abnormal item (size difference, score 2 or more)  
![22](media/22.gif)  
  
Move the normal item A (scores of differences in position are hardly seen)  
![23](media/23.gif)  
  
Normal product B (score 1.2) → Normal product A (score 1.3)  
![24](media/24.gif)  

## 11. Acceleration of LOF
I will write about speedup of LOF.  
Unlike neural networks, LOF changes the inference time greatly depending on the number of learning data.  
The figure below shows the relationship between the number of LOF learning data and inference time, using sneaker data.  
![25](media/25.png)  
Since LOF stores learning data and detects abnormality just like the k-nearest neighbor method, as the number of learning data increases, inference time increases accidentally.  
What I would like to pay attention to here is that the transition of AUC is saturated when the learning data is 1000 pieces.  
In other words, even if there are more than 1000 training data, performance will not be affected, only inference time will increase.  
So, I am constructing LOF using 1000 learning data this time.  
As a result, performance and reasoning speed can be incorporated in a well-balanced manner, reducing the time by 180 msec or more.  
As a result, the inference time of DOC + LOF is about 200 msec (5 FPS) with RaspberryPi alone.  
If you use the GPU, it may run at 20 msec (50 FPS).  
## 12. Structure of the model
1. Execute below.  
```bash
$ sudo -H pip3 install netron
$ netron -b [MODEL_FILE]
```
2. Access http://localhost:8080 from the browser.  
<img src="media/26.png" width=20%><img src="media/30.png" width=20%><img src="media/27.png" width=20%><img src="media/28.png" width=45%><img src="media/29.png" width=15%>  

## 13. Model Convert
### 13-1. MMdnn
```bash
$ sudo -H pip3 install -U git+https://github.com/Microsoft/MMdnn.git@master
$ sudo -H pip3 install onnx-tf
$ mmconvert -h
usage: mmconvert [-h]
                 [--srcFramework {caffe,caffe2,cntk,mxnet,keras,tensorflow,tf,pytorch}]
                 [--inputWeight INPUTWEIGHT] [--inputNetwork INPUTNETWORK]
                 --dstFramework
                 {caffe,caffe2,cntk,mxnet,keras,tensorflow,coreml,pytorch,onnx}
                 --outputModel OUTPUTMODEL [--dump_tag {SERVING,TRAINING}]

optional arguments:
  -h, --help            show this help message and exit
  --srcFramework {caffe,caffe2,cntk,mxnet,keras,tensorflow,tf,pytorch}, -sf {caffe,caffe2,cntk,mxnet,keras,tensorflow,tf,pytorch}
                        Source toolkit name of the model to be converted.
  --inputWeight INPUTWEIGHT, -iw INPUTWEIGHT
                        Path to the model weights file of the external tool
                        (e.g caffe weights proto binary, keras h5 binary
  --inputNetwork INPUTNETWORK, -in INPUTNETWORK
                        Path to the model network file of the external tool
                        (e.g caffe prototxt, keras json
  --dstFramework {caffe,caffe2,cntk,mxnet,keras,tensorflow,coreml,pytorch,onnx}, -df {caffe,caffe2,cntk,mxnet,keras,tensorflow,coreml,pytorch,onnx}
                        Format of model at srcModelPath (default is to auto-
                        detect).
  --outputModel OUTPUTMODEL, -om OUTPUTMODEL
                        Path to save the destination model
  --dump_tag {SERVING,TRAINING}
                        Tensorflow model dump type
```
### 13-2. Keras -> Tensorflow -> OpenVINO
```bash
$ python3 keras2tensorflow/keras_to_tensorflow.py \
--input_model="OneClassAnomalyDetection-RaspberryPi3/DOC/model/weights.h5" \
--input_model_json="OneClassAnomalyDetection-RaspberryPi3/DOC/model/model.json" \
--output_model="models/tensorflow/weights.pb"
```
```bash
$ sudo python3 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \
--input_model models/tensorflow/weights.pb \
--output_dir irmodels/tensorflow/FP16 \
--input input_1 \
--output global_average_pooling2d_1/Mean \
--data_type FP16 \
--batch 1
```
```txt
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/home/xxxx/git/Keras-OneClassAnomalyDetection/models/tensorflow/weights.pb
	- Path for generated IR: 	/home/xxxx/git/Keras-OneClassAnomalyDetection/irmodels/tensorflow/FP16
	- IR output name: 	weights
	- Log level: 	ERROR
	- Batch: 	1
	- Input layers: 	input_1
	- Output layers: 	global_average_pooling2d_1/Mean
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	False
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Offload unsupported operations: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	None
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	None
Model Optimizer version: 	1.5.12.49d067a0
/usr/local/lib/python3.5/dist-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/xxxx/git/Keras-OneClassAnomalyDetection/irmodels/tensorflow/FP16/weights.xml
[ SUCCESS ] BIN file: /home/xxxx/git/Keras-OneClassAnomalyDetection/irmodels/tensorflow/FP16/weights.bin
[ SUCCESS ] Total execution time: 5.31 seconds. 
```
```bash
$ sudo python3 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py \
--input_model models/tensorflow/weights.pb \
--output_dir irmodels/tensorflow/FP32 \
--input input_1 \
--output global_average_pooling2d_1/Mean \
--data_type FP32 \
--batch 1
```
```txt
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/home/xxxx/git/Keras-OneClassAnomalyDetection/models/tensorflow/weights.pb
	- Path for generated IR: 	/home/xxxx/git/Keras-OneClassAnomalyDetection/irmodels/tensorflow/FP32
	- IR output name: 	weights
	- Log level: 	ERROR
	- Batch: 	1
	- Input layers: 	input_1
	- Output layers: 	global_average_pooling2d_1/Mean
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP32
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	False
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Offload unsupported operations: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	None
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	None
Model Optimizer version: 	1.5.12.49d067a0
/usr/local/lib/python3.5/dist-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/xxxx/git/Keras-OneClassAnomalyDetection/irmodels/tensorflow/FP32/weights.xml
[ SUCCESS ] BIN file: /home/xxxx/git/Keras-OneClassAnomalyDetection/irmodels/tensorflow/FP32/weights.bin
[ SUCCESS ] Total execution time: 5.59 seconds. 
```
### 13-3. Keras -> ONNX -> OpenVINO
```bash
$ mmconvert \
-sf keras \
-iw OneClassAnomalyDetection-RaspberryPi3/DOC/model/weights.h5 \
-in OneClassAnomalyDetection-RaspberryPi3/DOC/model/model.json \
-df onnx \
-om models/onnx/weights.onnx
```
```bash
$ sudo python3 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py \
--framework onnx \
--input_model models/onnx/weights.onnx \
--output_dir irmodels/onnx/FP16 \
--data_type FP16 \
--batch 1
```
```txt
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/home/xxxx/git/Keras-OneClassAnomalyDetection/models/onnx/weights.onnx
	- Path for generated IR: 	/home/xxxx/git/Keras-OneClassAnomalyDetection/irmodels/onnx/FP16
	- IR output name: 	weights
	- Log level: 	ERROR
	- Batch: 	1
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	False
ONNX specific parameters:
Model Optimizer version: 	1.5.12.49d067a0
[ ERROR ]  Cannot infer shapes or values for node "Conv1_relu".
[ ERROR ]  There is no registered "infer" function for node "Conv1_relu" with op = "Clip". Please implement this function in the extensions. 
 For more information please refer to Model Optimizer FAQ (<INSTALL_DIR>/deployment_tools/documentation/docs/MO_FAQ.html), question #37. 
[ ERROR ]  
[ ERROR ]  It can happen due to bug in custom shape infer function <UNKNOWN>.
[ ERROR ]  Or because the node inputs have incorrect values/shapes.
[ ERROR ]  Or because input shapes are incorrect (embedded to the model or passed via --input_shape).
[ ERROR ]  Run Model Optimizer with --log_level=DEBUG for more information.
[ ERROR ]  Stopped shape/value propagation at "Conv1_relu" node. 
 For more information please refer to Model Optimizer FAQ (<INSTALL_DIR>/deployment_tools/documentation/docs/MO_FAQ.html), question #38. 
```
### 13-4. Keras -> Caffe -> OpenVINO
```bash
$ mmconvert \
-sf keras \
-iw OneClassAnomalyDetection-RaspberryPi3/DOC/model/weights.h5 \
-in OneClassAnomalyDetection-RaspberryPi3/DOC/model/model.json \
-df caffe \
-om models/caffe/weights
```
```bash
$ sudo python3 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py \
--framework caffe \
--input_model models/caffe/weights.caffemodel \
--input_proto models/caffe/weights.prototxt \
--output_dir irmodels/caffe/FP16 \
--data_type FP16 \
--batch 1
```
```txt
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/home/xxxx/git/Keras-OneClassAnomalyDetection/models/caffe/weights.caffemodel
	- Path for generated IR: 	/home/xxxx/git/Keras-OneClassAnomalyDetection/irmodels/caffe/FP16
	- IR output name: 	weights
	- Log level: 	ERROR
	- Batch: 	1
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	False
Caffe specific parameters:
	- Enable resnet optimization: 	True
	- Path to the Input prototxt: 	/home/xxxx/git/Keras-OneClassAnomalyDetection/models/caffe/weights.prototxt
	- Path to CustomLayersMapping.xml: 	Default
	- Path to a mean file: 	Not specified
	- Offsets for a mean file: 	Not specified
Model Optimizer version: 	1.5.12.49d067a0
[ ERROR ]  Unexpected exception happened during extracting attributes for node block_14_depthwise_BN.
Original exception message: Found custom layer "DummyData2". Model Optimizer does not support this layer. Please, implement extension. 
 For more information please refer to Model Optimizer FAQ (<INSTALL_DIR>/deployment_tools/documentation/docs/MO_FAQ.html), question #45.
```
### 13-5. Keras -> PyTorch
```bash
$ mmconvert \
-sf keras \
-iw OneClassAnomalyDetection-RaspberryPi3/DOC/model/weights.h5 \
-in OneClassAnomalyDetection-RaspberryPi3/DOC/model/model.json \
-df pytorch \
-om models/pytorch/weights.pth
```
### 13-6. Keras -> Tensorflow -> Tensorflow Lite
```bash
$ python3 keras2tensorflow/keras_to_tensorflow.py \
--input_model="OneClassAnomalyDetection-RaspberryPi3/DOC/model/weights.h5" \
--input_model_json="OneClassAnomalyDetection-RaspberryPi3/DOC/model/model.json" \
--output_model="models/tensorflow/weights.pb"
```
```bash
$ cd ~
$ wget https://github.com/PINTO0309/Bazel_bin/blob/master/0.17.2/Raspbian_armhf/install.sh
$ sudo chmod +x install.sh
$ ./install.sh
$ git clone -b v1.11.0 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout v1.11.0
$ ./tensorflow/contrib/lite/tools/make/download_dependencies.sh
$ sudo bazel build tensorflow/contrib/lite/toco:toco
```
```bash
$ cd ~/tensorflow
$ mkdir output
$ cp ~/Keras-OneClassAnomalyDetection/models/tensorflow/weights.tflite.pb . #<-- "." required
$ sudo bazel-bin/tensorflow/contrib/lite/toco/toco \
--input_file=weights.pb  \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--output_file=output/weights.tflite \
--input_shapes=1,96,96,3 \
--inference_type=FLOAT \
--input_type=FLOAT \
--input_arrays=input_1 \
--output_arrays=global_average_pooling2d_1/Mean \
--post_training_quantize
```
## 14. Issue
**https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer#FAQ76**  
**https://software.intel.com/en-us/forums/computer-vision/topic/802689**  
  
**[OpenVINO Official FAQ](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer#inpage-nav-6)**  
