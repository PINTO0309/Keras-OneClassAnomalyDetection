# Keras-OneClassAnomalyDetection
Learning Deep Features for One-Class Classification(AnomalyDetection).  
  
**[Jan 06, 2019] Start of work. Under construction.**  

# Introduction
This repository was created inspired by  
**[Image abnormality detection using deep learning ーPapers and implementationー - Qiita - shinmura0](https://qiita.com/shinmura0/items/cfb51f66b2d172f2403b)**.  
I would like to express my deepest gratitude for having pleasantly accepted his skill, consideration and article quotation.  
His articles that were supposed to be used practically, not limited to logic alone, are wonderful.  
However, I don't have the skills to read papers, nor do I have skills to read mathematical expressions.  
I only want to verify the effectiveness of his wonderful article content in a practical range.  

# Translating shinmura0's article
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
In the future, I want to implement it with RaspberryPi3.  

### 7-1. Load data
Data use this time is Fashion-MNIST.  
And I distributed the data as follows.  
  
||Number<br>of<br>data|Number<br>of<br>classes|Remarks|
|:--|--:|--:|:--|
|Reference data|6,000|8|Excluding sneakers and boots|
|Target data|6,000|1|sneakers|
|Test data（Normal）|1,000|1|sneakers|
|Test data（Abnormal）|1,000|1|boots|

***Logic 1**  
  
### 7-2. Data resizing
In MobileNetv2, the minimum input size is <img src="https://latex.codecogs.com/gif.latex?(96\times96\times3)" />.  
Therefore, Fashion-MNIST <img src="https://latex.codecogs.com/gif.latex?(28\times28\times1)" /> can not be used as it is.  
So I will resize the data.  

***Logic 2**  
  
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
  
***Logic 3**  

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
<img src="media/09.png" width=56%>  

### 8-2. Abnormality detection performance
Next let's detect abnormality with **`g`** output. In the paper the k-neighbor method was used, but this implementation uses LOF.  
  
***Logic 4**  
  
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
**[Visualization of deep learning gaze region - Qiita - bele_m](https://qiita.com/bele_m/items/a7bb15313e2a52d68865)**  
**[With Keras, Grad-CAM, a model I made by myself - Qiita - haru1977](https://qiita.com/haru1977/items/45269d790a0ad62604b3)**  
  
This time, I tried using Grad-CAM directly in DOC.  
### 9-2. Results
