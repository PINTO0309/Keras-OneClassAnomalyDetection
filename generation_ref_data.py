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
