from keras.applications.mobilenetv2 import MobileNetV2#, VGG16
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras import backend as K
from keras.engine.network import Network

input_shape = (96, 96, 3)
classes = 10
batchsize = 128
#feature_out = 512 #secondary network out for VGG16
feature_out = 1280 #secondary network out for MobileNet
alpha = 0.5 #for MobileNetV2
lambda_ = 0.1 #for compact loss

#Loss function
def original_loss(y_true, y_pred):
    lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]) / ((batchsize-1)**2)
    return lc

#Learning
def train(x_target, x_ref, y_ref, epoch_num):

    #Read VGG16, S network
    print("Model build...")
    #mobile = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')

    #Read mobile net, S network
    mobile = MobileNetV2(include_top=False, input_shape=input_shape, alpha=alpha, depth_multiplier=1, weights='imagenet')

    #Fixed weight
    for layer in mobile.layers:
        if layer.name == "block_13_expand": #""block5_conv1":
            break
        else:
            layer.trainable = False

    model_t = mobile

    #R network　S and Weight sharing
    model_r = Network(inputs=model_t.input,
                      outputs=model_t.output,
                      name="shared_layer")

    #Apply a Fully Connected Layer to R
    x = model_t.output
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(classes, activation='softmax')(x)
    model_r = Model(inputs=model_r.input, outputs=prediction)

    #Compile
    optimizer = SGD(lr=5e-5, decay=0.00005)
    model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
    model_t.compile(optimizer=optimizer, loss=original_loss)

    model_t.summary()
    model_r.summary()

    print("x_target is",x_target.shape[0],'samples')
    print("x_ref is",x_ref.shape[0],'samples')

    ref_samples = np.arange(x_ref.shape[0])
    loss, loss_c = [], []

    print("training...")

    #Learning
    for epochnumber in range(epoch_num):
        x_r, y_r, lc, ld = [], [], [], []

        #Shuffle target data
        np.random.shuffle(x_target)

        #Shuffle reference data
        np.random.shuffle(ref_samples)
        for i in range(len(x_target)):
            x_r.append(x_ref[ref_samples[i]])
            y_r.append(y_ref[ref_samples[i]])
        x_r = np.array(x_r)
        y_r = np.array(y_r)

        for i in range(int(len(x_target) / batchsize)):

            #Load data for batch size
            batch_target = x_target[i*batchsize:i*batchsize+batchsize]
            batch_ref = x_r[i*batchsize:i*batchsize+batchsize]
            batch_y = y_r[i*batchsize:i*batchsize+batchsize]

            #target data
            #Get loss while learning
            lc.append(model_t.train_on_batch(batch_target, np.zeros((batchsize, feature_out))))

            #reference data
            #Get loss while learning
            ld.append(model_r.train_on_batch(batch_ref, batch_y))

        loss.append(np.mean(ld))
        loss_c.append(np.mean(lc))

        if (epochnumber+1) % 5 == 0:
            print("epoch:",epochnumber+1)
            print("Descriptive loss:", loss[-1])
            print("Compact loss", loss_c[-1])

    #Result graph
    plt.plot(loss,label="Descriptive loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    plt.plot(loss_c,label="Compact loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()    

    return model_t

model = train(X_train_s, X_ref, y_ref, 5)
