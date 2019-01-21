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
