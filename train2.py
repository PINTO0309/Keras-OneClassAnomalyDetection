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
