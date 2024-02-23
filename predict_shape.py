from keras.models import load_model
import time
import numpy as np
t1 = time.time()

models_path = ['model_DenseNet121_C3.h5',
               'model_MobileNetV2_C3.h5',
               'model_NASNetMobile_C3.h5']

models = [load_model(i) for i in models_path]
t2 = time.time()
print(t2-t1)


def predict(img):
    categorical = {0: 'Circle',
                   1: 'star',
                   2: 'triangle'}
    predictions = []
    img = img.astype('float32')/255
    img = np.array([img])
    for model in models:
        propability_class = model.predict(img)
        img_cls = np.argmax(propability_class)
        predictions.append(categorical[img_cls])
    return predictions
