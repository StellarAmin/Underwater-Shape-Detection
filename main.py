from keras.models import load_model

model = load_model('model_DenseNet121_C3.h5')
model.summary()