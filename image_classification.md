# Image classification

what is achieved
- spot nuance in the photo
- search untagged photos with keywords
- automated tags

why does weighted average of pixel RGB value not work?
- factors that has nothing to do with the object you look for may affects pixel value
    1. light
    2. camera angle
    3. backgroup object
- thus, we need to transform pixel values so that object can be distinguished from the noises

CNN(convolution neural network)
- extract texture and shape from features without transforming dataset
- 3D matrix
    - length of pixel
    - wideth of pixel
    - RGB channel
- convolution
    - extract tiles from input feature map, applies filter to the tiles, and produces a convolved feature map(also called output feature map)
    - parameters
        - depth of the map
            - number of filters
        - size of the tile
            - e.g. 3x3 or 5x5
    - padding
        - blank cells are added to the feature map so that size of output feature map is larger
    - steps
        - slide filters over input feature map
            - extract new feature for a convolved map

implication of more filters
- more computation
- more extracted features

ReLU is applied to convolved feature
- model has nonlinearity

pooling downsample the convolved features
- max pooling
    - slide a tiles over the feature map
    - extract the max from the tile
    - output max value to the new feature map

fully connected layer
- classify convolved features
- every node in the first layer is connected to every node in the second layer
- use softmax function

example: predict whether dogs or cats are in the image
- configuration
    - data split
        - training set
            - 1000 dogs, 1000 cats
        - validation set
            - 500 dogs, 500 cats
    - 150 x 150, RGB image
    - below stack * 3
        - convolution
            - 3 x 3, 16 filters
        - relu
            - 3 x 3, 32 filters
        - max pooling
            - 2 x 2
    - fully connected layer
        - 512 hidden units by flatten convolved map with relu activation
    - generate output with sigmoid
    - approach for gradient descent: rms prop
    - define loss: binary_crossentropy
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 150, 150, 3)]     0

 conv2d (Conv2D)             (None, 148, 148, 16)      448

 max_pooling2d (MaxPooling2  (None, 74, 74, 16)        0
 D)

 conv2d_1 (Conv2D)           (None, 72, 72, 32)        4640

 max_pooling2d_1 (MaxPoolin  (None, 36, 36, 32)        0
 g2D)

 conv2d_2 (Conv2D)           (None, 34, 34, 64)        18496

 max_pooling2d_2 (MaxPoolin  (None, 17, 17, 64)        0
 g2D)

 flatten_1 (Flatten)         (None, 18496)             0

 dense (Dense)               (None, 512)               9470464

 dense_1 (Dense)             (None, 1)                 513

=================================================================
Total params: 9494561 (36.22 MB)
Trainable params: 9494561 (36.22 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
- note about example
    - each filter represent a feature that model likes to learn about the picture
    - sparsity
        - downstream layer only output what is important
            - e.g. you see contour of the animal without background object

[sample code](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/pc/exercises/image_classification_part2.ipynb?utm_source=practicum-IC&utm_campaign=colab-external&utm_medium=referral&hl=en&utm_content=imageexercise2-colab#scrollTo=NKCjHegASXaA) that train CNN with dropout and augmentation
```python
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

base_dir = 'path_to_the_root_of_image_folder'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
img_input = layers.Input(shape=(150, 150, 3))
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Convolution2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = Model(img_input, output)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
```

Check out [Kera image data loading](https://keras.io/api/data_loading/image/) to see how folder should be structured and how image is loaded


pretained model
- goal
    - Training CNN with large dataset takes a lot of time. We like to speed it up
    - Use model that has been trained
- With trained model, generate output that has high level feature(color, texture, contour...etc) and then feed those output to your own model

