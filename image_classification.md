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
