def vgg16_model (num_classes = None ) :
    model = VGG16(weights = 'imagenet', include_top = True, input_shape = (28, 28, 1))
    x = Dense(1024, activation = 'relu')(model.layers[-4].output)
    x = Dropout(0.7)(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation = 'softmax')(x)
    model = Model(model.input, x)
    return model 
