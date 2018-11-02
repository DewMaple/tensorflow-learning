import keras
from keras import layers, Model, losses
from keras.datasets import cifar10
from keras.optimizers import Adam


def build_model(num_classes):
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(10, (3, 3), activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME', name='max_pool1')(x)

    x = layers.Flatten()(x)
    
    x = layers.Dense(128, name='fc1')(x)
    prediction = layers.Dense(num_classes, activation='softmax', name='prediction')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.summary()
    return model


def main():
    num_classes = 10
    model = build_model(num_classes)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(type(x_train))

    print(x_train.shape)
    model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
