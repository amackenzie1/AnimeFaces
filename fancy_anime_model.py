import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, ReLU, Dense, Attention, Dropout
import tensorflow.keras.backend as backend

if backend.backend()  =='tensorflow':
    backend.set_image_data_format("channels_last")

# a standard UNet 

SIZES = [128, 256, 256]
RES = 32
name = 0

def get_model(strategy):
    with strategy.scope():
        input = tf.keras.Input(shape=(RES, RES, 3), name='input')
        time_embedding = tf.keras.Input(shape=(128), name='time')
        time1 = Dense(SIZES[0]*4, activation='relu')(time_embedding)
        time1 = Dense(SIZES[0]*4, activation='relu')(time1)


        def attention_block(filters, input):
            global name
            x = BatchNormalization()(input)
            b, h, w, c = tf.shape(x)
            x = Conv2D(filters*3, (1, 1), padding='same', name=f"attn_conv_{name}")(x)
            name += 1
            q, k, v = tf.split(x, num_or_size_splits=3, axis=-1)
            q = tf.reshape(q, (b, h*w, c))
            k = tf.reshape(k, (b, h*w, c))
            v = tf.reshape(v, (b, h*w, c))
            x = Attention()([q, k, v]) 
            x = tf.reshape(x, (b, h, w, filters))
            x = Conv2D(filters, (1, 1), padding='same')(x)
            # make sure the reshape works good?
            return x 

        def resnet_block(filters, input, attention=False):
            # two convolutional layers with a residual connection
            x = Conv2D(filters, (3, 3), padding='same')(input)
            time = Dense(filters, activation='relu')(time1)[:, None, None, :]
            x = BatchNormalization()(x + time)
            x = ReLU()(x)
            x = Dropout(0.1)(x)
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            y = Conv2D(filters, (1, 1), padding='same')(input)
            if attention:
                x = attention_block(filters, x)
            x = ReLU()(x + y)
            return x
            

        intermediates = []
        x = None 
        for i, size in enumerate(SIZES[:-1]):
            if x is None:
                x = resnet_block(size, input)
                # x = resnet_block(size, x)
            else:
                x = resnet_block(size, x) 
                # x = resnet_block(size, x)
            intermediates.append(x)
            x = MaxPooling2D((2, 2))(x)

        x = resnet_block(SIZES[-1], x, attention=True)
        x = resnet_block(SIZES[-1], x)

        for i, size in reversed(list(enumerate(SIZES[:-1]))):
            x = Conv2DTranspose(size, (2, 2), strides=(2, 2))(x)
            x = tf.concat([intermediates[i], x], axis=-1)
            x = resnet_block(size, x)
            # x = resnet_block(size, x)

        x = Conv2D(3, (1, 1), activation='linear', padding='same')(x)

        return tf.keras.Model(inputs=[input, time_embedding], outputs=x)

if __name__ == "__main__":
    print("Need a strategy! Just edit the file to remove that.")
    model = get_model()
    model.summary()
    
