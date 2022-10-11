#######################################
#                                     #
#    STREAMLIT APP BASED GAN          #
#                                     #
#######################################


from email.policy import default
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import ImageEnhance
from tensorflow.keras import layers
from tensorflow.keras.utils import array_to_img, img_to_array

st.title("Car GAN")

"""
This is a demonstration application built to show the working of Generative Advesarial Network which uses the ideas from
[DCGAN](https://arxiv.org/pdf/2006.14380.pdf), [WGAN](https://arxiv.org/pdf/2006.14380.pdf) and [BoolGAN](https://arxiv.org/pdf/2006.14380.pdf).
This application is built using [Streamlit](https://streamlit.io/) as a part of MBRDI's Datathon organized in [KSHITIJ 2022](https://ktj.in/)
"""

model = None
contrast, brightness, rows, cols, infer = 1.0, 1.0, 5, 5, 0.0


def make_generator_model():
    # function to create model for the generator
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(
        128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())

    model.add(layers.Conv2DTranspose(
        64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())

    model.add(layers.Conv2DTranspose(
        64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1),
                                     padding='same', use_bias=False, activation='tanh'))
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2),
                            padding='same', input_shape=[128, 128, 3]))
    model.add(layers.ELU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.ELU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.ELU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# @st.experimental_singleton()
def get_model():
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    discriminator_optimizer = tf.keras.optimizers.Adam(5e-5)
    generator_optimizer = tf.keras.optimizers.Adam(5e-5)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # Debug generator weights
    checkpoint.restore('training_checkpoints\\training_checkpoints\\ckpt-30')
    return generator


# with st.spinner("Loading model..."):
model = get_model()
st.success('Model loaded successfully!!')


def generate_image(random_vector_for_generation):
    generated_image = model(random_vector_for_generation, training=False)
    generated_image = tf.squeeze(generated_image)
    generated_image = tf.math.divide(tf.math.add(generated_image, 1), 2)
    return generated_image.numpy()


st.sidebar.title("Parameters")
contrast = st.sidebar.slider('Contrast', 0.1, 5.0, 1.4)
brightness = st.sidebar.slider('Brightness', 0.1, 5.0, 1.1)
cols = st.sidebar.slider('Columns', 1, 10, 5)
rows = st.sidebar.slider('Rows', 1, 10, 5)
st.sidebar.info("""
    This application is built using [Streamlit](https://streamlit.io/) as a part of MBRDI's Datathon organized in [KSHITIJ 2022](https://ktj.in/)
    """)


def display_grid():
    start_time = time.time()
    images = np.zeros((128*rows, 128*cols, 3))
    for x in range(rows):
        for y in range(cols):
            random_vector_for_generation = tf.random.normal([1, 100])
            generated_image = generate_image(random_vector_for_generation)
            images[x*128:(x+1)*128, y*128:(y+1)*128, :] = generated_image
    end_time = time.time()

    global infer
    infer = (end_time - start_time) * 1000 / (rows * cols)
    print(f"Average inference time: {infer:.2f} ms")
    st.sidebar.warning(f"Average inference time: {infer:.2f} ms")

    clear_image = array_to_img(images)
    clear_image = ImageEnhance.Contrast(clear_image).enhance(contrast)
    clear_image = ImageEnhance.Brightness(clear_image).enhance(brightness)
    images = img_to_array(clear_image) / 255.

    st.image(images, use_column_width=True)


if st.button("Generate"):
    display_grid()
