import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.nn import softmax

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from IPython.display import Image, display

from model import AttentionLSTM
import sys

model_builder = AttentionLSTM(15)

def get_img_seq_array(path):
    index = np.arange(15)*10
    img_seq = np.load(path)[index]
    return np.expand_dims(img_seq, axis=0)

img_array = get_img_seq_array(sys.argv[1])

# Make model
# model = model_builder(weights="imagenet")
model = model_builder.generate()
model.load_weights("checkpoint/inception.005-0.60.hdf5")
#for l in model.layers: print("here1", l.name, l.trainable, l.trainable)
# Remove last layer's softmax
model.layers[-1].activation = None

last_conv_layer = "time_distributed_1"

grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
)

heatmap_npy = None
with tf.GradientTape() as tape:
    last_conv_layer_output_seq, preds = grad_model([img_array, np.zeros((1, 15, 12)), np.zeros((1, 15, 30))])
    pred_index = tf.argmax(preds[0])
    print("pred_index, confidence", pred_index, softmax(preds[0])[pred_index])
    class_channel = preds[:, pred_index]
    grads_seq = tape.gradient(class_channel, last_conv_layer_output_seq)
    heatmap_seq = []
    
    for i in range(15):
        grads = grads_seq[:, i]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output_seq[0, i]
        print("last_conv_layer", last_conv_layer_output.get_shape())
        print("pooled_grads", pooled_grads.get_shape())
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap_npy = heatmap.numpy()
        heatmap_seq.append(heatmap_npy)
heatmap_seq = np.array(heatmap_seq)
print(heatmap_seq.shape)


def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

img = img_array.squeeze()
heatmap = heatmap_seq.squeeze()
print(img.shape, heatmap.shape)
for i in range(15):
    save_and_display_gradcam(img[i], heatmap[i], cam_path=sys.argv[2] + "cam_{}.png".format(i))
