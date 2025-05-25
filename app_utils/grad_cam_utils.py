import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array_processed, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array_processed)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def generate_gradcam_overlay(original_img_array_uint8, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (original_img_array_uint8.shape[1], original_img_array_uint8.shape[0]))
    heatmap_rgb = np.uint8(255 * heatmap_resized)
    heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_rgb, alpha, original_img_array_uint8, 1 - alpha, 0)
    return superimposed_img