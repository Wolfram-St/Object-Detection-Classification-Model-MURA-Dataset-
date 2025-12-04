import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


model_path = 'models/xray_mobilenetv2.h5'


@st.cache_resource
def get_model():
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        return None


def get_heatmap(model, img_array):
    
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer.name
            break
            
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
        
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    return heatmap

# App Layout
st.title("X-Ray Abnormality Detector with Heatmap Explanation")
st.write("Upload a image")

model = get_model()

if model is None:
    st.error("Could not load the model. Please check if 'models/xray_mobilenetv2.h5' exists.")
else:
    file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file).convert('RGB')
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        with st.spinner('Analyzing...'):
            preds = model.predict(img_batch)
            score = float(preds[0][0])
            
            heatmap = get_heatmap(model, img_batch)
            
            heatmap = cv2.resize(heatmap, (image.width, image.height))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original X-Ray")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Analysis Overlay")
            st.image(overlay, use_container_width=True, caption="Red areas show abnormalities")
            
        st.divider()
        
        if score > 0.5:
            confidence = score * 100
            st.error(f"Prediction: ABNORMAL ({confidence:.2f}%)")
            st.write("The AI detected potential fractures or abnormalities in the highlighted regions.")
        else:
            confidence = (1 - score) * 100
            st.success(f"Prediction: NORMAL ({confidence:.2f}%)")
            st.write("The AI did not find significant abnormalities.")