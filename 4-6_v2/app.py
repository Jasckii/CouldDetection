import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer
import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

# GradCAM Implementation
class GradCAM:
    def __init__(self, model, last_conv_layer_name):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        
        # Create gradient model
        try:
            self.grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[
                    model.get_layer(last_conv_layer_name).output,
                    model.output
                ]
            )
        except ValueError as e:
            print(f"Failed to create gradient model: {e}")
            raise e

    def compute_heatmap(self, img_array, class_idx=0, eps=1e-8):
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(preds[0])
            class_channel = preds[:, class_idx]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        heatmap = heatmap.numpy()
        
        return heatmap

    def overlay_heatmap(self, heatmap, original_img, alpha=0.5):
        # Ensure correct heatmap size
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Convert to colorful heatmap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert to the same data type
        original_img = original_img.astype(np.float32)
        heatmap = heatmap.astype(np.float32)
        
        # Overlay heatmap
        superimposed_img = cv2.addWeighted(original_img, alpha, heatmap, 1-alpha, 0)
        
        return superimposed_img

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
cnn_model = None
att_cnn_model = None
cnn_gradcam = None
att_cnn_gradcam = None

# Load models when app starts
def load_models():
    global cnn_model, att_cnn_model, cnn_gradcam, att_cnn_gradcam
    
    try:
        # Register custom layer
        custom_objects = {'AttentionLayer': AttentionLayer}
        
        # Try loading .keras format models first
        try:
            cnn_model = load_model("modle-cnn-v2.keras", compile=False)
            att_cnn_model = load_model("modle-att-cnn-v2.keras", compile=False, custom_objects=custom_objects)
            print("Models loaded successfully (.keras format)!")
        except Exception as e:
            # If .keras fails, try .h5 format
            cnn_model = load_model("modle-cnn-v2.h5", compile=False)
            att_cnn_model = load_model("modle-att-cnn-v2.h5", compile=False, custom_objects=custom_objects)
            print("Models loaded successfully (.h5 format)!")
        
        # Find last convolutional layers for GradCAM
        cnn_last_conv_layer = None
        for layer in reversed(cnn_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                cnn_last_conv_layer = layer.name
                break
        
        att_cnn_last_conv_layer = None
        for layer in reversed(att_cnn_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                att_cnn_last_conv_layer = layer.name
                break
        
        if cnn_last_conv_layer and att_cnn_last_conv_layer:
            cnn_gradcam = GradCAM(cnn_model, cnn_last_conv_layer)
            att_cnn_gradcam = GradCAM(att_cnn_model, att_cnn_last_conv_layer)
            print("GradCAM initialized successfully!")
        else:
            print("Warning: Could not find convolutional layers for GradCAM")
    
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models on startup
load_models()

# Helper functions
def preprocess_image(img_file):
    img = Image.open(img_file)
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization
    
    return img_array, img

def generate_gradcam(img_array, original_img, model_type):
    gradcam = cnn_gradcam if model_type == 'cnn' else att_cnn_gradcam
    if not gradcam:
        return None
    
    # Calculate heatmap
    heatmap = gradcam.compute_heatmap(img_array)
    
    # Convert PIL img to numpy array
    original_img_array = np.array(original_img)
    
    # Apply heatmap
    superimposed_img = gradcam.overlay_heatmap(heatmap, original_img_array)
    
    # Convert to base64 for web display
    plt.figure(figsize=(5, 5))
    plt.imshow(superimposed_img / 255.0)
    plt.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'img' not in request.files:
        return jsonify({'error': 'No img provided'}), 400
    
    file = request.files['img']
    if file.filename == '':
        return jsonify({'error': 'No img selected'}), 400
    
    if file:
        # Get model type from form
        model_type = request.form.get('model', 'cnn')
        
        try:
            # Preprocess img
            img_array, original_img = preprocess_image(file)
            
            # Select model
            current_model = cnn_model if model_type == 'cnn' else att_cnn_model
            
            if current_model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            # Make prediction
            prediction = current_model.predict(img_array)[0][0]
            
            # Create GradCAM heatmap
            heatmap_img = generate_gradcam(img_array, original_img, model_type)
            
            # Save original img for display
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.jpg')
            original_img.save(img_path)
            
            # Return results
            return jsonify({
                'prediction': float(prediction),
                'cloud_probability': float(prediction * 100),
                'no_cloud_probability': float((1 - prediction) * 100),
                'original_image': f'static/uploads/temp.jpg',
                'heatmap_image': f'data:img/png;base64,{heatmap_img}' if heatmap_img else None
            })
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 