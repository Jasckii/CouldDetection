import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class CloudDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cloud Detection System")
        self.root.geometry("900x750")
        
        # 加载模型
        self.setup_models()
        self.image_path = None
        self.display_image = None
        self.last_conv_layer_name = None  # 将在setup_models中确定
        
        # GUI组件初始化
        self.create_widgets()
    
    def setup_models(self):
        try:
            # 加载Keras模型
            model_path = 'model-cnn-v1.keras'
            self.cnn_model = load_model(model_path)
            
            # 查找最后一个卷积层的名称
            conv_layers = [layer.name for layer in self.cnn_model.layers if 'conv' in layer.name.lower()]
            if conv_layers:
                self.last_conv_layer_name = conv_layers[-1]
                print(f"找到最后一个卷积层: {self.last_conv_layer_name}")
            else:
                messagebox.showwarning("Warning", "未找到卷积层，Grad-CAM可能无法正常工作")
                self.last_conv_layer_name = None
            
        except Exception as e:
            messagebox.showerror("Error", f"加载模型失败: {str(e)}")
    
    def preprocess_image(self, image_path):
        # 读取图像
        image = cv2.imread(image_path)
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 保存原始图像用于显示
        self.original_img = image.copy()
        # 调整图像大小为256x256
        image = cv2.resize(image, (256, 256))
        # 归一化
        image = image.astype('float32') / 255.0
        # 增加批次维度
        image = np.expand_dims(image, axis=0)
        return image
    
    def create_widgets(self):
        # 上传按钮
        self.upload_btn = tk.Button(
            self.root, 
            text="Upload Image",
            command=self.upload_image,
            height=2,
            width=15
        )
        self.upload_btn.pack(pady=10)
        
        # 图片显示区域
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # 检测按钮框架
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # CNN检测按钮
        self.cnn_btn = tk.Button(
            btn_frame,
            text="CNN Detection",
            command=self.cnn_detect,
            height=2,
            width=15
        )
        self.cnn_btn.pack(side=tk.LEFT, padx=10)
        
        # ATT-CNN检测按钮
        self.att_cnn_btn = tk.Button(
            btn_frame,
            text="ATT-CNN Detection",
            command=self.att_cnn_detect,
            height=2,
            width=15
        )
        self.att_cnn_btn.pack(side=tk.LEFT, padx=10)
        
        # Grad-CAM按钮
        self.gradcam_btn = tk.Button(
            btn_frame,
            text="Generate Grad-CAM",
            command=self.generate_gradcam,
            height=2,
            width=15
        )
        self.gradcam_btn.pack(side=tk.LEFT, padx=10)
        
        # 结果显示区域
        self.result_text = tk.Text(
            self.root,
            height=5,
            width=50,
            font=("Arial", 12)
        )
        self.result_text.pack(pady=20)
        
        # Grad-CAM结果显示
        self.gradcam_label = tk.Label(self.root)
        self.gradcam_label.pack(pady=10)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            # 转换为绝对路径
            self.image_path = os.path.abspath(file_path)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"已加载图片: {self.image_path}\n")
            
            # 显示图片
            image = Image.open(self.image_path)
            # 调整图片大小以适应显示
            display_size = (400, 400)
            image = image.resize(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.display_image = photo
            self.image_label.config(image=photo)
            
            # 清除之前的Grad-CAM结果
            self.gradcam_label.config(image='')
    
    def cnn_detect(self):
        if not self.image_path:
            messagebox.showerror("Error", "请先上传图片!")
            return
        
        try:
            # 预处理图像
            input_image = self.preprocess_image(self.image_path)
            
            # 使用模型进行预测
            predictions = self.cnn_model.predict(input_image)
            
            # 处理预测结果
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # 清除之前的结果
            self.result_text.delete(1.0, tk.END)
            
            # 显示新结果
            result_str = f"CNN检测结果:\n"
            result_str += f"预测类别: {predicted_class}\n"
            result_str += f"置信度: {confidence:.4f}\n"
            
            self.result_text.insert(tk.END, result_str)
            
        except Exception as e:
            messagebox.showerror("Error", f"检测失败: {str(e)}")
    
    def att_cnn_detect(self):
        if not self.image_path:
            messagebox.showerror("Error", "请先上传图片!")
            return
        
        # 如果您有ATT-CNN模型，可以实现类似的检测逻辑
        messagebox.showinfo("Info", "ATT-CNN模型检测尚未实现")
    
    def get_gradcam_heatmap(self, img_array, pred_index=None):
        if self.last_conv_layer_name is None:
            messagebox.showerror("Error", "未找到卷积层，无法生成Grad-CAM")
            return None
            
        # 创建一个模型，输出最后卷积层和预测
        grad_model = tf.keras.models.Model(
            inputs=[self.cnn_model.inputs],
            outputs=[
                self.cnn_model.get_layer(self.last_conv_layer_name).output,
                self.cnn_model.output
            ]
        )
        
        # 计算类激活图的梯度
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # 提取卷积层关于目标类的梯度
        grads = tape.gradient(class_channel, conv_outputs)
        
        # 取特征图的全局平均
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 将特征图与权重相乘，创建类激活图
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # 标准化热图
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()
    
    def generate_gradcam(self):
        if not self.image_path:
            messagebox.showerror("Error", "请先上传图片!")
            return
            
        try:
            # 预处理图像
            img_array = self.preprocess_image(self.image_path)
            
            # 先进行预测，获取预测类别
            predictions = self.cnn_model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            
            # 获取Grad-CAM热图
            heatmap = self.get_gradcam_heatmap(img_array, pred_index)
            if heatmap is None:
                return
                
            # 读取原始图像
            img = cv2.imread(self.image_path)
            
            # 调整热图大小以匹配原始图像
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # 转换热图为RGB格式
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 混合热图和原始图像
            superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            
            # 保存结果
            output_path = os.path.join(os.path.dirname(self.image_path), 'gradcam_result.jpg')
            cv2.imwrite(output_path, superimposed_img)
            
            # 显示结果
            gradcam_img = Image.open(output_path)
            display_size = (400, 400)
            gradcam_img = gradcam_img.resize(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(gradcam_img)
            self.gradcam_image = photo  # 保持引用以防垃圾回收
            self.gradcam_label.config(image=photo)
            
            # 更新结果文本
            self.result_text.insert(tk.END, f"\nGrad-CAM可视化已生成\n保存路径: {output_path}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"生成Grad-CAM失败: {str(e)}")
            import traceback
            traceback.print_exc()

# 主程序执行
if __name__ == "__main__":
    root = tk.Tk()
    app = CloudDetectionGUI(root)
    root.mainloop()