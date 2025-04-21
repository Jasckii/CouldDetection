import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class CloudDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cloud Detection System")
        self.root.geometry("800x600")
        
        # 图片变量
        self.image_path = None
        self.display_image = None
        
        self.create_widgets()
        
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
        
        # 结果显示区域
        self.result_label = tk.Label(
            self.root,
            text="Detection Result: ",
            font=("Arial", 12)
        )
        self.result_label.pack(pady=20)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.image_path = file_path
            # 显示图片
            image = Image.open(file_path)
            # 调整图片大小以适应显示
            image = image.resize((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.display_image = photo
            self.image_label.config(image=photo)
            
    def cnn_detect(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an img first!")
            return
        # 这里添加您的CNN模型检测代码
        # result = your_cnn_model(self.image_path)
        result = "CNN Detection Result" # 替换为实际结果
        self.result_label.config(text=f"Detection Result: {result}")
        
    def att_cnn_detect(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an img first!")
            return
        # 这里添加您的ATT-CNN模型检测代码
        # result = your_att_cnn_model(self.image_path)
        result = "ATT-CNN Detection Result" # 替换为实际结果
        self.result_label.config(text=f"Detection Result: {result}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CloudDetectionGUI(root)
    root.mainloop()