import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import threading

# 从文本文件中创建类别名称字典
def create_class_names_dict(file_path):
    class_names = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            class_names[i] = line.strip()
    return class_names

# 读取防治建议
def create_advice_dict(file_path):
    advice_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(';')
            if len(parts) == 2:
                advice_dict[parts[0]] = parts[1]
    return advice_dict

# 设置模型和设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = models.resnet18(weights=None)
num_ftrs = net.fc.in_features
num_classes = 88  #模型X类
net.fc = torch.nn.Linear(num_ftrs, num_classes)
net.load_state_dict(torch.load('E:\\ModelWeights\\plant_disease_model_interrupted.pth')) #导入训练好的模型
net.to(device)
net.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 读取类别名称和防治建议
class_names = create_class_names_dict('E:\\TrainPicture\\Plant Disease Classification Merged Dataset.txt')
advice_dict = create_advice_dict('E:\\TrainPicture\\Device medicle.txt')

# 创建UI窗口
def create_ui():
    root = tk.Tk()
    root.title("一种基于旋翼无人机智能农业病虫害监测与防控系统研究")
    root.geometry("900x600")

    # 侧边栏框架
    sidebar = tk.Frame(root, width=100, bg='gray')
    sidebar.pack(fill='y', side='left')

    # 上传图片按钮
    btn_select_image = tk.Button(sidebar, text="上传图片", padx=10, pady=5, command=select_image)
    btn_select_image.pack(padx=10, pady=5, fill='x')

    # 开始识别按钮
    btn_predict = tk.Button(sidebar, text="开始识别", padx=10, pady=5, command=lambda: threading.Thread(target=predict).start())
    btn_predict.pack(padx=10, pady=5, fill='x')

    # 关闭按钮
    btn_close = tk.Button(sidebar, text="关闭", padx=10, pady=5, command=root.destroy)
    btn_close.pack(padx=10, pady=5, fill='x')

    # 主内容区域
    content = tk.PanedWindow(root, bg='white', orient='vertical')
    content.pack(expand=True, fill='both', side='right')

    # 图像状态标签
    global label_image_status
    label_image_status = tk.Label(content, text="待处理的图片", font=("Arial", 20), bg='white')
    content.add(label_image_status, stretch="always", height=480)

    # 结果和建议框架
    result_frame = tk.Frame(content, bg='white', height=240)
    content.add(result_frame, stretch="never")

    # 识别结果标签和文本区域
    global output_text_area
    label_recognition_results = tk.Label(result_frame, text="识别结果:", font=("Arial", 14), bg='white')
    label_recognition_results.pack(anchor='nw', padx=20, pady=10)
    output_text_area = tk.Text(result_frame, height=4, width=50)
    output_text_area.pack(padx=20, pady=0)

    # 防治建议标签和文本区域
    global advice_text_area
    label_prevention_advice = tk.Label(result_frame, text="防治建议:", font=("Arial", 14), bg='white')
    label_prevention_advice.pack(anchor='nw', padx=20, pady=10)
    advice_text_area = tk.Text(result_frame, height=4, width=50)
    advice_text_area.pack(padx=20, pady=0)

    root.mainloop()

# 选择图片并显示
def select_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename()
    if selected_image_path:
        img = Image.open(selected_image_path)
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label_image_status.config(image=photo)
        label_image_status.image = photo

# 模型推理
def predict():
    if selected_image_path:
        img = Image.open(selected_image_path)
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = net(img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            advice = advice_dict.get(predicted_class, "暂无建议")
            output_text_area.delete(1.0, tk.END)
            output_text_area.insert(tk.END, predicted_class)
            advice_text_area.delete(1.0, tk.END)
            advice_text_area.insert(tk.END, advice)

# 运行函数以创建UI
create_ui()
