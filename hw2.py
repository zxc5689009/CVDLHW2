from PyQt5.QtCore import QByteArray, QBuffer, QIODevice
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from sklearn.decomposition import PCA 
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import sys
import cv2
import torch
import numpy as np
from hw2UI import Ui_MainWindow
from torchvision import models
from tqdm import tqdm
from torchsummary import summary
import torch.nn as nn
import io
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import os
labels=['cat','dog']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
global model_VGG19
model_VGG19 = models.vgg19_bn(pretrained=False, num_classes=10).to(device)
# 修改第一层以接受单通道输入
model_VGG19.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

# 确保模型在正确的设备上
model_VGG19 = model_VGG19.to(device)

# 加载权重
#model.load_state_dict(torch.load('model_epoch_78.pth', map_location=device))
try:
    model_VGG19.load_state_dict(torch.load('VGG19_model_epoch_68.pth', map_location=device))
    print("Model_VGG19 weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
global model_ResNet50
model_ResNet50=models.resnet50(pretrained=True)
num_features = model_ResNet50.fc.in_features
model_ResNet50.fc = nn.Sequential(
    nn.Linear(num_features, 1),  # 2048个输入特征到1个输出
    nn.Sigmoid()
)
# 将模型移至合适的设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_ResNet50.to(device)
try:
    model_ResNet50.load_state_dict(torch.load('cat_dog_resnet50_epoch_95.pth', map_location=device))
    print("Model_ResNet50 weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
def preprocess(image):
    # 转换为三通道图像
    image = image.convert('RGB')

    # 调整图像大小并进行标准化
    transform = Compose([
        Resize((32, 32)),  # VGG19模型输入为224x224
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # 添加批量维度

# 预测函数
def predict(model, image):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 确保不计算梯度
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)  # 转换为概率分布
        _, predicted = torch.max(probabilities, 1)  # 获取预测结果

        # 绘制概率图表
        plt.figure(figsize=(10, 4))
        plt.bar(range(10), probabilities[0].cpu().numpy())  # 假设有10个类别
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.title('Prediction Probabilities')
        plt.show()

        return predicted.item()
def Load_Video():
    global Video_name
    Video_name, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'All Files (*);;Text Files (*.txt)')
    print(Video_name)
def Load_Image():
    global Image_name
    Image_name, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'All Files (*);;Text Files (*.txt)')
    print(Image_name)

def Background_Subtraction():
    video = cv2.VideoCapture(Video_name)
    subtractor = cv2.createBackgroundSubtractorKNN(history=500,dist2Threshold=400.0,detectShadows=True)
    while True:
        ret, frame = video.read()
        if not ret:
         break

        # 高斯模糊
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # 獲取背景遮罩
        mask = subtractor.apply(blurred_frame)

        # 提取前景物體
        foreground = cv2.bitwise_and(frame, frame, mask=mask)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = cv2.hconcat([frame, mask_colored, foreground])#合併影像

        # 顯示結果
        cv2.imshow('Combined', combined)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# 釋放資源
    video.release()
    cv2.destroyAllWindows()

def Preprocessing():
    video = cv2.VideoCapture(Video_name)
    ret, frame = video.read()
    if not ret:
        print("無法讀取視頻")
        video.release()
        exit()
    # 將彩色幀轉換為灰度圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 檢測特徵點
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1, qualityLevel=0.3, minDistance=7,blockSize=7)
    corners = np.int0(corners)

    # 假設 corners[0] 是最接近玩偶鼻子底部的點
    # 繪製紅色十字標記
    x, y = corners[0].ravel()
    cv2.line(frame, (x - 10, y), (x + 10, y), (0, 0, 255), 2)
    cv2.line(frame, (x, y - 10), (x, y + 10), (0, 0, 255), 2)

    # 顯示標記後的影像
    cv2.imshow('Feature Point', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    video.release()

def Video_tracking():
    video = cv2.VideoCapture(Video_name)
    ret, old_frame = video.read()
    if not ret:
        print("無法讀取視頻")
        video.release()
        exit()

    # 將第一幀轉換為灰度圖像
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # 使用 goodFeaturesToTrack 檢測初始追蹤點
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)

    # 為光流法創建一個參數字典
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 創建一個用於繪製軌跡的空圖像
    track_img = np.zeros_like(old_frame)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用光流法計算新的追蹤點位置
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 選擇好的追蹤點
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 繪製追蹤點軌跡
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            track_img = cv2.line(track_img, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 255), -1)
            cv2.line(frame, (int(a) - 10, int(b)), (int(a) + 10, int(b)), (0, 0, 255), 2)
            cv2.line(frame, (int(a), int(b) - 10), (int(a), int(b) + 10), (0, 0, 255), 2)
        img = cv2.add(frame, track_img)

        # 顯示結果
        cv2.imshow('Tracking', img)

        # 更新上一幀和追蹤點
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 釋放資源並關閉所有視窗
    video.release()
    cv2.destroyAllWindows()    

def calculate_mse(imageA, imageB):
    # 將圖像數據轉換成一維數組
    flatA = imageA.flatten()
    flatB = imageB.flatten()
    # 計算均方誤差
    mse = np.mean((flatA - flatB) ** 2)
    return mse
def Dimension_Reduction():
    image = cv2.imread(Image_name)
# 轉換為灰度圖像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 正規化灰度圖像到 [0,1]
    normalized_gray_image = gray_image / 255.0

    # 獲取圖像的寬度和高度
    h, w = normalized_gray_image.shape

    # 初始化 PCA，並開始逐步增加主成分的數量
    max_components = min(w, h)
    n_components = 1
    mse = float('inf')
    while mse > 3.0:
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(normalized_gray_image)
        reconstructed_data = pca.inverse_transform(transformed_data)
        # 對 reconstructed_data 進行歸一化
        normalized_reconstructed_data = (reconstructed_data - np.min(reconstructed_data)) / (np.max(reconstructed_data) - np.min(reconstructed_data))

        reconstructed_image = normalized_reconstructed_data * 255.0
        reconstructed_image = np.clip(reconstructed_image, 0, 255)

        # 計算 MSE
        mse = mean_squared_error(gray_image.ravel(), reconstructed_image.ravel())
        # 如果 MSE 大於 3.0，增加成分數量
        if mse > 3.0:
            n_components += 1
    # for n_components in range(1, max_components+1):
    #     pca = PCA(n_components=n_components)
    #     pca.fit(normalized_gray_image)
    #     transformed_img = pca.transform(normalized_gray_image)
    #     reconstructed_img = pca.inverse_transform(transformed_img)*255.0
    #     mse = np.mean((gray_image - reconstructed_img) ** 2)
    #     if mse <= 3.0:
    #         print(f'Breaking loop at n_components = {n_components}')
    #         break

    if reconstructed_image is not None:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(normalized_gray_image, cmap='gray')
        plt.title('Original Gray Scale Image')
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title(f'Reconstructed Image with {n_components} Components')
        plt.show()
    else:
        print("Unable to reconstruct the image with the given conditions.")

def Show_Model_Structure():
    summary(model_VGG19, input_size=(3, 224, 224))
def Show_Accuracy_and_Loss():
    Imagetmp = cv2.imread('training_validation_loss_accuracy.jpg')
    cv2.imshow('lossandaccuracy',Imagetmp)
def Predict():
    pixmap = UI.paint_widget.pix
    image = pixmap.toImage()

    # 创建缓冲区以获取图像数据
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, 'PNG')

    # 从缓冲区读取图像并转换为PIL图像
    pil_image = Image.open(io.BytesIO(buffer.data()))
    pil_image = pil_image.crop((0,0,200,200))
    #pil_image.save('temp.png')
    # 预处理图像
    image_tensor = preprocess(pil_image).to(device)

    # 使用模型进行预测
    predicted_number = predict(model_VGG19, image_tensor)
    UI.myLabel.setText(str(predicted_number))
    print(f'Predicted number: {predicted_number}')
def Reset():
    UI.paint_widget.pix.fill(QtGui.QColor('black'))
    UI.paint_widget.update()  # 更新涂鸦板以显示更改
def Load_Image5():
    global Image_name5
    Image_name5, _ = QFileDialog.getOpenFileName(None, 'Select a file', '', 'All Files (*);;Text Files (*.txt)')
    print(Image_name5)    
    tmp = f'<img src = "{Image_name5}" width = "128" height = "128"/>'
    UI.ResNet50Text.setHtml(tmp)
def Show_Images():
    dog_folder_path = './Dataset_Cvdl_Hw2_Q5/resnet_dataset/inference_dataset/Dog'
    cat_folder_path = './Dataset_Cvdl_Hw2_Q5/resnet_dataset/inference_dataset/Cat'

    # 从文件夹中获取所有图片文件名
    dog_images = [img for img in os.listdir(dog_folder_path) if img.endswith(".jpg")]
    cat_images = [img for img in os.listdir(cat_folder_path) if img.endswith(".jpg")]

    # 随机选择一张狗和一张猫的图片
    dog_image_path = os.path.join(dog_folder_path, random.choice(dog_images))
    cat_image_path = os.path.join(cat_folder_path, random.choice(cat_images))

    # 加载并显示图片
    plt.figure(figsize=(10, 5))
    
    # 显示狗的图片
    plt.subplot(1, 2, 1)
    dog_img = Image.open(dog_image_path)
    plt.imshow(dog_img)
    plt.title("Dog")
    plt.axis('off')  # 关闭坐标轴

    # 显示猫的图片
    plt.subplot(1, 2, 2)
    cat_img = Image.open(cat_image_path)
    plt.imshow(cat_img)
    plt.title("Cat")
    plt.axis('off')  # 关闭坐标轴

    plt.show()
def Show_Model_Structure_5():
    summary(model_ResNet50, input_size=(3, 224, 224))
def Show_Comparison():
# 定义数据
    categories = ["Without Random Erasing", "With Random Erasing"]
    values = [0.9467, 0.9711]

    # 创建条形图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=['blue', 'green'])

    # 在条形上添加文本
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), ha='center', va='bottom')

    # 设置图表标题和轴标签
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')

    # 显示图表
    plt.show()
def Inference():
    transform = Compose([
        Resize((224, 224)),  # VGG19模型输入为224x224
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(Image_name5)
    image = transform(image).unsqueeze(0)  # 添加一个批次维度
    image = image.to(device)  # 将数据移至正确的设备
    model_ResNet50.eval()  # 设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        outputs = model_ResNet50(image)
        prediction = torch.sigmoid(outputs)  # 如果最后一层是Linear，需要手动应用Sigmoid
        print(float(prediction))
        predicted_class = prediction.round()  # 将输出转换为类别标签
        print(predicted_class)
    UI.myLabel2.setText(labels[int(predicted_class)])
    print(f'Predicted class: {labels[int(predicted_class)]}')  # 输出预测类别
def main():
    global UI
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    UI = Ui_MainWindow()
    UI.setupUi(MainWindow)
    UI.Load_Video.clicked.connect(Load_Video)
    UI.Load_Image.clicked.connect(Load_Image)
    UI.Background_Subtraction_button.clicked.connect(Background_Subtraction)
    UI.Preprocessing.clicked.connect(Preprocessing)
    UI.Video_tracking.clicked.connect(Video_tracking)
    UI.Dimension_Reduction.clicked.connect(Dimension_Reduction)
    UI.Show_Model_Structure.clicked.connect(Show_Model_Structure)
    UI.Show_Accuracy_and_Loss.clicked.connect(Show_Accuracy_and_Loss)
    UI.Predict.clicked.connect(Predict)
    UI.Reset.clicked.connect(Reset)
    UI.Load_Image5.clicked.connect(Load_Image5)
    UI.Show_Images.clicked.connect(Show_Images)
    UI.Show_Comparison.clicked.connect(Show_Comparison)
    UI.Show_Model_Structure5.clicked.connect(Show_Model_Structure_5)
    UI.Inference.clicked.connect(Inference)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()