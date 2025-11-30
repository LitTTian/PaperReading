# tool
import cv2
import os
from pathlib import Path
import matplotlib.colors as mcolors
from PIL import Image
import numpy as np
from ultralytics import YOLO
camp = mcolors.CSS4_COLORS
color_list = list(camp.values())

def draw_boxes(img, results, class_names, return_cv2_img=False,
               toSave=False, savePath=None):
    # nc = len(class_names)  # 类别数量
    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        pass
    elif isinstance(img, str):
        if savePath is None:
            savePath = img.rsplit('.', 1)[0] + '_detected.' + img.rsplit('.', 1)[1]
        img = cv2.imread(img)
    for r in results:
        boxes = r.boxes  # 检测框（xyxy、置信度、类别）
        for box in boxes:
            cls = int(box.cls)  # 类别索引
            conf = box.conf[0]  # 置信度
            xyxy = box.xyxy[0].cpu().numpy().astype(int)  # 边框坐标（xyxy格式）
            label = f"{class_names[cls]} {conf:.2f}"
            color = [int(c * 255) for c in mcolors.to_rgb(color_list[cls])]
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if toSave:
        if savePath is None:
            # 原来的文件名上添加'_detected'
            savePath = 'detected_image.jpg'
        # print(savePath)
        cv2.imwrite(savePath, img)
    if not return_cv2_img:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def save_label(
    model,
    img_path,
    save_label_dir=None,
    save_label_path=None,
    conf_threshold=0.5,
    img_size=640,
    suffix=None
):
    if save_label_path is None:
        if save_label_dir is None:
            save_label_dir = "./labels"
        Path(save_label_dir).mkdir(parents=True, exist_ok=True)
        label_file = os.path.splitext(os.path.basename(img_path))[0]
        if suffix is not None:
            label_file += suffix  # 添加label后缀
        label_file += '.txt'
        label_path = os.path.join(save_label_dir, label_file)
    else:
        label_path = save_label_path
    
    results = model.predict(
        source=img_path,
        imgsz=img_size,
        conf=conf_threshold,
        save=False,       # 不保存预测可视化图
        verbose=False     # 关闭详细日志
    )
    
    with open(label_path, 'w', encoding='utf-8') as f:
        for r in results:
            img_h, img_w = r.orig_shape  # 原始图片高、宽
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"已保存标签：{label_path}")

def save_labels_with_img_dir(
    model=None,
    img_dir=None,
    save_label_dir=None,
    model_path=None,
    conf_threshold=0.5,
    img_size=640,
    suffix=None
):
    if model is None:
        model = YOLO(model_path)
    
    if img_dir is None:
        img_dir = "./images"
    if save_label_dir is None:
        save_label_dir = "./labels"
    Path(save_label_dir).mkdir(parents=True, exist_ok=True)
    
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for img_file in os.listdir(img_dir):
        if not any(img_file.lower().endswith(ext) for ext in img_extensions):
            continue
        img_path = os.path.join(img_dir, img_file)
        save_label(
            model=model,
            img_path=img_path,
            save_label_dir=save_label_dir,
            conf_threshold=conf_threshold,
            img_size=img_size,
            suffix=suffix
        )