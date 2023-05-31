import argparse
import time
from pathlib import Path
from arc_face import *
import cv2
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
from torch.nn import DataParallel

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import sys

sys.path.append("./yoloV5_face")


def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size, expected_size
    scale = min(eh / ih, ew / iw)  # 最大边缩放至416得比例
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)  # 等比例缩放，使得有一边416
    top = (eh - nh) // 2  # 上部分填充的高度
    bottom = eh - nh - top  # 下部分填充的高度
    left = (ew - nw) // 2  # 左部分填充的距离
    right = ew - nw - left  # 右部分填充的距离
    # 边界填充
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


def cosin_metric(x1, x2):
    # 计算余弦距离
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# 这是一个Python函数，它使用OpenCV库加载图像文件，并将其转换为适合深度学习模型处理的格式
def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    # image = cv2_letterbox_image(image,128)
    image = cv2.resize(image, (128, 128))
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


# 这是一个Python函数，该函数使用给定的深度学习模型获取目录中所有图像文件的特征，并将结果保存在字典中
def get_featuresdict(model, dir):
    list = os.listdir(dir)
    person_dict = {}
    for i, each in enumerate(list):
        image = load_image(f"pic/{each}")
        data = torch.from_numpy(image)
        data = data.to(torch.device("cuda"))
        output = model(data)  # 获取特征
        output = output.data.cpu().numpy()
        # print(output.shape)

        # 获取不重复图片 并分组
        fe_1 = output[0]
        fe_2 = output[1]
        # print("this",cnt)
        # print(fe_1.shape,fe_2.shape)
        feature = np.hstack((fe_1, fe_2))
        # print(feature.shape)

        person_dict[each] = feature
    return person_dict


def detect(save_img=False):
    # source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    source, weights, Areface_weights, view_img, save_txt, imgsz = \
        opt.source, opt.weights, opt.Areface_weights, opt.view_img, opt.save_txt, opt.img_size  # 加载配置信息
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    dir = "pic"
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # 使用attempt_load函数加载指定路径下的FP32模型权重文件，并将其存储在变量model中。
    # 通过model.stride.max()方法获取模型的步幅（stride），并将其存储在变量stride中。
    # 使用check_img_size函数检查输入图像的尺寸是否符合模型要求，如果不符合，则进行调整。
    # 如果half参数为True，则使用model.half()方法将模型转换为FP16格式。
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # 创建一个ArcFace模型（resnet_face18）并使用DataParallel函数将其移动到GPU上。
    arcface_model = resnet_face18(False)

    # 使用torch.load函数从指定路径加载ArcFace模型参数。
    arcface_model = DataParallel(arcface_model)
    # load_model(model, opt.test_model_path)

    # 使用torch.load函数从指定路径加载ArcFace模型参数。
    # 将ArcFace模型移到GPU上进行推理操作，并设置为评估模式。
    # 创建一个AntiSpoofPredict对象（该对象未在代码片段中定义，可能是自定义的类）。
    arcface_model.load_state_dict(torch.load(Areface_weights), strict=False)
    arcface_model.to(device).eval()
    # pred_model = AntiSpoofPredict(0)
    if half:
        model.half()  # to FP16
    # 调用get_featuresdict函数以获取指定目录下所有图像文件的特征向量，并将结果存储在features变量中。
    features = get_featuresdict(arcface_model, dir)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference     推理过程
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    # 数据及处理
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                temp=len(det)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # 针对每个通过yoloV5检测得到的人脸，从原始图像中提取出相应的面部图像
                    face_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    # 将面部图像调整为大小为(128, 128)的正方形，并将其转换为灰度图像。
                    face_img = cv2.resize(face_img, (128, 128))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                    # 使用np.dstack函数将原始图像与水平翻转后的图像堆叠在一起。
                    # 使用image.transpose函数交换图像轴的顺序，使其变为(C, H, W)。
                    # 添加额外的维度来创建四维张量，即(B, C, H, W)，其中B为batch size，为1。
                    # 将图像数据类型转换为float32，并将所有像素值减去127.5进行归一化处理。
                    # 将图像数据转换为torch张量，并将其移到GPU上进行模型推理。
                    # 使用arcface_model模型（resnet_face18）对输入图像进行推理操作，以获取图像的特征向量。
                    # 将特征向量从GPU中转移到CPU中，并转换为numpy数组。
                    # 将两个特征向量连接成一个特征向量。
                    # 针对已知的每个人，将其存储在文件夹（dir）中，并遍历该文件夹中的所有文件，
                    # 计算当前面部特征和文件夹中每个人脸特征之间的欧几里得距离。
                    # 找到最小距离（即最相似的人）并将其标签存储在变量label中。
                    face_img = np.dstack((face_img, np.fliplr(face_img)))

                    face_img = face_img.transpose((2, 0, 1))
                    face_img = face_img[:, np.newaxis, :, :]

                    face_img = face_img.astype(np.float32, copy=False)
                    face_img -= 127.5
                    face_img /= 127.5

                    face_data = torch.from_numpy(face_img)
                    face_data = face_data.to(device)

                    _output = arcface_model(face_data)  # 获取特征
                    _output = _output.data.cpu().numpy()
                    fe_1 = _output[0]
                    fe_2 = _output[1]
                    _feature = np.hstack((fe_1, fe_2))
                    label = "none"
                    list = os.listdir(dir)
                    max_f = 0
                    t = 0
                    for i, each in enumerate(list):
                        t = cosin_metric(features[each], _feature)
                        # print(each, t)
                        if t > max_f:
                            max_f = t
                            max_n = each
                        # print(max_n,max_f)
                        print("**********************")
                        print(max_f)
                        if (max_f > 0.46):
                            label = max_n[:-4]
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #
                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{names[int(cls)]} {conf:.2f}'
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                text = f"person: {temp}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (0, 0, 255)
                thickness = 2
                # 在图像上绘制文本
                im0 = cv2.putText(im0, text, (35, 35), font, font_scale, color, thickness)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--Areface_weights', type=str, default=r'weights/resnet18_110.pth', help='arcface.pth path')
    # parser.add_argument('--source', type=str, default='rtsp://172.16.25.245/13', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='huge05.jpg', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
