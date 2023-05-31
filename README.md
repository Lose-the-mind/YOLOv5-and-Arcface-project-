
# YOLOv5-and-Arcface-project-
使用YOLOv5算法实现人数统计，使用Arcface算法实现人脸识别，结合上述两种算法在人数统计的目标检测基础上实现人脸识别
一、YOLOv5算法实现人数统计
1、使用新的数据集本地自主训练得到.pt文件
2、使用这个pt模型文件在原代码上运行detect.py文件可以实现人数统计功能
二、Arcface算法实现对人脸特征的提取
1、将需要识别的人脸图片放入pic文件夹中（尽量是脸部图片）
2、在arc_face文件中有本算法相关流程代码
3、使用arcface算法实现人脸识别，CASIA-WebFace （我没有自己训练，直接用arcface源码提供的权重，训练模型权重：yoloV5的预训练权重包、训练好的侦测人脸的权重包、人脸识别的权重包resnet110 下载地址：链接：https://pan.baidu.com/s/1YzgQcFVl4Rd6skN5q7mw-w 提取码：kusi
4、将下载的权重文件放在文件夹weights中即可

三、运行dect_tect.py文件
运行本文件可以实现对pic文件夹内图片特征的提取，并且根据yolov5算法对人的检测结果，使用余弦距离判断这个所检测出的标框与pic文件夹中所提取特征之间的距离来判断相似度。
具体流程可以查看我的CSDN博客（还没有写，等我啥时候写了再更新
