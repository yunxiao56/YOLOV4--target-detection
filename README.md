# YOLOV4--target-detection
## Linux环境下的YOLOv4目标检测
环境准备：Ubuntu16.04、CUDA10.1、cuDNN 7.65、OpenCV 3.4

## YOLOv4权重文件下载

`yolov4.weight下载，拷贝权重文件到D:\darknet\build\darknet\x64`

## YOLOv4下载和编译
1、下载darknet：git clone https://github.com/AlexeyAB/darknet.git

2、cd darknet(cd 到你的下载的代码的文件夹那）

3、make #直接make则使用的是darknet原始配置（使用CPU）

4、如果使用GPU和OpenCV，则将Makefile文件中的对应项改为1，然后再执行make命令
    修改完成之后在直接make。
    GPU=1
    CUDNN=1
    CUDNN_HALF=1
    OPENCV=1
    OPENMP=1
    LIBSO=1
    DEBUG=1

5、最后，在终端输入：./darknet

6、出现以下输出则说明安装成功：usage: ./darknet <function>
## 进行测试
单张测试

`./darknet detect cfg/yolov4.cfg yolov4.weights data/dog.jpg`

测试多张图片（根据提示输入图片路径）

`./darknet detect cfg/yolov4.cfg yolov4.weights`

##  训练自己的数据集


（1）数据集准备

  使用PASCAL VOC数据集的目录结构（建立文件夹层次为 ./darknet/(yourdataname)/VOCdevkit/ VOC2007）:
  
  
  JPEGImages放所有的训练和测试图片；Annotations放所有的xml标记文件
  
（2）制作yolov4需要的label以及txt

  这个时候只用voc数据集的格式是不满足我们这里需要的格式。首先打开路径下 build/darknet/x64/data/voc/voc_label.py，修改voc_label.py里面的内容。
  
  先把7行的关于2012的去掉，再把第9行改成自己的类别。
    `sets=[('2020', 'train'), ('2020', 'val'), ('2020', 'test')]
    classes = ["target"]
    `
  然后给每个路径前面加个data
  
  ` in_file = open('data/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
  
    out_file = open('data/VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')`
    
   `for year, image_set in sets:
   
    if not os.path.exists('data/VOCdevkit/VOC%s/labels/'%(year)):
    
        os.makedirs('data/VOCdevkit/VOC%s/labels/'%(year))
        
    image_ids = open('data/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    
    for image_id in image_ids:
    
        list_file.write('%sdata//VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        
        convert_annotation(year, image_id)
        
    list_file.close()
    
`

修改完了之后在主目录darknet-master下执行voc_label.py，否则哪些文件会生成在build/darknet/x64/data下面，执行完成后你会看到主目录下的data/目录下会生成几个txt。

主目录darknet-master下的data/VOCdevkit/VOC2007/下面会生成一个label文件夹。

(3)修改配置文件

可以自己新建一个文件夹，将自己所需的所有配置文件都放在这里，我自己感觉那个cfg文件夹里的文件太多了

<a>cfg/目录下复制coco.data，并且重命名为obj.data。然后使用修改下面以下内容

原则上，以下路径可以写法不固定，不论你放到哪，但是你一定要写对，就是一定要是相应文件的路径

`classes= 1  #类别数目

train  =/home/admin/PycharmProjects/yolov4-darknet-master/darknet-master/targetdata/2007_train.txt #训练集路径 

valid  = /home/admin/PycharmProjects/yolov4-darknet-master/darknet-master/targetdata/multi_scat_6.txt # 测试集路径

#valid = data/coco_val_5k.list
names = /home/admin/PycharmProjects/yolov4-darknet-master/darknet-master/targetdata/target.names #自己类别路径

backup = /home/admin/PycharmProjects/yolov4-darknet-master/darknet-master/backup/ #权重保存路径

`

<b>cfg/目录下复制coco.names，并且重命名为obj.names。改成自己类别的名称,我的就是一种目标

`target` 

<c>复制cfg/yolov4-custom.cfg，并且重命名为yolo-obj.cfg，同时修改一下内容
  --1
  
  --2 找到classes,修改类别数
  
  --3 修改三个filters=18， #（类别数+5）*3
  
（4）训练自己的数据集

  `./darknet detector train cfg/obj.data cfg/yolo-obj.cfg yolov4.conv.137`
 
 (5) 预测
 
 `./darknet detector test cfg/obj.data cfg/yolo-obj.cfg backup/yolov4-target_13000.weights`
 
 
 (6)计算map,记住测试的标签txt,要和测试图片放在一起，才能计算
 
 `./darknet detector map cfg/obj.data cfg/yolo-obj.cfg backup/yolov4-target_12000.weights`
 
 
  


