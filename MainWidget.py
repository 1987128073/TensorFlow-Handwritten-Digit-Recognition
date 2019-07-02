'''
Created on 2018年8月8日

@author: Freedom
'''
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter,\
    QComboBox, QLabel, QSpinBox, QFileDialog
from PaintBoard import PaintBoard
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QBrush
import tensorflow as tf
from PIL import ImageGrab, Image
from PyQt5.QtWidgets import QApplication
import PIL.ImageOps
import cv2
import numpy as np
import os

class MainWidget(QWidget):


    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        self.b_list = list()
        self.num = 0
        self.pix = []
        self.result = 0
        super().__init__(Parent)
        
        self.__InitData() #先初始化数据，再初始化界面
        self.__InitView()
        self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        
        self.initUI()
        print(self.width,self.height)
    
    def __InitData(self):
        '''
                  初始化成员变量
        '''
        self.__paintBoard = PaintBoard(self)
        
    def __InitView(self):
        '''
                  初始化界面
        '''
        self.setFixedSize(480,300)
        self.setWindowTitle("手写数字识别")     
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("123.jpg")))  
        self.setPalette(palette)

        main_layout = QHBoxLayout(self) #新建一个水平布局作为本窗体的主布局
        main_layout.setSpacing(10) #设置主布局内边距以及控件间距为10px
        
        main_layout.addWidget(self.__paintBoard) #在主界面左侧放置画板
        
        sub_layout = QVBoxLayout() #新建垂直子布局用于放置按键
        sub_layout.setContentsMargins(15, 15, 10, 10) #设置此子布局和内部控件的间距

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setFixedSize(116,25)
        self.__btn_Clear.setStyleSheet("QPushButton{color:rgb(101,153,26)}"
                                 "QPushButton{background-color:	#EED5D2}"
                                 "QPushButton:hover{color:red}"
                                 "QPushButton{border-radius:6px}"
                                 "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}")
        self.__btn_Clear.setParent(self) #设置父对象为本界面
        self.__btn_Clear.clicked.connect(self.Clear) #将按键按下信号与画板清空函数相关联
        sub_layout.addWidget(self.__btn_Clear)
        
        self.__btn_Save = QPushButton("图片识别")
        self.__btn_Save.setFixedSize(116,25)
        self.__btn_Save.setStyleSheet("QPushButton{color:rgb(101,153,26)}"
                                 "QPushButton{background-color:#FFE1FF}"
                                 "QPushButton:hover{color:red}"
                                 "QPushButton{border-radius:6px}"
                                 "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.btn_recognize_on_clicked)
        sub_layout.addWidget(self.__btn_Save)
        
        self.__btn_Add = QPushButton("添加图片并识别")
        self.__btn_Add.setFixedSize(116,25)
        self.__btn_Add.setStyleSheet("QPushButton{color:rgb(101,153,26)}"
                                 "QPushButton{background-color:#FFE1EE}"
                                 "QPushButton:hover{color:red}"
                                 "QPushButton{border-radius:6px}"
                                 "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}")
        self.__btn_Add.setParent(self)
        self.__btn_Add.clicked.connect(self.add_pic)
        sub_layout.addWidget(self.__btn_Add)
        
        self.__btn_Quit = QPushButton("程序退出")
        self.__btn_Quit.setFixedSize(116,25)
        self.__btn_Quit.setStyleSheet("QPushButton{color:rgb(101,153,26)}"
                                 "QPushButton{background-color:#CFCFCF}"
                                 "QPushButton:hover{color:red}"
                                 "QPushButton{border-radius:6px}"
                                 "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}")
        self.__btn_Quit.setParent(self) #设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)
        
#        splitter = QSplitter(self) #占位符
#        sub_layout.addWidget(splitter)
        
        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("识别结果：")
        self.__label_penThickness.setFont(QFont("Roman times",8,QFont.Bold))
        self.__label_penThickness.setFixedHeight(18)
        sub_layout.addWidget(self.__label_penThickness)
        
        self.label_result = QLabel(self)
        self.label_result.setStyleSheet("QLabel{border:1px solid #EED5D2;border-radius:4px;color:#FFE4C4;font-size:25px;font-weight:bold;text-align: center;}")
        self.label_result.setFixedSize(140,28)
        sub_layout.addWidget(self.label_result)
        
        main_layout.addLayout(sub_layout) #将子布局加入主布局
        
    def btn_recognize_on_clicked(self):

        if self.__paintBoard.pos_xy == []:
            self.label_result.setText('')  # 显示识别结果
            return self.update()
            
        bbox = ( self.width//2-226,self.height//2-155, self.width//2+53,self.height//2+124)
        print(bbox)
        im = ImageGrab.grab(bbox)    # 截屏，手写数字部分
        
        im = im.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素
        
        recognize_result,flag = self.recognize_img(im)  # 调用识别函数
        if flag == 1:
            return
        cnn_acc,knn_acc,tree_acc = self.acc()
        if cnn_acc<knn_acc:
            cnn_acc,knn_acc=knn_acc,cnn_acc
        if recognize_result != 0:
            #im.save('./0/mnist_0_%d.png' %self.num)
            self.num += 1
        self.label_result.setText(str(recognize_result))  # 显示识别结果
        self.label_cnn_acc.setText(str("cnn:%.6f"%cnn_acc)) 
        self.label_knn_acc.setText(str("knn:%.6f"%knn_acc))
        self.label_tree_acc.setText(str("DTree:%.6f"%tree_acc))
        self.update()
    
    def recognize_img(self, img):  # 手写体识别函数
        myimage = img.convert('L')  # 转换成灰度图
        
        tv = list(myimage.getdata())  # 获取图片像素值
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]  # 转换像素范围到[0 1], 0是纯白 1是纯黑
        flag = 0
        if self.pix == tva:
            flag = 1
            return self.result,flag
        self.pix = tva
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver  

        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.import_meta_graph("./model_version4/model.ckpt.meta")  # 载入模型结构
            saver.restore(sess, "./model_version4/model.ckpt")  # 载入模型参数

            graph = tf.get_default_graph()  # 加载计算图
            x = graph.get_tensor_by_name("x:0")  # 从模型中读取占位符变量
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            y_conv = graph.get_tensor_by_name("y_conv:0")  # 关键的一句  从模型中读取占位符变量
            prediction = tf.argmax(y_conv, 1)
            
            predint = prediction.eval(feed_dict={x: [tva], keep_prob: 1.0}, session=sess)  # feed_dict输入数据给placeholder占位符
            self.result = predint[0]
            print(predint[0])
        return predint[0],flag
        
    
    def Clear(self):
        #清空画板
        self.__paintBoard._InitData()
        self.__paintBoard.pos_xy = []
        self.label_result.setText('')
        self.label_cnn_acc.setText('')
        self.label_knn_acc.setText('')
        self.label_tree_acc.setText('')
        self.update()
        
    def Quit(self):
        self.close()
        
    def add_pic(self):
        self.Clear()
        fname = QFileDialog.getOpenFileName(self, "Open File", "./", "Image files (*.jpg *.png)")
        if fname[0]:
            #判断路径非空
            f = fname[0]   #创建文件对象
            print(f)
            #jpg = QtGui.QPixmap(f).scaled(self.label.width(), self.label.height())
            self.__paintBoard._InitData(f)
            self.pic_deal_v2(f)
             

        
    def initUI(self):
 
        self.desktop = QApplication.desktop()
 
        #获取显示器分辨率大小
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
       
    def pic_deal(self, path):
        a_list = []
        path_list = [int(p.split('.')[0]) for p in os.listdir('pic_list')]
        path_list.sort()

        for a in path_list:
            a = str(a)+'.png'
            print(a)
            img = Image.open('pic_list/%s' % a)
            
            img = img.resize((28, 28), Image.ANTIALIAS)
            
            #img = PIL.ImageOps.invert(img)
            # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
            Img = img.convert('L')
            #反转颜色
            Img = PIL.ImageOps.invert(Img)
            threshold = 48
         
            table = []
            for i in range(256):
                if i > threshold:
                    table.append(0)
                else:
                    table.append(1)
            Img = Img.point(table, '1')

            recognize_result,flag = self.recognize_img(Img)
            a_list.append(str(recognize_result))
        print(a_list) 
        self.label_result.setText(''.join(a_list))  # 显示识别结果
        self.update()
    
    def pic_deal_v2(self,path):

        # 1.Canny边缘检测
        img = cv2.imread(path, 0)
        # 2.先阈值，后边缘检测
        # 阈值分割（使用到了番外篇讲到的Otsu自动阈值）
        _, img_thre = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#        _, img_thre = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#        img = cv2.dilate(img_thre, (8,8), iterations=18)
#        _, img_thre = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #cv2.imshow('canny', np.hstack((img_thre,)))
        
        cv2.waitKey(0)
        
        # 4、分割字符
        white = []  # 记录每一列的白色像素总和
        black = []  # ..........黑色.......
        height = img_thre.shape[0]
        width = img_thre.shape[1]
        white_max = 0
        black_max = 0
        # 计算每一列的黑白色像素总和
        for i in range(width):
            s = 0  # 这一列白色总数
            t = 0  # 这一列黑色总数
            for j in range(height):
                if img_thre[j][i] == 255:
                    s += 1
                if img_thre[j][i] == 0:
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)
            white.append(s)
            black.append(t)
            print(s)
            print(t)
         
        arg = False  # False表示白底黑字；True表示黑底白字
        if black_max > white_max:
            arg = True
         
        
         
        n = 1
        start = 1
        end = 2
        if os.path.exists('pic_list'):
            import shutil
            shutil.rmtree('pic_list')
            os.mkdir('pic_list')
                        
        else:
            os.mkdir('pic_list')
        num = 0 
        while n < width-2:
            num += 1
            n += 1
            if (white[n] if arg else black[n]) > (0.01 * white_max if arg else 0.01 * black_max):
                # 上面这些判断用来辨别是白底黑字还是黑底白字
                # 0.01这个参数请多调整，对应上面的0.99
                start = n
                end = self.find_end(start,width,arg,white,black_max,white_max,black)
                n = end
                if end-start > 5:
                    cj = img_thre[1:height, start:end]
        #            cv2.imshow('caijian', cj)
                    h = cj.shape[0]
                    w = cj.shape[1]
                    if h > w:
                        a = cv2.copyMakeBorder(cj,0,0,h//2-w//2,h//2-w//2,cv2.BORDER_CONSTANT,value=[255,255,255])
                    else:
                        a = cv2.copyMakeBorder(cj,w//2-h//2,w//2-h//2,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
                    #cv2.imshow('白色填充图', a)
                    
#                    cj = img_thre[height//3:height-height//4, start:end]
#                    h = cj.shape[0]
#                    w = cj.shape[1]
#                    if h > w:
#                        a = cv2.copyMakeBorder(cj,0,0,h//2-w//2,h//2-w//2,cv2.BORDER_CONSTANT,value=[255,255,255])
#                    else:
#                        a = cv2.copyMakeBorder(cj,w/2-h//2,w//2-h//2,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
                    cv2.imwrite('pic_list/%d.png' % num,a)
                    
                    cv2.waitKey(0)
        self.pic_deal(path)
    
    # 分割图像
    def find_end(self, start_, width, arg, white, black_max, white_max,black):
        end_ = start_+1
        for m in range(start_+1, width-1):
            if (black[m] if arg else white[m]) > (0.99 * black_max if arg else 0.99 * white_max):  # 0.99这个参数请多调整，对应下面的0.05
                end_ = m
                break
        return end_
    