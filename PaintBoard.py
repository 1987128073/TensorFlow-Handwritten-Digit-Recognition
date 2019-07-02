'''
Created on 2018年8月9日

@author: Freedom
'''
from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen,\
    QColor, QSize
from PyQt5.QtCore import Qt

class PaintBoard(QWidget):


    def __init__(self, Parent=None, fpath=None):
        '''
        Constructor
        '''
        
        super().__init__(Parent)
        self.pos_xy = []  #保存鼠标移动过的点
        self._InitData(fpath) #先初始化数据，再初始化界面
        self.__InitView()
        
    def __InitView(self):
        self.setFixedSize(self.__size)
        
    def _InitData(self,fpath=None):
        
        self.__size = QSize(280,280)
        if fpath is None:
            self.__board = QPixmap(self.__size) #新建QPixmap作为画板，宽350px,高250px
            self.__board.fill(Qt.white) #用白色填充画板
        else:
            self.__board = QPixmap(fpath).scaled(280,280)
        
      
    def paintEvent(self, event):
        self.__painter = QPainter()
        self.__painter.begin(self)
        self.__painter.drawPixmap(0,0,self.__board)
        pen = QPen(Qt.black, 30, Qt.SolidLine)
        self.__painter.setPen(pen)
    
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp
    
                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue
    
                self.__painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
            self.__painter.end()
            
    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
        '''
        #中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()
        
    def Clear(self):
        pass
    
    