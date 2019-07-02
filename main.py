'''
Created on 2018-08-09 00:00

@author: Freedom
'''

from MainWidget import MainWidget
from PyQt5.QtWidgets import QApplication

import sys

def main():
    app = QApplication(sys.argv) 
    
    mainWidget = MainWidget()
    mainWidget.show()
    
    app.exec_()
    
    
if __name__ == '__main__':
    main()

    