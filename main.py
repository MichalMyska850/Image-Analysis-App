# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:42:23 2020

@author: Michal
"""

# this script is for starting the application
from ui import AppWidget
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = AppWidget()
    w.show()
    w.showMaximized()

    sys.exit(app.exec_())
