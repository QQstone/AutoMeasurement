# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'detectModeEdit.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

from ui.func_list import FuncListWidget
from ui.func_steps import FuncStepsWidget


class Ui_ModeDialog(object):
    def __init__(self):
        super(Ui_ModeDialog, self).__init__()

        self.funcListWidget = FuncListWidget(self)
        self.stepListWidget = FuncStepsWidget(self)
        self.argumentWidget = ArgumentWidget(self)
        self.dock_func = QDockWidget(self)
        self

    def StackedWidget(self):
        # 平滑处理
        self.addWidget()
        # 形态学

        # 阈值处理

        # 轮廓检测