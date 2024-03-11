# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainMeasurement.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import os

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDir
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QFileDialog, QDialog, QMessageBox
import configparser

from ui.mode_editor import Ui_ModeEditor
from ui.step import Step


class Ui_MainWindow(object):
    modeEditor = None
    #__detectFlow = []

    def __init__(self):
        self.src_img = None
        self.cur_img = None
        self.background = None
        self.scene = None
        # 初始化模式编辑
        self.configPath = 'config.ini'
        self.__detectFlow = []
        self.loadDetectFlow()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(875, 635)
        # root widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        centralSizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # flex layout
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.setAlignment(QtCore.Qt.AlignRight)

        self.graphicsView = QtWidgets.QGraphicsView()
        self.graphicsView.setMinimumSize(591, 581)
        #self.graphicsView.setGeometry(QtCore.QRect(7, 10, 591, 581))
        self.horizontalLayout.addWidget(self.graphicsView, stretch=1)
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.graphicsView.show()

        # sub-horizon layout for group measure
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.processWidget = QtWidgets.QWidget()
        # self.widget.setGeometry(QtCore.QRect(611, 21, 143, 151))
        self.processWidget.setObjectName("widget")
        self.verticalLayout.addWidget(self.processWidget)
        self.groupMeasure = QtWidgets.QGroupBox()
        self.groupMeasure.setFixedSize(220, -1)
        self.groupMeasure.setObjectName("groupMeasure")
        self.verticalLayout.addWidget(self.groupMeasure)


        self.formLayout = QtWidgets.QFormLayout(self.processWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.btn_import = QtWidgets.QToolButton(self.processWidget)
        self.btn_import.setObjectName("btn_import")
        self.btn_import.clicked.connect(self.importImg)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.btn_import)

        self.btn_reset = QtWidgets.QToolButton(self.processWidget)
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.clicked.connect(self.reset_img)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.btn_reset)

        self.btn_usemode = QtWidgets.QToolButton(self.processWidget)
        self.btn_usemode.setObjectName("btn_usemode")
        self.btn_usemode.clicked.connect(self.usemode)
        self.btn_usemode.setEnabled(False)

        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.btn_usemode)
        self.btn_editmode = QtWidgets.QToolButton(self.processWidget)
        self.btn_editmode.setObjectName("btn_editmode")
        self.btn_editmode.clicked.connect(self.editmode)

        self.txt_filePathLabel = QtWidgets.QLabel(self.processWidget)
        self.txt_filePathLabel.setObjectName("txt_filePathLabel")
        self.txt_filePathLabel.setText("配置文件：")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.txt_filePathLabel)
        self.txt_filePath = QtWidgets.QLabel(self.processWidget)
        self.txt_filePath.setObjectName("txt_filePath")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.txt_filePath)

        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.btn_editmode)
        self.btn_createFile = QtWidgets.QToolButton(self.processWidget)
        self.btn_createFile.setObjectName("btn_preprocess")
        self.btn_createFile.clicked.connect(self.create_new_config_file)
        self.btn_createFile.setEnabled(False)

        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.btn_createFile)
        self.btn_loadfile = QtWidgets.QToolButton(self.processWidget)
        self.btn_loadfile.setObjectName("btn_processconfig")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.btn_loadfile)
        self.btn_loadfile.clicked.connect(self.load_config_file)
        self.btn_loadfile.setEnabled(False)

        right_container = QtWidgets.QWidget()
        right_container.setLayout(self.verticalLayout)
        self.horizontalLayout.addWidget(right_container)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 875, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "鹿茸菇图像预处理配置软件"))
        self.groupMeasure.setTitle(_translate("MainWindow", "测量结果"))
        self.btn_import.setText(_translate("MainWindow", "导入图像"))
        self.btn_reset.setText(_translate("MainWindow", "重置图像"))
        self.btn_usemode.setText(_translate("MainWindow", "应用模式"))
        self.btn_editmode.setText(_translate("MainWindow", "编辑模式"))
        self.btn_createFile.setText(_translate("MainWindow", "新建配置"))
        self.btn_loadfile.setText(_translate("MainWindow", "加载配置"))

    def importImg(self):
        fname = QFileDialog.getOpenFileName(self, '导入图像', './', 'Image files (*.jpg *.gif *.png *.jpeg)')

        if fname[0]:
            self.src_img = self.imread(fname[0])
            self.graphicsShow(self.src_img)
            self.btn_import.setEnabled(False)
            # self.btn_background.setEnabled(True)
            self.btn_createFile.setEnabled(True)
            self.btn_loadfile.setEnabled(True)
            self.btn_usemode.setEnabled(True)

    def reset_img(self):
        self.src_img = None
        self.cur_img = None
        self.scene.clear()
        self.btn_import.setEnabled(True)
        self.btn_createFile.setEnabled(False)
        self.btn_usemode.setEnabled(False)
        self.btn_loadfile.setEnabled(False)

    def usemode(self):
        if self.src_img is None:
            return
        self.update_image()


    def editmode(self):
        self.modeEditor = Ui_ModeEditor(self)
        self.modeEditor.setWindowTitle("编辑模式")
        retVal = self.modeEditor.exec()
        if retVal == QDialog.Accepted:
            # write into config.ini
            self.saveDetectFlow()
        else:
            # reload from config.ini
            self.loadDetectFlow()

    def create_new_config_file(self):
        # 选择目录
        filePath, filetype = QFileDialog.getSaveFileName(self, "保存配置文件", QDir.homePath(), "Config Files (*.ini)")
        if filePath:
            # 在目录中创建空白文件
            with open(filePath, 'w') as f:
                f.write('')
            self.txt_filePath.setText(os.path.basename(filePath))
            self.configPath = filePath

    def load_config_file(self):
        # 选择文件
        filePath, filetype = QFileDialog.getOpenFileName(self, '选择配置文件', '', 'Config Files (*.ini)')
        if filePath:
            # 将文件路径显示在文本框中
            self.txt_filePath.setText(os.path.basename(filePath))
            self.configPath = filePath
            self.loadDetectFlow()

    def imread(self, imgPath):
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def graphicsShow(self, img):
        y, x = img.shape[:-1]
        ratio = float(y / x)
        frame = None
        if ratio > 1:
            fit_height = self.graphicsView.height()
            fit_width = int(fit_height / ratio)
            img = cv2.resize(img, (fit_width, fit_height), interpolation=cv2.INTER_AREA)
            frame = QImage(img, fit_width, fit_height, fit_width * 3, QImage.Format_RGB888)
        else:
            fit_width = self.graphicsView.height()
            fit_height = int(fit_width * ratio)
            img = cv2.resize(img, (fit_width, fit_height), interpolation=cv2.INTER_AREA)
            frame = QImage(img, fit_width, fit_height, fit_width*3, QImage.Format_RGB888)
        self.scene.clear()
        self.pix = QPixmap.fromImage(frame)
        self.scene.addPixmap(self.pix)

    def process_image(self):
        img = self.src_img.copy()
        for step in self.__detectFlow:
            # for key, value in step.getArguments().items():
            # setattr(step.getWidget(), key, value)
            img = step.getWidget()(img)
        return img

    def update_image(self):
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        self.graphicsShow(img)

    def loadDetectFlow(self):
        self.__detectFlow = []
        config = configparser.ConfigParser()
        config.read(self.configPath)
        sections = config.sections()
        for sec in sections:
            funcName = sec.split('-')[1]
            step = Step(funcName)
            for arg in step.getArguments().keys():
                val = config.get(sec, arg)
                if val is not None:
                    step.update_params({arg: val})
            self.appendDetectFlowItem(step)

    def clearDetectFlow(self):
        self.__detectFlow = []

    def getDetectFlow(self):
        return self.__detectFlow

    def saveDetectFlow(self):
        config = configparser.ConfigParser()
        for index, step in enumerate(self.__detectFlow):
            sec_name = str(index) + '-' + step.getFuncName()
            config[sec_name] = step.getArguments()
            # for arg in step.getArguments().keys():
            #     config[sec_name][arg] = step.getArguments()[arg]
        o = open(self.configPath, 'w')
        config.write(o)
        o.close()

    def appendDetectFlowItem(self, item):
        self.__detectFlow.append(item)

    def removeDetectFlowItem(self, index: int):
        del self.__detectFlow[index]

    def updateDetectFlowItem(self, index: int, params: dict):
        self.__detectFlow[index].update_params(params)
