from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from ui.func_item import func_items


class FuncStepsWidget(QListWidget):
    __currentIndex = -1
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parentDialog = parent
        self.setDragDropMode(True)
        self.setFocusPolicy(Qt.NoFocus)

        self.setAcceptDrops(True)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setFlow(QListView.TopToBottom)
        self.itemClicked.connect(self.show_attr)
        self.move_item = None

    def contextMenuEvent(self, e):
        # 右键菜单事件
        item = self.itemAt(self.mapFromGlobal(QCursor.pos()))
        if not item: return  # 判断是否是空白区域
        menu = QMenu()
        delete_action = QAction('删除', self)
        delete_action.triggered.connect(lambda: self.delete_item(item))  # 传递额外值
        menu.addAction(delete_action)
        menu.exec(QCursor.pos())

    def delete_item(self, item):
        # 删除操作
        currentIndex = self.row(item)
        self.parentDialog.currentStep = currentIndex
        self.parentDialog.mainWindow.removeDetectFlowItem(currentIndex)
        self.takeItem(currentIndex)
        self.parentDialog.mainWindow.update_image()  # 更新frame
        self.parentDialog.stepArgWidget.close()

    def dropEvent(self, event):
        super().dropEvent(event)
        self.parentDialog.mainwindow.update_image()

    def show_attr(self):
        item = self.itemAt(self.mapFromGlobal(QCursor.pos()))
        if not item: return
        self.__currentIndex = self.row(item)
        if type(item) in func_items:
            index = func_items.index(type(item))  # 获取item对应的table索引
            self.parentDialog.stepArgWidget.setCurrentIndex(index)
            if hasattr(item, "params"):
                self.parentDialog.stepArgWidget.currentWidget().set_params(item.params)  # 更新对应的table
            else:
                params = self.parentDialog.mainWindow.getDetectFlow()[self.__currentIndex].getArguments()
                self.parentDialog.stepArgWidget.currentWidget().set_params(params)
            self.parentDialog.stepArgWidget.show()
        self.parentDialog.currentStep = self.row(item)

    def getCurrentIndex(self):
        return self.__currentIndex
