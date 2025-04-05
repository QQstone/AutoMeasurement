from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from flags import *
from ui.func_item import *
from ui.step import Step


class FuncListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parentDialog = parent
        self.setFocusPolicy(Qt.NoFocus)

        self.setFixedHeight(64)
        self.setFlow(QListView.LeftToRight)  # 设置列表方向
        self.setViewMode(QListView.IconMode)  # 设置列表模式
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关掉滑动条
        self.setAcceptDrops(False)

        for item in func_items:
            self.addItem(item())
        self.itemClicked.connect(self.add_used_function)

    def add_used_function(self):
        func_item = self.currentItem()
        if type(func_item) in func_items:
            use_item = type(func_item)()
            func_name = getattr(type(func_item), '__name__').replace('Item', '')
            step = Step(func_name)

            # 获取当前选中的step索引
            current_index = self.parentDialog.stepListWidget.getCurrentIndex()
            
            # 如果current_index == -1，则追加到末尾
            if current_index == -1:
                self.parentDialog.mainWindow.appendDetectFlowItem(step)
                self.parentDialog.stepListWidget.addItem(use_item)
            else:
                # 在当前选中的step之后插入
                self.parentDialog.mainWindow.insertDetectFlowItem(current_index + 1, step)
                self.parentDialog.stepListWidget.insertItem(current_index + 1, use_item)
                
            self.parentDialog.mainWindow.update_image()

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        self.setCurrentRow(-1)  # 取消选中状态


