from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class TableWidget(QTableWidget):
    def __init__(self, parent=None):
        super(TableWidget, self).__init__(parent=parent)
        self.update_enable = False
        self.parentDialog = parent
        self.setShowGrid(False)  # 显示网格
        self.setAlternatingRowColors(True)  # 隔行显示颜色
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().sectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().sectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setStretchLastSection(True)
        self.setFocusPolicy(Qt.NoFocus)

    def signal_connect(self):
        for spinbox in self.findChildren(QSpinBox):
            spinbox.valueChanged.connect(self.update_item)
            # Connect slider if it exists
            slider = self.findChild(QSlider, name=f"{spinbox.objectName()}_slider")
            if slider:
                spinbox.valueChanged.connect(lambda v, s=slider: s.setValue(v))
                slider.valueChanged.connect(lambda v, sb=spinbox: sb.setValue(v))
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            doublespinbox.valueChanged.connect(self.update_item)
        for combox in self.findChildren(QComboBox):
            combox.currentIndexChanged.connect(self.update_item)
        for checkbox in self.findChildren(QCheckBox):
            checkbox.stateChanged.connect(self.update_item)

    def update_item(self):
        if not self.update_enable:
            return
        param = self.get_params()
        self.parentDialog.stepListWidget.currentItem().update_params(param)
        self.parentDialog.mainWindow.updateDetectFlowItem(self.parentDialog.stepListWidget.getCurrentIndex(), param)
        self.parentDialog.mainWindow.update_image()

    def set_params(self, param):
        self.update_enable = False
        self.update_params(param)
        self.update_enable = True

    def update_params(self, param=None):
        for key in param.keys():
            box = self.findChild(QWidget, name=key)
            if isinstance(box, QSpinBox) or isinstance(box, QDoubleSpinBox):
                box.setValue(int(param[key]))
                # Update slider if it exists
                slider = self.findChild(QSlider, name=f"{key}_slider")
                if slider:
                    slider.setValue(int(param[key]))
            elif isinstance(box, QComboBox):
                box.setCurrentIndex(int(param[key]))
            elif isinstance(box, QCheckBox):
                box.setChecked(int(param[key]))

    def get_params(self):
        param = {}
        for spinbox in self.findChildren(QSpinBox):
            param[spinbox.objectName()] = spinbox.value()
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            param[doublespinbox.objectName()] = doublespinbox.value()
        for combox in self.findChildren(QComboBox):
            param[combox.objectName()] = combox.currentIndex()
        for combox in self.findChildren(QCheckBox):
            param[combox.objectName()] = combox.isChecked()
        return param

    def add_spinbox_with_slider(self, name, min_val, max_val, step, row, col, label):
        # Create container widget
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create spinbox
        spinbox = QSpinBox()
        spinbox.setObjectName(name)
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setSingleStep(step)
        
        # Create slider
        slider = QSlider(Qt.Horizontal)
        slider.setObjectName(f"{name}_slider")
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setSingleStep(step)
        
        # Add widgets to layout
        layout.addWidget(spinbox)
        layout.addWidget(slider)
        
        # Set cell widget
        self.setItem(row, col, QTableWidgetItem(label))
        self.setCellWidget(row, col + 1, container)
        
        return spinbox, slider


class GrayingTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(GrayingTableWidget, self).__init__(parent=parent)


class FilterTabledWidget(TableWidget):
    def __init__(self, parent=None):
        super(FilterTabledWidget, self).__init__(parent=parent)
        self.horizontalHeader().setVisible(True)
        self.kind_comBox = QComboBox()
        self.kind_comBox.addItems(['均值滤波', '高斯滤波', '中值滤波'])
        self.kind_comBox.setObjectName('kind')

        self.ksize_spinBox = QSpinBox()
        self.ksize_spinBox.setObjectName('ksize')
        self.ksize_spinBox.setMinimum(1)
        self.ksize_spinBox.setSingleStep(2)

        self.setColumnCount(2)
        self.setRowCount(2)
        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.kind_comBox)
        self.setItem(1, 0, QTableWidgetItem('核大小'))
        self.setCellWidget(1, 1, self.ksize_spinBox)

        self.signal_connect()


class MorphTabledWidget(TableWidget):
    def __init__(self, parent=None):
        super(MorphTabledWidget, self).__init__(parent=parent)

        self.op_comBox = QComboBox()
        self.op_comBox.addItems(['腐蚀操作', '膨胀操作', '开操作', '闭操作', '梯度操作', '顶帽操作', '黑帽操作'])
        self.op_comBox.setObjectName('op')

        self.ksize_spinBox = QSpinBox()
        self.ksize_spinBox.setMinimum(1)
        self.ksize_spinBox.setSingleStep(2)
        self.ksize_spinBox.setObjectName('ksize')

        self.kshape_comBox = QComboBox()
        self.kshape_comBox.addItems(['方形', '十字形', '椭圆形'])
        self.kshape_comBox.setObjectName('kshape')

        self.setColumnCount(2)
        self.setRowCount(3)

        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.op_comBox)
        self.setItem(1, 0, QTableWidgetItem('核大小'))
        self.setCellWidget(1, 1, self.ksize_spinBox)
        self.setItem(2, 0, QTableWidgetItem('核形状'))
        self.setCellWidget(2, 1, self.kshape_comBox)
        self.signal_connect()


class GradTabledWidget(TableWidget):
    def __init__(self, parent=None):
        super(GradTabledWidget, self).__init__(parent=parent)

        self.kind_comBox = QComboBox()
        self.kind_comBox.addItems(['Sobel算子', 'Scharr算子', 'Laplacian算子'])
        self.kind_comBox.setObjectName('kind')

        self.ksize_spinBox = QSpinBox()
        self.ksize_spinBox.setMinimum(1)
        self.ksize_spinBox.setSingleStep(2)
        self.ksize_spinBox.setObjectName('ksize')

        self.dx_spinBox, self.dx_slider = self.add_spinbox_with_slider('dx', 0, 1, 1, 2, 0, 'x方向')
        self.dy_spinBox, self.dy_slider = self.add_spinbox_with_slider('dy', 0, 1, 1, 3, 0, 'y方向')

        self.setColumnCount(2)
        self.setRowCount(4)

        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.kind_comBox)
        self.setItem(1, 0, QTableWidgetItem('核大小'))
        self.setCellWidget(1, 1, self.ksize_spinBox)
        self.signal_connect()


class ThresholdTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(ThresholdTableWidget, self).__init__(parent=parent)

        

        self.method_comBox = QComboBox()
        self.method_comBox.addItems(['二进制阈值化', '反二进制阈值化', '截断阈值化', '阈值化为0', '反阈值化为0', '大津算法'])
        self.method_comBox.setObjectName('method')

        self.setColumnCount(2)
        self.setRowCount(3)

        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.method_comBox)
        self.thresh_spinBox, self.thresh_slider = self.add_spinbox_with_slider('thresh', 0, 255, 1, 1, 0, '阈值')
        self.maxval_spinBox, self.maxval_slider = self.add_spinbox_with_slider('maxval', 0, 255, 1, 2, 0, '最大值')
        self.signal_connect()


class EdgeTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(EdgeTableWidget, self).__init__(parent=parent)


        self.setColumnCount(2)
        self.setRowCount(2)
        self.thresh1_spinBox, self.thresh1_slider = self.add_spinbox_with_slider('thresh1', 0, 255, 1, 0, 0, '阈值1')
        self.thresh2_spinBox, self.thresh2_slider = self.add_spinbox_with_slider('thresh2', 0, 255, 1, 1, 0, '阈值2')
        self.signal_connect()


class ContourTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(ContourTableWidget, self).__init__(parent=parent)

        self.bbox_comBox = QComboBox()
        self.bbox_comBox.addItems(['正常轮廓', '外接矩形', '最小外接矩形', '最小外接圆'])
        self.bbox_comBox.setObjectName('bbox')

        self.mode_comBox = QComboBox()
        self.mode_comBox.addItems(['外轮廓', '轮廓列表', '外轮廓与内孔', '轮廓等级树'])
        self.mode_comBox.setObjectName('mode')

        self.method_comBox = QComboBox()
        self.method_comBox.addItems(['无近似', '简易近似'])
        self.method_comBox.setObjectName('method')

        self.setColumnCount(2)
        self.setRowCount(3)

        self.setItem(0, 0, QTableWidgetItem('轮廓模式'))
        self.setCellWidget(0, 1, self.mode_comBox)
        self.setItem(1, 0, QTableWidgetItem('轮廓近似'))
        self.setCellWidget(1, 1, self.method_comBox)
        self.setItem(2, 0, QTableWidgetItem('边界模式'))
        self.setCellWidget(2, 1, self.bbox_comBox)
        self.signal_connect()


class EqualizeTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(EqualizeTableWidget, self).__init__(parent=parent)
        self.red_checkBox = QCheckBox()
        self.red_checkBox.setObjectName('red')
        self.red_checkBox.setTristate(False)
        self.blue_checkBox = QCheckBox()
        self.blue_checkBox.setObjectName('blue')
        self.blue_checkBox.setTristate(False)
        self.green_checkBox = QCheckBox()
        self.green_checkBox.setObjectName('green')
        self.green_checkBox.setTristate(False)

        self.setColumnCount(2)
        self.setRowCount(3)

        self.setItem(0, 0, QTableWidgetItem('R通道'))
        self.setCellWidget(0, 1, self.red_checkBox)
        self.setItem(1, 0, QTableWidgetItem('G通道'))
        self.setCellWidget(1, 1, self.green_checkBox)
        self.setItem(2, 0, QTableWidgetItem('B通道'))
        self.setCellWidget(2, 1, self.blue_checkBox)
        self.signal_connect()


class HoughLineTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(HoughLineTableWidget, self).__init__(parent=parent)


        self.setColumnCount(2)
        self.setRowCount(3)
        self.thresh_spinBox, self.thresh_slider = self.add_spinbox_with_slider('thresh', 0, 255, 1, 0, 0, '交点阈值')
        self.min_length_spinBox, self.min_length_slider = self.add_spinbox_with_slider('min_length', 0, 255, 1, 1, 0, '最小长度')
        self.max_gap_spinbox, self.max_gap_slider = self.add_spinbox_with_slider('max_gap', 0, 255, 1, 2, 0, '最大间距')
        self.signal_connect()


class LightTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(LightTableWidget, self).__init__(parent=parent)

        self.alpha_spinBox = QDoubleSpinBox()
        self.alpha_spinBox.setMinimum(0)
        self.alpha_spinBox.setMaximum(3)
        self.alpha_spinBox.setSingleStep(0.1)
        self.alpha_spinBox.setObjectName('alpha')


        self.setColumnCount(2)
        self.setRowCount(2)

        self.setItem(0, 0, QTableWidgetItem('alpha'))
        self.setCellWidget(0, 1, self.alpha_spinBox)
        self.beta_spinbox, self.beta_slider = self.add_spinbox_with_slider('beta', 0, 255, 1, 1, 0, 'beta')
        self.signal_connect()


class GammaITabelWidget(TableWidget):
    def __init__(self, parent=None):
        super(GammaITabelWidget, self).__init__(parent=parent)
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0)
        self.gamma_spinbox.setSingleStep(0.1)
        self.gamma_spinbox.setObjectName('gamma')

        self.setColumnCount(2)
        self.setRowCount(1)

        self.setItem(0, 0, QTableWidgetItem('gamma'))
        self.setCellWidget(0, 1, self.gamma_spinbox)
        self.signal_connect()


class FrequencyFilterTabledWidget(TableWidget):
    def __init__(self, parent=None):
        super(FrequencyFilterTabledWidget, self).__init__(parent=parent)
        
        # 滤波器类型选择
        self.filter_type_comBox = QComboBox()
        self.filter_type_comBox.addItems(['低通滤波', '高通滤波', '带通滤波'])
        self.filter_type_comBox.setObjectName('filter_type')

        self.setColumnCount(2)
        self.setRowCount(4)

        self.setItem(0, 0, QTableWidgetItem('滤波器类型'))
        self.setCellWidget(0, 1, self.filter_type_comBox)
        
        # 截止频率/半径
        self.radius_spinBox, self.radius_slider = self.add_spinbox_with_slider('radius', 1, 100, 1, 1, 0, '截止频率')
        
        # 带通滤波的带宽
        self.width_spinBox, self.width_slider = self.add_spinbox_with_slider('width', 1, 50, 1, 2, 0, '带宽')
        
        # Butterworth滤波器阶数
        self.order_spinBox, self.order_slider = self.add_spinbox_with_slider('order', 1, 10, 1, 3, 0, '滤波器阶数')
        
        self.signal_connect()


tables = [
    GrayingTableWidget,
    FilterTabledWidget,
    EqualizeTableWidget,
    MorphTabledWidget,
    GradTabledWidget,
    ThresholdTableWidget,
    EdgeTableWidget,
    ContourTableWidget,
    HoughLineTableWidget,
    LightTableWidget,
    GammaITabelWidget,
    FrequencyFilterTabledWidget
]


class StackedWidget(QStackedWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        for table in tables:
            self.addWidget(table(parent=parent))
        self.setMinimumWidth(200)
