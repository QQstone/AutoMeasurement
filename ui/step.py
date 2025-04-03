from flags import *
from ui.func_item import *

class Step:
    _funcName = ""
    _arguments = {}
    _widget = None

    def __init__(self, funcName):
        # set default arguments
        self._funcName = funcName
        if funcName == 'Graying':
            self._arguments = {}
            self._widget = GrayingItem()
        elif funcName == 'Filter':
            self._arguments = {'kind': MEDIAN_FILTER, 'ksize': 3}
            self._widget = FilterItem()
        elif funcName == 'Equalize':
            self._arguments = {'blue': True, 'green': True, 'red': True}
            self._widget = EqualizeItem()
        elif funcName == 'Morph':
            self._arguments = {'op': ERODE_MORPH_OP, 'kshape': RECT_MORPH_SHAPE, 'ksize': 3}
            self._widget = MorphItem()
        elif funcName == 'Grad':
            self._arguments = {'kind': SOBEL_GRAD, 'ksize': 3, 'dx': 1, 'dy': 0}
            self._widget = GradItem()
        elif funcName == 'Threshold':
            self._arguments = {'method': BINARY_THRESH_METHOD, 'thresh': 127, 'maxval': 255}
            self._widget = ThresholdItem()
        elif funcName == 'Edge':
            self._arguments = {'thresh1': 20, 'thresh2': 100}
            self._widget = EdgeItem()
        elif funcName == 'Contour':
            self._arguments = {'mode': TREE_CONTOUR_MODE, 'method': SIMPLE_CONTOUR_METHOD, 'bbox': NORMAL_CONTOUR}
            self._widget = ContourItem()
        elif funcName == 'HoughLine':
            self._arguments = {'thresh': 10, 'min_length': 20, 'max_gap': 5}
            self._widget = HoughLineItem()
        elif funcName == 'Light':
            self._arguments = {'alpha': 1, 'beta': 0}
            self._widget = LightItem()
        elif funcName == 'Gamma':
            self._arguments = {'gamma': 1}
            self._widget = GammaItem()
        elif funcName == 'FrequencyFilter':
            self._arguments = {
                'filter_type': 0,
                'radius': 30,
                'width': 10,
                'order': 2
            }
            self._widget = FrequencyFilterItem()

    def update_params(self, params: dict):
        for key in params.keys():
            self._arguments[key] = params[key]
        self._widget.update_params(self._arguments)

    def getFuncName(self):
        return self._funcName

    def getArguments(self):
        return self._arguments

    def getWidget(self):
        return self._widget