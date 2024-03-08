import configparser


class Configure():
    __config = None
    # 均值滤波核
    average_core = 3
    # 闭操作核
    close_core = 3
    # 阈值1
    threshold1 = 28
    # 阈值2
    threshold2 = 255
    # 开操作核
    open_core = 5

    axis_radio = 0.5

    def __init__(self):
        self.initConfig()

    def initConfig(self):
        self.__config = configparser.ConfigParser()
        self.__config.read('config.ini')
        self.__config.sections()
        self.average_core = self.__config.get('default', 'average_core')
        self.close_core = self.__config.get('default', 'close_core')
        self.threshold1 = self.__config.get('default', 'threshold1')
        self.threshold2 = self.__config.get('default', 'threshold2')
        self.open_core = self.__config.get('default', 'open_core')

    def save(self):
        self.threshold1 = self.__config.get('default', 'average_core')
        self.threshold1 = self.__config.get('default', 'close_core')
        self.threshold1 = self.__config.get('default', 'threshold1')
        self.threshold1 = self.__config.get('default', 'threshold2')
        self.threshold1 = self.__config.get('default', 'open_core')


        with open('config.ini', 'w') as configfile:
            self.__config.write(configfile)


