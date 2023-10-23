import time
# 颜色类
class BColors(object):
    # 紫色
    HEADER = '\033[95m'
    # 蓝色
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class timeLogger(object):

    def __init__(self):
        self.record_t = None
        self.bcolors = BColors()
        print() # misc
    # 用于输出一个epoch的时间
    def print(self, msg = None):
        if msg is not None:
            if self.record_t is None:
                self.record_t = time.time()
            print(self.bcolors.HEADER + msg + self.bcolors.ENDC, end='')
        else:
            cost = round(time.time() - self.record_t, 4)
            print("....[{}success{}][{}{} sceonds{}]".format(
                self.bcolors.OKCYAN, self.bcolors.ENDC, self.bcolors.OKGREEN, cost, self.bcolors.ENDC))
            self.record_t = None