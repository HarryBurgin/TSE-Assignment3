import math

class MLmaths(object):
    """maths used in machine learning"""

    @staticmethod
    def sigmoid(net):
        calc = 1/(1 + (math.exp(-net)))
        return calc

