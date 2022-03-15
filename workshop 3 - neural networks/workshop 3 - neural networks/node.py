class node(object):
    """description of class"""
    def __init__(self, _value, _forPointer, _backPointer, _nodeNo):
        self.value = _value ## value is sigmoid(net)
        self.forPointer = _forPointer ## each pointer will be a list containing where it is pointing too
        self.backPointer = _backPointer ## elements in this list is another list that has the key as the node it points to and the value as the weight
        self.nodeNo = _nodeNo
        ## nodeNo only added to keep aligned with the example given in lecture


class inputNode(node):

    def __init__(self, _value, _forPointer, _nodeNo):
        self.value = _value ## value is the input into the system
        self.forPointer = _forPointer 
        self.backPointer = None
        self.nodeNo = _nodeNo


class outputNode(node):

    def __init__(self, _value, _backPointer, _nodeNo):
        self.value = _value ## value is net
        self.forPointer = None 
        self.backPointer = _backPointer
        self.nodeNo = _nodeNo
