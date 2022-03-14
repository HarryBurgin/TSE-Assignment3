## code from workshop 3 of AI 
## working neural network only missing the softmax func and proper sqaured error uses

from node import *
from MLmaths import *

print("this is a program for a neural network with hidden layers")

##----------------------------------------------Forward step func
def forStep(nodeList):
    for index, NODE in enumerate(nodeList):
        if NODE.backPointer == None:
            continue
        else:
            net = 0
            for point in NODE.backPointer:
                net += point[0].value * point[1]
            NODE.value = net
            ##print(index, NODE.nodeNo, "Value:", NODE.value) ## test to check if net is being calculated properly

            ## now that we have the net, the sigmoid func must be applied to the net to get the value
            ## should not be applied to O6 and O7
            if index < 6:
                sigNet = MLmaths.sigmoid(net)
                NODE.value = sigNet

            ##print(NODE.value, net) ## checks net value has been calculated properly





def backStep(nodeList):
    squaredError = 0
    learningRate = 0.1
    errorList = [x for x in range(8)] ## creates list for error values
    for index in range(len(nodeList)-1, -1, -1):
        ##print(nodeList[index].nodeNo) ##checks for loop will look at all nodes backwards
        if index == 7:
            t = 0
            errorList[index] = 0 - nodeList[index].value
        elif index == 6:
            t = 1
            errorList[index] = 1 - nodeList[index].value


        else:       ## weight                               error value
            calc = nodeList[index].forPointer[0][1] * errorList[nodeList[index].forPointer[0][0].nodeNo]
            calc += nodeList[index].forPointer[1][1] * errorList[nodeList[index].forPointer[1][0].nodeNo]
                    
            errorList[index] = nodeList[index].value * (1 - nodeList[index].value) * calc

        squaredError += (t - nodeList[index].value)**2
    squaredError = squaredError * (1/2)
    print(squaredError)

    ## changing forwards weights
    for index in range(len(nodeList)):
        if index > 5:
            continue
        else:

            weightChange1 = learningRate * errorList[nodeList[index].forPointer[0][0].nodeNo] * nodeList[index].value
            weightChange2 = learningRate * errorList[nodeList[index].forPointer[1][0].nodeNo] * nodeList[index].value

            ##print(index, "-->", nodeList[index].forPointer[0][0].nodeNo, weightChange1)
            ##print(index, "-->", nodeList[index].forPointer[1][0].nodeNo, weightChange2)
            

            nodeList[index].forPointer[0][1] += weightChange1
            nodeList[index].forPointer[1][1] += weightChange2

    print("\n\n\n")

    ## changing backwards weights
    for index in range(len(nodeList)-1, 3, -1):
        weightChange1 = learningRate * errorList[index] * nodeList[nodeList[index].backPointer[0][0].nodeNo].value
        weightChange2 = learningRate * errorList[index] * nodeList[nodeList[index].backPointer[1][0].nodeNo].value
        weightChange3 = learningRate * errorList[index] * nodeList[nodeList[index].backPointer[2][0].nodeNo].value

        ## Used for testing
        ##print("LR:", learningRate, "Error value:", errorList[nodeList[index].backPointer[0][0].nodeNo], "Node value:", nodeList[nodeList[index].backPointer[0][0].nodeNo].value)
        ##print(index, "-->", nodeList[index].backPointer[0][0].nodeNo, weightChange1)
        ##print(index, "-->", nodeList[index].backPointer[1][0].nodeNo, weightChange2)
        ##print(index, "-->", nodeList[index].backPointer[2][0].nodeNo, weightChange2)
            

        nodeList[index].backPointer[0][1] += weightChange1
        nodeList[index].backPointer[1][1] += weightChange2
        nodeList[index].backPointer[2][1] += weightChange3






## class for all types of nodes in other files
## class for maths in other file

## forward pointers first set to empty as node next in line is not made yet
## Sets up all input nodes
Inode1 = inputNode(0, None, 0)
Inode2 = inputNode(1, None, 1)
Inode3 = inputNode(1, None, 2) ## this is a bias node

## Sets up all hidden nodes
Hnode1 = node(0, None, [[Inode1, 0.5], [Inode2, -0.2], [Inode3, 0.5]], 4)
Hnode2 = node(0, None, [[Inode1, 0.1], [Inode2, 0.2], [Inode3, 0.3]], 5)
Hnode3 = inputNode(1, None, 3) ## this is a bias node

## Sets up all output nodes
Onode1 = outputNode(0, [[Hnode1, 0.7], [Hnode2, 0.6], [Hnode3, 0.2]], 6)
Onode2 = outputNode(0, [[Hnode1, 0.9], [Hnode2, 0.8], [Hnode3, 0.4]], 7)

## Setting up the forward pointers
Inode1.forPointer = [[Hnode1, 0.5], [Hnode2, 0.1]]
Inode2.forPointer = [[Hnode1, -0.2], [Hnode2, 0.2]]
Inode3.forPointer = [[Hnode1, 0.5], [Hnode2, 0.3]]

Hnode1.forPointer = [[Onode1, 0.7],[Onode2, 0.9]]
Hnode2.forPointer = [[Onode1, 0.6],[Onode2, 0.8]]
Hnode3.forPointer = [[Onode1, 0.2],[Onode2, 0.4]]




nodeList = [Inode1, Inode2, Inode3, Hnode3, Hnode1, Hnode2, Onode1, Onode2] ## list of all nodes, could have seperate layer lists





epochs = 10000 ## sets how many times the data set will pass throguh the model
for epoch in range(epochs):
    print("epoch:", epoch)

    forStep(nodeList) ## forward step, calculates output based on weights

    
    for index in range(len(nodeList)):
        if nodeList[index].forPointer == None:
            print(index, "-->", "None")
        else:
            print(index, "-->", nodeList[index].forPointer[0][0].nodeNo, "Weight:", nodeList[index].forPointer[0][1])
            print(index, "-->", nodeList[index].forPointer[1][0].nodeNo, "Weight:", nodeList[index].forPointer[1][1])


    backStep(nodeList)






