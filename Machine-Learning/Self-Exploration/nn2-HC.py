#!/usr/bin/env python -tt

'''
Implementing a neural network from scratch.

'''

import random
import math

def totalError(targetO, actualO):
  Eo1 = 0.5 * ((targetO[0] - actualO[0]) ** 2)
  Eo2 = 0.5 * ((targetO[1] - actualO[1]) ** 2)
  return [Eo1,Eo2]

def Sigmoid(z):
  return 1 / (1 + math.exp(-1 * z))

def FeedForward(X, W, b, arch):
  
  # Architecture info
  print 'Architecture structure : ', arch

  # FOR LAYER 1 -> LAYER 2
  activL = arch[2:]
  netN = []
  for i in range(len(activL)):
    h = [] 
    for j in range(activL[i]):
      h.append(0)
    netN.append(h)  
  print 'Net Nodes : ', netN
  print 'Inputs : ', X
  print 'Weigths : ', W
  print 'Activation layers : ', activL 
  a = []
  a.append(X)
  for i in range(len(activL[:-1])):
    for j in range(activL[i]):
      for k in range(activL[i]):
        x = j
        netN[i + 1][j] += a[i][k] * W[i][x] 
      netN[i+1][j] += a[i][j] * W[i][j + 1]  
  # Add the bias
  neth1 += b[0]
  neth2 += b[0]
  #print neth1
  #print neth2

  outh1 = Sigmoid(neth1) # Activations!
  #print outh1
  outh2 = Sigmoid(neth2) # Activations!
  #print outh2

  # FOR LAYER 2 -> LAYER 3
  neto1 = 0
  neto2 = 0

  for i in range(2):
    neto1 += outh1 * W[i + 4]
    neto2 += outh2 * W[i + 6]  
  # Add the bias
  neto1 += b[1]
  neto2 += b[1]
  #print neto1
  #print neto2

  outo1 = Sigmoid(neto1) # Activations!
  #print outo1
  outo2 = Sigmoid(neto2) # Activations!
  #print outo2

  allParams = [neth1, neth2, outh1, outh2, neto1, neto2, outo1, outo2]

  return allParams

def BackPropogation(X, y, W, b, ffParams, E):
  
  # Saving original weights
  W_tmp = W[:]
  # Learning-Rate
  eeta = 0.5 

  # Layer 3 -> Layer 2
  # Partial derivative : del(Etotal)/del(w5) = (del(Etotal)/del(outo1)) * (del(outo1)/del(neto1)) * (del(neto1)/del(w5))
  # del(Etotal)/del(outo1)
  delEtotal_delouto1 = ffParams[-2] - y[0]
  # del(out1)/del(neto1)
  delouto1_delneto1 = ffParams[-2] * (1 - ffParams[-2])
  # del(neto1)/del(w5)
  delneto1_delw5 = ffParams[2]
  # del_o1
  del_o1 = delEtotal_delouto1 * delouto1_delneto1
  # del(Etotal)/del(w5)
  delEtotal_delw5 = del_o1 * delneto1_delw5
  #print delEtotal_delw5
  W[4] = W[4] - eeta * delEtotal_delw5
  #print W[4]

  # Partial derivative : del(Etotal)/del(w6) = (del(Etotal)/del(outo1)) * (del(out1)/del(neto1)) * (del(neto1)/del(w6))
  delneto1_delw6 = ffParams[3]
  delEtotal_delw6 = del_o1 * delneto1_delw6
  #print delEtotal_delw6
  W[5] = W[5] - eeta * delEtotal_delw6
  #print W[5]

  # Partial derivative : del(Etotal)/del(w7) = (del(Etotal)/del(outo2)) * (del(out2)/del(neto2)) * (del(neto2)/del(w7))
  # del(Etotal)/del(outo2)
  delEtotal_delouto2 = ffParams[-1] - y[1]
  # del(out2)/del(neto2)
  delouto2_delneto2 = ffParams[-1] * (1 - ffParams[-1])
  # del(neto2)/del(w7)
  delneto2_delw7 = ffParams[2]
  # del_o2
  del_o2 = delEtotal_delouto2 * delouto2_delneto2
  delEtotal_delw7 = del_o2 * delneto2_delw7
  #print delEtotal_delw7
  W[6] = W[6] - eeta * delEtotal_delw7
  #print W[6]

  # Partial derivative : del(Etotal)/del(w8) = (del(Etotal)/del(outo2)) * (del(out2)/del(neto2)) * (del(neto2)/del(w8))
  delneto2_delw7 = ffParams[3]
  delEtotal_delw7 = del_o2 * delneto2_delw7
  #print delEtotal_delw6
  W[7] = W[7] - eeta * delEtotal_delw7
  #print W[7]

  # Layer 2 -> Layer 1

  # Partial derivative : del(Etotal)/del(w1) = (del(Etotal)/del(outh1)) * (del(outh1)/del(neth1)) * (del(neth1)/del(w1))

  # del(Etotal)/del(outh1) = (del(Eo1)/del(outh1)) + (del(Eo2)/del(outh1))
  # (del(Eo1)/del(outh1)) = (del(Eo1)/del(neto1)) * (del(neto1)/del(outh1))
  # del(Eo1)/del(neto1) = (del(Eo1)/del(outo1)) * (del(outo1)/del(neto1))
  # (del(Eo1)/del(outo1)) = del(Etotal)/del(outo1) &  del(neto1)/del(outh1) = w5. Therefore :- 
  delEo1_delouth1 = delEtotal_delouto1 * delouto1_delneto1 * W_tmp[4]
  #print delEo1_delouth1

  delEo2_delouth1 = delEtotal_delouto2 * delouto2_delneto2 * W_tmp[6]
  #print delEo2_delouth1

  # del(Etotal)/del(outh1)
  delEtotal_delouth1 = delEo1_delouth1 + delEo2_delouth1
  #print delEtotal_delouth1

  # del(outh1)/del(neth1) 
  delouth1_delneth1 = ffParams[2] * (1 - ffParams[2])
  #print delouth1_delneth1

  # del(neth1)/del(w1)
  delneth1_delw1 = X[0]
  #print delneth1_delw1

  # Finally, del(Etotal)/del(w1)
  delEtotal_delw1 = delEtotal_delouth1 * delouth1_delneth1 * delneth1_delw1
  #print delEtotal_delw1

  W[0] = W[0] - eeta * delEtotal_delw1
  #print W[0]

  # Partial derivative : del(Etotal)/del(w2) = (del(Etotal)/del(outh1)) * (del(outh1)/del(neth1)) * (del(neth1)/del(w2))
  # Go similarly to the previous partial derivative steps
  
  delneth1_delw2 = X[1]
  
  # Finally, del(Etotal)/del(w2)
  delEtotal_delw2 = delEtotal_delouth1 * delouth1_delneth1 * delneth1_delw2
  #print delEtotal_delw2
  W[1] = W[1] - eeta * delEtotal_delw2
  print W[1]

  # Partial derivative : del(Etotal)/del(w3) = (del(Etotal)/del(outh2)) * (del(outh2)/del(neth2)) * (del(neth2)/del(w3))
  # del(Etotal)/del(outh2) = (del(Eo1)/del(outh2)) + (del(Eo2)/del(outh2))
  # (del(Eo1)/del(outh2)) = (del(Eo1)/del(neto1)) * (del(neto1)/del(outh2))
  # del(Eo1)/del(neto1) = (del(Eo1)/del(outo1)) * (del(outo1)/del(neto1))
  # (del(Eo1)/del(outo1)) = del(Etotal)/del(outo1) &  del(neto1)/del(outh2) = w6. Therefore :- 
  delEo1_delouth2 = delEtotal_delouto1 * delouto1_delneto1 * W_tmp[5]
  delEo2_delouth2 = delEtotal_delouto2 * delouto2_delneto2 * W_tmp[7]
  delEtotal_delouth2 = delEo1_delouth2 + delEo2_delouth2
  delouth2_delneth2 =  ffParams[3] * (1 - ffParams[3])
  delneth2_delw3 = X[0]
  delEtotal_delw3 = delEtotal_delouth2 * delouth2_delneth2 * delneth2_delw3
  W[2] = W[2] - eeta * delEtotal_delw3
  print W[2]

  # Partial derivative : del(Etotal)/del(w3) = (del(Etotal)/del(outh2)) * (del(outh2)/del(neth2)) * (del(neth2)/del(w4))
  # Go similarly to the previous partial derivative steps
  delneth2_delw4 = X[1]
  delEtotal_delw4 = delEtotal_delouth2 * delouth2_delneth2 * delneth2_delw4
  W[3] = W[3] - eeta * delEtotal_delw4
  print W[3]  

def main():

# 3 layers
# Layer 1 : Input - 2 nodes
# Layer 2 : Hidden - 2 nodes
# Layer 3 : Output - 2 nodes
  Architecture =  [2, 3, 3, 2]
  Architecture = [len(Architecture)] + Architecture
  X = [0.05, 0.1]
  y = [0.01, 0.99]
  W = []
  bias = []
  '''
  For testing purposes
  '''
  # b = [0.35, 0.60]
  # W = [0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55]
  
  # Preparing the architecture!
  # Weights!
  random.seed()
  Layers = Architecture[1:]
  for i in range(len(Layers) - 1):
    L = [] 
    for j in range(Layers[i] * Layers[i + 1]):
      L.append(round(random.uniform(0, 1), 2))
    W.append(L)  
  #print W
  # Bias units!
  for i in range(len(Architecture[1:-1])):
    bias.append(round(random.uniform(0, 1), 2))
  #print bias  

  allParams = FeedForward(X, W, bias, Architecture)

  E = totalError(y, allParams[-2:])
  Etotal = E[0] + E[1]

  BackPropogation(X, y, W, b, allParams, E)



if __name__ == '__main__':
 main()  