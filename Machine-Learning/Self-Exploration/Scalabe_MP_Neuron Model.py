#!/ur/bin/env python -tt

# Mcculloh-Pitts neuron model. Scalable

def ArchitectureInit():
  arch = []

  print 'How many layers? ',
  totalL = int(raw_input())
  arch.append(totalL)

  for i in range(totalL):
    if i == 0:
      print 'How many inputs? ',
    elif i == totalL - 1:
      print 'How many outputs? ',
    else:
      print 'How many nodes in hidden layer', i,'? ', 
    tmp = int(raw_input())
    arch.append(tmp)  
  return arch    

def InputsInit(num_inputs):
  inputs = []
  print 'How many training examples? ',
  train_exmp = int(raw_input())
  for i in range(train_exmp):
    print 'Training example', i + 1, ': '
    inp = []
    for j in range(num_inputs):
      print 'Input', j + 1, ': ',
      inp.append(int(raw_input()))
    inputs.append(inp) 
  return inputs   

def ActivationFunc(yin, thresh):
  if yin >= thresh:
    return 1
  else:
    return 0

def ActivationLayers(layers):
  activateMat = []
  for i in range(len(layers)):
    temp = []
    for j in range(layers[i]):
      temp.append(0)
    activateMat.append(temp)
  return activateMat

def WeightsGen(Layers):
  W = []
  for i in range(len(Layers) - 1):
    L = [] 
    for j in range(Layers[i] * Layers[i + 1]):
      print 'Input weight for node', j + 1, 'layer', i + 1, ': ',
      tmp = int(raw_input())
      L.append(tmp)
    W.append(L)
  return W

def ThresholdSet():
  print 'Input the threshold: ',
  theta = int(raw_input())
  return theta

def FeedForward(layers, inputs, weights, threshold):
  # Feed-forward propogation
  nw = []
  for n in range(len(inputs)):
    a = ActivationLayers(layers)
    net = ActivationLayers(layers)   
    a[0] = inputs[n]
    #print a
    for i in range(len(layers) - 1):
      #print layers[i + 1]
      for j in range(layers[i + 1]):
        m = 0
        for k in range(layers[i]):
          net[i + 1][j] = net[i + 1][j] + (net[i][k] * weights[i][j + m])
          #print a[i + 1][j]
          m += layers[i + 1]
        a[i + 1][j] = ActivationFunc(a[i + 1][j], threshold)  
    print 'For input', inputs[n], 'output is', a[-1]
    nw.append(a)
  return nw

def PrintNetwork(layers, inputs, threshold, nodes):
  
  for n in range(len(inputs)):
    print 'Network', n + 1, ':'
    print 'Threshold:', threshold
    for i in range(len(layers)):
      for j in range(layers[i]):
        print nodes[n][i][j],
      if i != len(layers) - 1:
        print '\n||'  
    print '\n'


def main():

  # Setup the architecture
  arch = ArchitectureInit()
  #print arch[1:]

  layers = arch[1:]

  # Setup the inputs
  #x = [[0, 0], [0, 1], [1, 0], [1, 1]]
  x = InputsInit(layers[0])

  # Setup the inputs
  #w = [[1, 1]]  
  w = WeightsGen(layers)
  #print w

  # Threshold
  theta = ThresholdSet()

  # Feed-forward propogation
  results = FeedForward(layers, x, w, theta)
  #print results

  # Display network
  PrintNetwork(layers, x, theta, results)

  # Bias remains!
if __name__ == '__main__':
  main()