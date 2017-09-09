#!/ur/bin/env python -tt

# Mcculloh-Pitts neuron model for XOR gate

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

def WeightsGen(layers):
  l = []
  for i in range(len(layers) - 1): 
    for j in range(layers[i] * layers[i + 1]):
      l.append(0)
  return l

def main():

  # Threshold
  theta = 1
  x = [[0, 0], [0, 1], [1, 0], [1, 1]]
  arch = [2, 2, 1]
  arch = [len(arch)] + arch
  #print arch
  layers = arch[1:]

  w = [-1, 1, 1, -1, 1, 1]
  
  print WeightsGen(layers)

  for n in range(len(x)):
    a = ActivationLayers(layers)
    m = 0
    a[0] = x[n]
    #print a
    for i in range(len(layers) - 1):
      for j in range(layers[i + 1]):
        for k in range(layers[i]):
          a[i + 1][j] = a[i + 1][j] + (a[i][k] * w[m])
          m += 1
        a[i + 1][j] = ActivationFunc(a[i + 1][j], theta)  
    print a[-1]

if __name__ == '__main__':
  main()