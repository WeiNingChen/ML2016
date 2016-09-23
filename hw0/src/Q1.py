import sys

if __name__ == '__main__':
  f = file(sys.argv[2])
  l = int(sys.argv[1])
  v=[]
  for line in f:
    v.append(float(line.split( )[l]))
  
  output = open('ans1.txt', 'w')
  output.write(",".join(str(i) for i in sorted(v)))
    
