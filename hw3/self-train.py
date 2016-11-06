import pickle
import keras
import os

def load_data(filename):
  
  return pickle.load(os.path.expanduser(filename), 'rb')


if __name__ == '__main__':
  data = load_data(filename)
