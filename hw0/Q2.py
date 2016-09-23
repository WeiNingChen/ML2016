import Image
import sys

if __name__ == "__main__":
  img = Image.open(sys.argv[1])
  img.transpose(Image.ROTATE_180).save("ans2.png")
