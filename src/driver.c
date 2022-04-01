#include <stdio.h>
#include <stdlib.h>

#include "mnist-digits.h"
#include "libbmp/libbmp.h"

int main(void) {
  Image_Array *array = malloc(sizeof(Image_Array));
  unsigned char *image;

  read_set(array, "./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte");
  render_image(array->images[5], "out/image_00005.bmp");

}