#include <stdio.h>
#include <stdlib.h>

#include "mnist-digits.h"
#include "neural-network.h"
#include "libbmp/libbmp.h"

int main(void) {
  Image_Array *array = malloc(sizeof(Image_Array));
  Image *image;
  size_t layer_sizes[] = {300, 300};
  double *input;
  int i, n;
  double avg_cost;

  Neural_Network *network = malloc(sizeof(Neural_Network));

  read_set(array, "./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte");
  image = array->images[0];
  render_image(image, "out/image_00000.bmp");

  int input_size = image->num_rows * image->num_cols;
  initialize_network(network, input_size, 10, layer_sizes, sizeof(layer_sizes)/sizeof(*layer_sizes));

  input = malloc(sizeof(*input) * input_size);
  
  load_weights(network, "nn_300_300.weights");

  for (i = 0; i < 2; i++)
    train(network, array, 500);
  
  save_weights(network, "nn_300_300.weights");
}