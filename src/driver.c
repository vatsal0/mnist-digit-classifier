#include <stdio.h>
#include <stdlib.h>

#include "mnist-digits.h"
#include "neural-network.h"
#include "libbmp/libbmp.h"

int main(void) {
  Image_Array *array = malloc(sizeof(Image_Array));
  Image *image;
  size_t layer_sizes[] = {300};
  int i;
  int correct = 0;

  Neural_Network *network = malloc(sizeof(Neural_Network));

  read_set(array, "./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte");
  image = array->images[5];
  render_image(image, "out/image_00005.bmp");

  int input_size = image->num_rows * image->num_cols;
  initialize_network(network, input_size, 10, layer_sizes, sizeof(layer_sizes)/sizeof(*layer_sizes));

  load_weights(network, "nn_300.weights");

  for (i = 0; i < array->num_images; i++)
    correct += predict(network, array->images[i]) == array->images[i]->label;
  printf("%.03f%% accuracy\n", 100.0 * correct / array->num_images);

  for (i = 0; i < 2; i++) {
    train(network, array, 1500);
    save_weights(network, "nn_300.weights");
  }
}