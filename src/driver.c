#include <stdio.h>
#include <stdlib.h>

#include "mnist-digits.h"
#include "neural-network.h"
#include "libbmp/libbmp.h"

int main(void) {
  Image_Array *train_array = malloc(sizeof(Image_Array));
  Image_Array *test_array = malloc(sizeof(Image_Array));
  Image *image;
  size_t layer_sizes[] = {300};
  int i;
  int correct = 0;

  Neural_Network *network = malloc(sizeof(Neural_Network));

  read_set(train_array, "./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte");
  read_set(test_array, "./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte");
  image = train_array->images[5];
  render_image(image, "out/image_00005.bmp");

  int input_size = image->num_rows * image->num_cols;
  initialize_network(network, input_size, 10, layer_sizes, sizeof(layer_sizes)/sizeof(*layer_sizes));

  load_weights(network, "nn_300.weights");

  printf("Prediction for image #0: %d \n", predict(network, train_array->images[0]));

  for (i = 0; i < test_array->num_images; i++)
    correct += predict(network, test_array->images[i]) == test_array->images[i]->label;
  printf("%.03f%% accuracy\n", 100.0 * correct / test_array->num_images);

  for (i = 0; i < 2; i++) {
    train(network, train_array, 1500);
    save_weights(network, "nn_300.weights");
  }

  free_set(train_array);
  free_set(test_array);
  free_network(network);
}