#ifndef MNIST_DIGITS_H
#define MNIST_DIGITS_H

typedef struct {
  unsigned int num_rows;
  unsigned int num_cols;

  unsigned char *pixels;
  unsigned char label;
  unsigned char prediction;
} Image;

typedef struct {
  unsigned int num_images;

  Image **images;
} Image_Array;

void read_set(Image_Array *array, const char *images_filename, const char *labels_filename);

void render_image(Image *image, const char *filename);

#endif