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

/* Reads the images from the given files into the passed struct. */
void read_set(Image_Array *array, const char *images_filename, const char *labels_filename);

/* Renders the passed image's pixels as a bitmap file and saves it to the passed file path. */
void render_image(Image *image, const char *filename);

/* Frees all memory associated with the passed image array, including the pointer to the struct itself */
void free_set(Image_Array *array);

#endif