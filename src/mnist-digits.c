#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>

#include "mnist-digits.h"
#include "libbmp/libbmp.h"

void read_set(Image_Array *array, const char *images_filename, const char *labels_filename) {
  FILE *images_file;
  FILE *labels_file;

  unsigned int num_images;
  unsigned int num_labels;
  unsigned int num_rows;
  unsigned int num_cols;

  unsigned int n = 0;

  images_file = fopen(images_filename, "rb");
  labels_file = fopen(labels_filename, "rb");

  fseek(images_file, 4, SEEK_SET);
  fseek(labels_file, 4, SEEK_SET);

  fread(&num_images, 4, 1, images_file);
  fread(&num_labels, 4, 1, labels_file);
  if (num_images != num_labels) {
    fprintf(stderr, "Image and label files do not have matching image count.\n");
    exit(-1);
  }

  num_images = ntohl(num_images);
  num_labels = ntohl(num_labels);

  fread(&num_rows, 4, 1, images_file);
  num_rows = ntohl(num_rows);

  fread(&num_cols, 4, 1, images_file);
  num_cols = ntohl(num_cols);

  printf("Loading %d images (%dx%d)...\n", num_images, num_rows, num_cols);

  array->num_images = num_images;
  array->images = malloc(sizeof(*array->images) * num_images);

  while (n < num_images) {
    Image *image = malloc(sizeof(Image));
    image->num_rows = num_rows;
    image->num_cols = num_cols;
    image->pixels = malloc(sizeof(*image->pixels) * num_rows * num_cols);
    int i = 0;

    while (i < num_rows) {
      int j = 0;

      while (j < num_cols) {
        fread(&image->pixels[i * num_cols + j], sizeof(*image->pixels), 1, images_file);
        j++;
      }

      i++;
    }

    array->images[n] = image;
    n++;
  }  


  printf("Loading %d labels...\n", num_labels);

  n = 0;
  while (n < num_images) {
    unsigned char label;

    fread(&label, sizeof(label), 1, labels_file);
    array->images[n]->label = label;

    n++;
  }

  fclose(images_file);
  fclose(labels_file);
}

void render_image(Image *image, const char *filename) {
  unsigned char *pixels = image->pixels;
  int num_rows = image->num_rows;
  int num_cols = image->num_cols;
  int i;

  bmp_img img;
  bmp_img_init_df(&img, num_rows, num_cols);

  i = 0;

  while (i < num_rows) {
    int j = 0;
    
    while (j < num_cols) {
      unsigned char val = pixels[i * num_cols + j];

      bmp_pixel_init(&img.img_pixels[i][j], val, val, val);
      j++;
    }

    i++;
  }

  bmp_img_write(&img, filename);
	bmp_img_free(&img);
}