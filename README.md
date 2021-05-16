# GlyphGAN

Based on the paper referenced from the [GlyphGAN paper](https://arxiv.org/pdf/1905.12502.pdf).
> Built with Tensorflow GPU 1.14 and Python 3.6.

## Usage
Executing `python glyphgan` will run the training of GlyphGAN and output a frozen model upon completion.

When using the code, please make sure this script is in a directory with the dataset. A pre-made dataset is available to use, if you want to use your own dataset to train GlyphGAN make sure the main folder that data is placed in is called ```glyphs``` and the folder structure places glyphs images in 26 folders; named 0 to 25.

## Dataset
The dataset was sourced from Adobe Fonts. A total of 5000 fonts were used. Each glyph image in the dataset is 64x64 and saved as a png file.

## JS
To run the GlyphGAN in JS, take the output .pb file and convert it using [tensorflowjs](https://www.tensorflow.org/js/tutorials/conversion/import_saved_model).

## Tensorfair
For a live example of the GlyphGAN model visit [tensorfair](https://tensorfair.org/#glyphgan).
