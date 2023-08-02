# Image Processing Pipeline

## Goal
Create a simple pipeline for image processing. This pipeline should be able to apply the following transformations: resize, Gaussian blur, and edge detection.

The flow is as follows:
1) Load a batch of images from an input folder
2) Group the images into batches
3) Apply one or more filter steps to each image in the batch
4) Save each batch and repeat

This is not intended to be a production pipeline which would likely run on a server but rather a simple example that focuses on the key logical pieces and Python OOP principles.

## Setup and installation
1. Clone this repo locally
2. Install pyenv using the steps here: (https://github.com/pyenv/pyenv)
3. Change directory to the cloned repo e.g. `cd img-pipeline`
4. Run the following command in your terminal: 
```
pyenv install 3.9.6
pyenv local 3.9.6
pip3 install -r requirements.txt
```
5. Next, run `python3 image_pipeline.py` to run the pipeline.
6. Then, run `python3 -m pytest test_image_pipeline.py` to run the tests.