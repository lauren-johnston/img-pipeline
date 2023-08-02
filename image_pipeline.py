# # Image Processing Pipeline
# Full code link with unit tests and readme: 
# https://github.com/lauren-johnston/img-pipeline
#
# ## Goal
# Create a simple pipeline for image processing. This pipeline should be able
# to apply the following transformations: resize, Gaussian blur, and edge detection.
#
# The flow is as follows:
# 1) Load a batch of images from an input folder
# 2) Group the images into batches
# 3) Apply one or more filter steps to each image in the batch
# 4) Save each batch and repeat
#
# This is not intended to be a production pipeline which would likely run on
# a server but rather a simple example that focuses on the key logical
# pieces and Python OOP principles.
import cv2
from abc import ABC, abstractmethod
import os
import glob
from typing import List, Iterator
import numpy as np

class ImagePipeline:
    """
    Given a folder of images (input_folder), create a pipeline for processing 
    images by batch and applying one or more steps. At the end of each batch,
    save the processed images to an output folder.
    """
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.steps: List['BasePiplineStep'] = []
        self._image_paths: List[str] = glob.glob(os.path.join(self.input_folder, '*'))

    def add_step(self, step: 'BasePiplineStep'):
        """Add a step to the pipeline"""
        self.steps.append(step)

    def load_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """Given an image_paths array, load a batch of images"""
        return [cv2.imread(path) for path in image_paths]
        
    def save_batch(self, image_paths: List[str], images: List[np.ndarray]):
        """Save a batch of images given an image_paths array and a list of images"""
        for image_path, image in zip(image_paths, images):
            output_path = os.path.join(self.output_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, image)

    def run(self, batch_size: int):
        """Run the pipeline on a list of images, batch by batch where batch_size is the number of images in each batch"""
        if batch_size <= 0:
            raise ValueError('Batch size must be > 0')
        if len(self.steps) == 0:
            raise ValueError('Pipeline must have at least one step.')
        for batch in gen_image_path_batch(self._image_paths, batch_size):
            images = self.load_batch(batch)

            for step in self.steps:
                images = step.batch_apply(images)
            self.save_batch(batch, images)

def gen_image_path_batch(image_paths: List[str], batch_size: int) -> Iterator[List[str]]:
    """Helper function to generate a batch of image paths"""
    for ix in range(0, len(image_paths), batch_size):
        yield image_paths[ix:ix+batch_size]

class BasePiplineStep(ABC):
    """Base class for pipeline steps"""
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply a transformation to an image"""
        pass

    def batch_apply(self, image_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Run the apply method on an image batch and return a list of images"""
        return [self.apply(image) for image in image_batch]

class ResizeStep(BasePiplineStep):
    """Resize an image to a given width and height"""
    def __init__(self, width: int, height: int):
        super().__init__()
        self.width = width
        self.height = height

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.width, self.height))

class EdgeDetectionStep(BasePiplineStep):
    """Detect edges in an image using Canny edge detector, given two thresholds
    (lower_thresh and upper_thresh)."""
    def __init__(self, lower_thresh: int, upper_thresh: int):
        super().__init__()
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.Canny(image, self.lower_thresh, self.upper_thresh)

class GaussianBlurStep(BasePiplineStep):
    """Apply a Gaussian blur to an image given a kernel size"""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)

if __name__ == '__main__':
    # This runs a simple image pipeline with all the steps.
    # For tests, see test_image_pipeline.py
    pipeline = ImagePipeline(input_folder='input', output_folder='output')
    pipeline.add_step(ResizeStep(width=250, height=250))
    pipeline.add_step(GaussianBlurStep(kernel_size=3))
    pipeline.add_step(EdgeDetectionStep(lower_thresh=150, upper_thresh=250))
    print(f"Running pipeline {pipeline}")
    pipeline.run(batch_size=5)
    print(f"Pipeline run complete, check output folder '{pipeline.output_folder}' for images")


