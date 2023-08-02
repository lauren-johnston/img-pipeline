import pytest
import cv2
import numpy as np
from image_pipeline import ImagePipeline, ResizeStep, GaussianBlurStep, EdgeDetectionStep, gen_image_path_batch

def test_gen_image_path_batch():
    """ Test that the gen_image_path_batch function returns correct number of 
    batches with the correct number of images within them"""
    image_paths = ["input/frog.jpg", "input/zebra.jpg", "input/bird.jpg", "input/tiger.jpg", "input/butterfly.jpg"]
    batch_size = 2
    
    batches = list(gen_image_path_batch(image_paths, batch_size))
    
    assert len(batches) == 3 
    assert all(len(batch) == batch_size for batch in batches[:-1])  
    assert len(batches[2]) == 1  

def test_image_pipeline_loads_images():
    """ Test that the load_image function loads correct number of images
    and has the writhe output format"""
    pipeline = ImagePipeline(input_folder='input', output_folder='output')
    image_paths = ['input/frog.jpg', 'input/zebra.jpg', 'input/bird.jpg']
    images = pipeline.load_batch(image_paths)
    assert len(images) == 3
    assert isinstance(images[0], np.ndarray)

def test_resize_step():
    """ Test that the resize step works correctly for given
    width and height"""
    image = cv2.imread('input/frog.jpg')
    resize_step = ResizeStep(width=100, height=200)
    resized_image = resize_step.apply(image)
    assert resized_image.shape[:2] == (200, 100)

def test_gaussian_blur_step():
    """ Test that the gaussian blur step is applied. This is tested by seeing
    if a transformation occured in the output image"""
    image = cv2.imread('input/frog.jpg')
    blur_step = GaussianBlurStep(kernel_size=3)
    blurred_image = blur_step.apply(image)
    # To test GaussianBlurStep, you can check that the output image is not the same as the input image
    assert not np.array_equal(blurred_image, image)

def test_edge_detection_step():
    """ Test that the edge detection step is applied. This is tested by seeing
    if a transformation occured in the output image"""
    image = cv2.imread('input/frog.jpg')
    edge_detection_step = EdgeDetectionStep(lower_thresh=100, upper_thresh=200)
    edge_detected_image = edge_detection_step.apply(image)
    # To test EdgeDetectionStep, you can check that the output image is not the same as the input image
    assert not np.array_equal(edge_detected_image, image)

def test_image_pipeline_runs_all_steps():
    """ Test that the image pipeline runs all steps correctly"""
    pipeline = ImagePipeline(input_folder='input', output_folder='output')
    pipeline.add_step(ResizeStep(width=200, height=200))
    pipeline.add_step(GaussianBlurStep(kernel_size=3))
    pipeline.add_step(EdgeDetectionStep(lower_thresh=100, upper_thresh=200))    
    image_paths = ['input/frog.jpg', 'input/zebra.jpg']
    pipeline.run(batch_size=2)

    output_images = pipeline.load_batch(['output/frog.jpg', 'output/zebra.jpg'])
    input_images = pipeline.load_batch(image_paths)
    assert len(output_images) == 2
    for output_image, input_image in zip(output_images, input_images):
        assert not np.array_equal(output_image, input_image)

def test_image_pipeline_works_single_step():
    """ Test that the image pipeline works correctly with a single step"""
    pipeline = ImagePipeline(input_folder='input', output_folder='output')
    pipeline.add_step(ResizeStep(width=200, height=200))
    image_paths = ['input/frog.jpg']
    pipeline.run(batch_size=1)
def test_steps_are_in_order():
    """ Test that the steps are in order in image pipeline"""
    pipeline = ImagePipeline(input_folder='input', output_folder='output')
    step1 = ResizeStep(width=200, height=200)
    step2 = GaussianBlurStep(kernel_size=3)
    pipeline.add_step(step1)
    pipeline.add_step(step2)
    assert pipeline.steps == [step1, step2]
def test_check_pipline_throws_error_no_steps():
    """ Test that an error is thrown if no steps are added to the pipeline"""
    with pytest.raises(ValueError):
        pipeline = ImagePipeline(input_folder='input', output_folder='output')
        pipeline.run(batch_size=2)

def test_pipeline_throws_error_batch_size_less_than_one():
    """ Test that an error is thrown if batch size is less than 1"""
    with pytest.raises(ValueError):
        pipeline = ImagePipeline(input_folder='input', output_folder='output')
        pipeline.add_step(ResizeStep(width=200, height=200))
        pipeline.run(batch_size=0)