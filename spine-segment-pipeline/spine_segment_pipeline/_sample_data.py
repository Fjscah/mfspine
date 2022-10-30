from .imgio import ski_imread
import os


def abspath(path):
    import os
    print(__file__)
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)

def sample_den_2d():
    img=ski_imread(r"D:\spine\napariplguin\spine-segment-pipeline\spine_segment_pipeline\data\images\dendrite_2d_decon.tif")
    return [(img, {"name": "dendrite_2d"})]


def sample_den_3d():
    img=ski_imread(r"D:\spine\napariplguin\spine-segment-pipeline\spine_segment_pipeline\data\images\dendrite_3d_decon.tif")
    return [(img, {"name": "dendrite_3d"})]
# print(abspath("55"))
def sample_den_2dt():
    img=ski_imread(r"D:\spine\napariplguin\spine-segment-pipeline\spine_segment_pipeline\data\images\dendrite_3d_decon.tif")
    return [(img, {"name": "dendrite_3d"})]