import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pickle
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import open3d as o3d
import matplotlib.colors as mcolors
import pickle
from typing import List, Dict, Tuple
import os

# Change the current working directory to 'Detic'
os.chdir('Detic')

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
sys.path.insert(0, 'third_party/CenterNet2/')

import json, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries 
from Detic.detic.modeling.text.text_encoder import build_text_encoder
from collections import defaultdict
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
# SAM libraries
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# %%
def DETIC_predictor():
    # Build the detector and download our pretrained weights
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    # cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
    detic_predictor = DefaultPredictor(cfg)

    return detic_predictor

# %%
def default_vocab():
    detic_predictor = DETIC_predictor()
    # Setup the model's vocabulary using build-in datasets
    BUILDIN_CLASSIFIER = {
        'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
        'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
        'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
        'coco': 'datasets/metadata/coco_clip_a+cname.npy',
    }

    BUILDIN_METADATA_PATH = {
        'lvis': 'lvis_v1_val',
        'objects365': 'objects365_v2_val',
        'openimages': 'oid_val_expanded',
        'coco': 'coco_2017_val',
    }

    vocabulary = 'openimages' # change to 'lvis', 'objects365', 'openimages', or 'coco'
    metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
    classifier = BUILDIN_CLASSIFIER[vocabulary]
    num_classes = len(metadata.thing_classes)
    reset_cls_test(detic_predictor.model, classifier, num_classes)


def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def visualize_detic(output):
    output_im = cv2.cvtColot(output.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    cv2.imshow("Detic Predictions", output_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def custom_vocab():
    vocabulary = 'custom'
    metadata = MetadataCatalog.get("__unused2")
    metadata.thing_classes = ['cup', 'bowl', 'mug', 'bottle', 'cardboard', 'box', 'Tripod', 'Baseball bat' , 'Lamp', 'Mug Rack', 'Plate', 'Toaster', 'Spoon'] # Change here to try your own vocabularies!
    classifier = get_clip_embeddings(metadata.thing_classes)
    num_classes = len(metadata.thing_classes)
    reset_cls_test(detic_predictor.model, classifier, num_classes)

    # Reset visualization threshold
    output_score_threshold = 0.3
    for cascade_stages in range(len(detic_predictor.model.roi_heads.box_predictor)):
        detic_predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
    return metadata

def Detic(im, metadata, visualize=False):
    if im is None:
        print("Error: Unable to read the image file")

    # Run model and show results
    output =detic_predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata)
    out = v.draw_instance_predictions(output["instances"].to('cpu'))
    instances = output["instances"].to('cpu')
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    if visualize:
        visualize_detic(out)
    return boxes, classes

# %%
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# %%
def visualize_output(im, masks, input_boxes, classes):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, class_name in zip(input_boxes, classes):
        show_box(box.cpu().numpy(), plt.gca())
        x, y = box[:2].cpu().numpy()
        plt.gca().text(x, y - 5, class_name, color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='green', edgecolor='green', alpha=0.5))
    plt.axis('off')
    plt.show()
    
def SAM_predictor():
    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

def SAM(im, boxes, class_idx, visualize=False):
    sam_predictor.set_image(im)
    input_boxes = torch.tensor(boxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, im.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    if visualize:
        classes = [metadata.thing_classes[idx] for idx in class_idx]  # Get class names from indices
        visualize_output(im, masks, input_boxes, classes)
    return masks


def generate_colors(num_colors):
    hsv_colors = []
    for i in range(num_colors):
        hue = i / float(num_colors)
        hsv_colors.append((hue, 1.0, 1.0))

    return [mcolors.hsv_to_rgb(color) for color in hsv_colors]
def filter_ground_plane(points, z_threshold):
    mask = points[:, 2] > z_threshold
    points_no_ground = points[mask]
    return points_no_ground


def visualize_segmented_point_cloud(points, segments):
    if len(segments) == 0:
        return

    ax = plt.gca()
    ax.set_autoscale_on(False)
    colors = generate_colors(len(segments))
    z_threshold = -0.04966512
    ## Uncomment the below 3 lines for Lab collected data to remove the ground plane
    # points_no_ground = filter_ground_plane(points, z_threshold)
    # h,w = points.shape
    # ground_mask = (points[:, 2] > z_threshold).reshape(h, w)[:,:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) # Pass the points_no_ground here for lab collected data
    pcd.colors = o3d.utility.Vector3dVector(colors)

    segment_pcds = []
    for idx, mask in enumerate(segments):
        segmentation_mask = mask.cpu().numpy()
        # updated_segmentation_mask = segmentation_mask[ground_mask]
        segment_indices = np.where(segmentation_mask.ravel())[0].tolist()
        segment_pcd = pcd.select_by_index(segment_indices)
        segment_pcd.paint_uniform_color(colors[idx])
        segment_pcds.append(segment_pcd)

    # Create a coordinate frame (scale)
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # Add the coordinate frame to the geometries to be visualized 
    o3d.visualization.draw_geometries(segment_pcds)

def combine_point_clouds(clouds):
    combined_cloud = np.concatenate(clouds, axis=0)
    return combined_cloud

def combine_masks(masks_list):
    combined_masks = []
    for masks in masks_list:
        combined_masks.extend(masks)
    return combined_masks

import time 
if __name__=="__main__":
    with open("../Data/ondrejs_kitchen_1.pkl", 'rb') as f:
        data = pickle.load(f)

    detic_predictor = DETIC_predictor()
    sam_predictor = SAM_predictor()
    metadata = custom_vocab()
    results = []
########################## For home collected data ##########################
    for sample_idx in range(len(data)):
        # select the point clouds and RGB images for the selected sample
        cloud = data[sample_idx]["clouds"]
        image = data[sample_idx]["images"]
        depth = data[sample_idx]["depths"]

        st = time.time()
        boxes, class_idx = Detic(image, metadata)
        masks = SAM(image, boxes, class_idx, True)

        result = {
            "cloud": cloud,
            "image": image,
            "depth": depth,
            "boxes": boxes,
            "class_idx": class_idx,
            "masks": masks,
        }
        results.append(result)

        # Uncomment this line if you want to visualize the segmented point cloud for each iteration
        # visualize_segmented_point_cloud(cloud, masks)

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

########################## For Lab collected data ##########################
    # for sample_idx in range(len(data)):
    #     # select the point clouds and RGB images for the selected sample
    #     clouds = data[sample_idx]["clouds"]
    #     images = data[sample_idx]["images"]
    #     depths = data[sample_idx]["depths"]
    #     # combined_point_cloud = cloud[0]
    #     # combined_mask_list = []

    #     # iterate over the clouds and images and plot them side by side
    #     for cloud, image, depth in zip(clouds, images, depths):

    #         st = time.time()
    #         boxes, class_idx = Detic(image, metadata)
    #         masks = SAM(image, boxes, class_idx, False) 
    #     # combined_point_cloud = np.concatenate((combined_point_cloud, cloud), axis=0)
    #     # combined_mask_list.extend(masks)
    #         visualize_segmented_point_cloud(cloud, masks)
    #     break