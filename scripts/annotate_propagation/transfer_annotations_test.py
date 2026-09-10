import os
import cv2
import torch
import json
import numpy as np
import argparse
from PIL import Image
import supervision as sv
from typing import List, Union
from glob import glob
import shutil
# Transformers and SAM imports
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor
from qwen_vl_utils import process_vision_info

import open3d as o3d
import yaml
from o3d_process import remove_outlier_o3d, trim_pcd_by_z
from system_params import *


import re

def natural_key(string_):
    """Helper to sort strings containing numbers in human order (1, 2, 10...)"""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]

########## Setup for open3d ########### 
def load_intrinsics_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    intr = cfg["intrinsics"]
    return intr

# Load camera intrinsics
intr = load_intrinsics_from_yaml("camera_config.yaml")
fx = intr["fx"]
fy = intr["fy"]
cx = intr["cx"]
cy = intr["cy"]
width = intr["width"]
height = intr["height"]
DEPTH_SCALE = intr["meters_per_unit"]
DEPTH_TRUNC = intr["far"]

# Set Open3D camera intrinsics
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def rgbd_to_pcd_mask_cropped(color_path, depth_path, mask, TF_base_cam, intrinsic, crop_indices):
    u_s, u_e, v_s, v_e = crop_indices
    
    # Load and Crop Color/Depth
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)[v_s:v_e, u_s:u_e]
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[v_s:v_e, u_s:u_e]
    if depth_image.dtype == np.uint16:
        depth_image = depth_image.astype(np.float32) * 0.001
    mask_cropped = mask[v_s:v_e, u_s:u_e]

    # Convert BGR to RGB
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Apply mask to both
    color_image_rgb = cv2.bitwise_and(color_image_rgb, color_image_rgb, mask=mask_cropped)
    depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask_cropped)

    # Create Open3D images
    o3d_color = o3d.geometry.Image(color_image_rgb)
    o3d_depth = o3d.geometry.Image(depth_image.astype(np.float32) * DEPTH_SCALE)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        convert_rgb_to_intensity=False,
        depth_scale=1.0, depth_trunc=DEPTH_TRUNC
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform(TF_base_cam)
    return pcd

def rgbd_to_pcd_mask(color_path, depth_path, mask, TF_base_cam, intrinsic):
    
    # Load color and depth images
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    

    # Convert depth to float32 meters if needed
    if depth_image.dtype == np.uint16:
        depth_image = depth_image.astype(np.float32) * 0.001
    # Convert BGR to RGB for Open3D
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # bitwise mask 
    color_image_rgb = cv2.bitwise_and(color_image_rgb, color_image_rgb, mask=mask)
    depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)

    

    # Create Open3D images
    o3d_color = o3d.geometry.Image(color_image_rgb)
    o3d_depth = o3d.geometry.Image(depth_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        convert_rgb_to_intensity=False,
        depth_scale=1.0,  # already in meters
        depth_trunc=DEPTH_TRUNC   # max depth in meters
    )

    # Creat pcd
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )
   
    # Voxel grid filter

    voxel_size = 0.005
    pcd = pcd.voxel_down_sample(voxel_size)
    # print(pcd.get_center())

    # Transform 
    # TF_base_cam[:3,3] = [0,0,0]
    # pcd.transform(np.linalg.inv(TF_base_cam))
    pcd.transform(TF_base_cam)
    #print(f"pcd shape after transform: {pcd.points}")

    # Generate point cloud
    return pcd

def rgbd_to_pcd(color_path, depth_path, TF_base_cam, intrinsic):
    # Load color and depth images
   
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Convert depth to float32 meters if needed
    if depth_image.dtype == np.uint16:
        depth_image = depth_image.astype(np.float32) / 1000.0

    # Convert BGR to RGB for Open3D
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Create Open3D images
    o3d_color = o3d.geometry.Image(color_image_rgb)
    o3d_depth = o3d.geometry.Image(depth_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        convert_rgb_to_intensity=False,
        depth_scale=1.0,  # already in meters
        depth_trunc=DEPTH_TRUNC   # max depth in meters
    )

    # Creat pcd
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )

    # Voxel grid filter
    voxel_size = 0.005
    pcd = pcd.voxel_down_sample(voxel_size)

    # Transform 
    # TF_base_cam[:3,3] = [0,0,0]
    # pcd.transform(np.linalg.inv(TF_base_cam))
    pcd.transform(TF_base_cam)

    # Generate point cloud
    return pcd

def draw_with_black_bg(geometry_list, estimate_if_missing=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    for geometry in geometry_list:
        if isinstance(geometry, o3d.geometry.PointCloud):
            # 1. Estimate normals if the PCD doesn't have them
            if estimate_if_missing and not geometry.has_normals():
                geometry.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
                )
        vis.add_geometry(geometry)
    
    # 2. Configure Render Options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    # Enable normal visualization
    opt.point_show_normal = True
    
    vis.run()
    vis.destroy_window()

# ==============================
# Configuration & Global Models
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAM_CHECKPOINT_PATH = "/home/aurora/132/pretraining/weights/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"
VQA_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
DINO_MODEL_NAME = "openmmlab-community/mm_grounding_dino_large_all"

# ==============================
# Model Loader Class
# ==============================
class ModelPipeline:
    def __init__(self):
        print("--- Initializing Models (This may take a minute) ---")
        # 1. Load Qwen-VL
        # self.vqa_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     VQA_MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        # )
        # self.vqa_processor = AutoProcessor.from_pretrained(VQA_MODEL_NAME)

        # 2. Load Grounding DINO
        self.dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_NAME).to(DEVICE).eval()

        # 3. Load SAM
        self.sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
        self.sam_predictor = SamPredictor(self.sam)
        
        # Annotators
        BLUE = sv.Color(0, 0, 255)
        self.mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette([BLUE]))
        self.box_annotator = sv.BoxAnnotator()

    def process_image(self, image_path, target_object, output_dir):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: return
        
        base_name = os.path.basename(image_path).split('.')[0]
        
        # # 1. Qwen-VL Inference
        # if target_object != "all":
        #     prompt = (
        #         f"Task: Search the image specifically for a '{target_object}' on the table. "
        #         f"If the '{target_object}' is detected, prioritize it in the output."
        #         #f"Also include other clearly visible and prominent distinct object categories. "
        #         f"Output ONLY a valid Python list of unique strings. Output ONLY one object"
        #     )
        # else: 
        #     prompt = (
        #         f"Task: Search the image for all distinct object categories on the table. "
        #         f"Output ONLY a valid Python list of unique strings."
        #     )
        
        # messages = [[{"content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]]
        # text = self.vqa_processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
        # image_inputs, _ = process_vision_info(messages[0])
        
        # inputs = self.vqa_processor(text=[text], images=[image_inputs], padding=True, return_tensors="pt").to(DEVICE)
        # generated_ids = self.vqa_model.generate(**inputs, max_new_tokens=128)
        # output_text = self.vqa_processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        # try:
        #     detected_classes = eval(output_text)
        #     # Simple refinement: if sugar/cheese, add 'box'
        #     detected_classes = [c + " box" if c.lower() in ["sugar", "cheeze it"] else c for c in detected_classes]
        # except:
        #     return

        # 2. Grounding DINO
        detected_classes = [target_object]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        dino_inputs = self.dino_processor(images=image_pil, text=[detected_classes], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.dino_model(**dino_inputs)
        
        dino_res = self.dino_processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=[image_pil.size[::-1]], threshold=0.15
        )[0]

        # Map labels to IDs
        raw_labels = dino_res.get("text_labels", dino_res.get("labels", []))
        processed_class_ids = []
        for label in raw_labels:
            clean_label = label.replace(" ", "").replace("_", "")
            match_idx = -1
            for i, p in enumerate(detected_classes):
                if clean_label == p.replace(" ", "").replace("_", ""):
                    match_idx = i
                    break
            processed_class_ids.append(match_idx)

        detections = sv.Detections(
            xyxy=dino_res["boxes"].cpu().numpy(),
            class_id=np.array(processed_class_ids),
            confidence=dino_res["scores"].cpu().numpy()
        )

        # 3. SAM Segmentation
        self.sam_predictor.set_image(image_rgb)
        if len(detections.xyxy) > 0:
            all_masks = []
            for box in detections.xyxy:
                masks, _, _ = self.sam_predictor.predict(box=box[None, :], multimask_output=False)
                all_masks.append(masks[0])
            detections.mask = np.stack(all_masks)
        
        # 4. Save Binary Masks (Target Specific)
        target_idx = -1
        for i, cid in enumerate(detections.class_id):
            if cid != -1 and detected_classes[cid].lower() == target_object.lower():
                target_idx = i
                break
        
        if target_idx != -1:
            binary_mask = (detections.mask[target_idx] * 255).astype(np.uint8)
            mask_out = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_out, binary_mask)

            # 5. Visualization
            valid_indices = detections.class_id != -1
            detections = detections[valid_indices]
            if len(detections) > 0:
                labels = [f"{detected_classes[cid]} {conf:.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]
                ann_img = self.mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
                ann_img = self.box_annotator.annotate(scene=ann_img, detections=detections, labels=labels)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_annotated.jpg"), ann_img)
            detections_json_dict = { "bboxes": detections.xyxy.tolist(),
                                    "class_ids": detections.class_id.tolist(),
                                    "confidences": detections.confidence.tolist(),
                                    "mask_path": mask_out}

            return detections_json_dict, detections 
        else:
            return None, None
    
def project_pcd_to_pixels(pcd, T_base_cam_target, intr):
    # 1. Transform points from World (Base) to Target Camera frame
    pcd_curr = o3d.geometry.PointCloud(pcd)
    pcd_curr.transform(np.linalg.inv(T_base_cam_target))
    pts_3d = np.asarray(pcd_curr.points)
    
    # 2. Filter points behind camera (Z must be positive)
    valid_depth = pts_3d[:, 2] > 0.1
    pts_3d = pts_3d[valid_depth]
    
    if len(pts_3d) == 0: 
        return np.array([])

    # 3. Use Open3D Intrinsic Matrix for Projection
    # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    K = intr.intrinsic_matrix 

    # Normalize the 3D points by their depth (Z coordinate)
    # This transforms [X, Y, Z] -> [X/Z, Y/Z, 1]
    pts_normalized = pts_3d / pts_3d[:, 2:3]

    # Matrix multiplication: [u, v, 1]^T = K * [X/Z, Y/Z, 1]^T
    # Using .T on K for right-side multiplication with (N, 3) array
    points_2d_homo = pts_normalized @ K.T

    # The first two columns are our u (horizontal) and v (vertical) pixels
    points_2d = points_2d_homo[:, :2]

    # 4. Boundary Check: Filter points that are outside the image frame
    mask_in_bounds = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < intr.width) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < intr.height)
    )
    
    return points_2d[mask_in_bounds]

def pixels_to_pcd(pixels, depth_map, T_base_cam_curr, intr):
    """
    Inverse of project_pcd_to_pixels.
    Converts 2D pixels + depth into a 3D Point Cloud in the World (Base) frame.
    """
    # 1. Extract Intrinsic parameters
    K = intr.intrinsic_matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pts_3d_cam = []
     # Convert depth to float32 meters if needed
    if depth_map.dtype == np.uint16:
        depth_map = depth_map.astype(np.float32) / 1000.0


    # 2. Back-project each pixel to Camera Frame
    for u, v in pixels:
        u_int, v_int = int(round(u)), int(round(v))
        
        # Boundary check for depth map access
        if 0 <= u_int < intr.width and 0 <= v_int < intr.height:
            z = depth_map[v_int, u_int]
            
            # Filter out invalid depth (0 or noise)
            if z > 0 and z < 1.0:
                # Inverse Projection Equations:
                # x = (u - cx) * z / fx
                # y = (v - cy) * z / fy
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                pts_3d_cam.append([x, y, z])

    if len(pts_3d_cam) == 0:
        return o3d.geometry.PointCloud()

    # 3. Create Open3D PointCloud in Camera Frame
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts_3d_cam))

    # 4. Transform from Camera Frame to World (Base) Frame
    # Since pts_3d_cam is in 'curr' frame, we multiply by T_base_cam_curr
    pcd.transform(T_base_cam_curr)

    return pcd

def create_cropped_pointcloud(anchor_mask, image_file, depth_file, T_anchor, crop_size=10):
    # Create the "Golden" PCD from the first frame's detection
    # (Assuming the first mask in the list is your target)
    
    # Get coordinates of all masked pixels
    y_coords, x_coords = np.where(anchor_mask > 0)
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    global fx, fy, cx, cy
    

    if len(x_coords) > 0:
        # Average of all x and y positions
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        anchor_mask_center = (center_x, center_y)
    else:
        anchor_mask_center = None

    # Define crop size (e.g., 200x200 around the center)    
    half_size = crop_size // 2

    # Calculate pixel boundaries
    u_start = max(0, anchor_mask_center[0] - half_size)
    u_end = min(width, anchor_mask_center[0] + half_size)
    v_start = max(0, anchor_mask_center[1] - half_size)
    v_end = min(height, anchor_mask_center[1] + half_size)

    # Update Camera Intrinsics for the cropped view
    # cx_new = cx - u_start, cy_new = cy - v_start
    cropped_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        u_end - u_start, v_end - v_start, 
        fx, fy, 
        cx - u_start, cy - v_start
    )

    # Define the 5x5 grid around the center
    grid_points = []
    for v in range(anchor_mask_center[1] - 2, anchor_mask_center[1] + 3):
        for u in range(anchor_mask_center[0] - 2, anchor_mask_center[0] + 3):
            grid_points.append([u, v])

    # Define the dimensions for cutting
    crop_indices = (u_start, u_end, v_start, v_end)
    
    anchor_pcd = rgbd_to_pcd_mask_cropped(image_file, depth_file, anchor_mask, T_anchor, cropped_intrinsic, crop_indices)
    box = np.array([min_x, min_y, max_x, max_y])
    return anchor_pcd, box[None, :]

def visualize_centroids(p1_pcd, curr_pcd, dist_threshold=0.15):
    # 1. Calculate Centroids
    p1_center = p1_pcd.get_center()
    curr_center = curr_pcd.get_center()
    
    # 2. Create Spheres to represent Centroids
    p1_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    p1_sphere.paint_uniform_color([0, 0, 1]) # Blue
    p1_sphere.translate(p1_center)
    
    dist = np.linalg.norm(curr_center - p1_center)
    curr_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    
    color = [0, 1, 0] if dist < dist_threshold else [1, 0, 0] # Green if pass, Red if fail
    curr_sphere.paint_uniform_color(color)
    curr_sphere.translate(curr_center)
    
    # 3. Create a Line between the two points
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([p1_center, curr_center])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([color])

    # 4. Add Origin Axis (X=Red, Y=Green, Z=Blue)
    # Size 0.1 means 10cm long axis lines
    origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # 5. Add Plane/Box at Origin (Size: 0.2 x 0.2 x 0.05)
    # create_box(width, height, depth)
    origin_plane = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.05)
    # Center the box so the origin is in its middle
    origin_plane.translate([-0.1, -0.1, -0.025]) 
    origin_plane.paint_uniform_color([0.7, 0.7, 0.7]) # Light Gray

    # 6. Prepare Point Clouds
    p1_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    curr_pcd.paint_uniform_color([0.8, 0.8, 0.2])

    # 7. Visualize
    print(f"Visualizing 3D Distance: {dist:.4f}m")
    geometries = [p1_pcd, curr_pcd, p1_sphere, curr_sphere, line_set, origin_axes, origin_plane]
    o3d.visualization.draw_geometries(geometries,
                                      window_name="Two-Pointer 3D Centroid Check",
                                      width=1280, height=720)


def get_spatial_drift(p1_pcd, prompt_points, depth_map, T_curr, intrinsic, base_height):
    """Checks if the projected points have drifted too far from the reference centroid."""
    # Process Current Points
    curr_pcd = pixels_to_pcd(prompt_points, depth_map, T_curr, intrinsic)
    curr_pcd, _ = remove_outlier_o3d(curr_pcd, nb_neighbors=20)
    curr_pcd = trim_pcd_by_z(curr_pcd, base_height)
    
    if curr_pcd.is_empty():
        return float('inf'), None
    
    # Process Reference
    ref_pcd, _ = remove_outlier_o3d(p1_pcd)
    ref_pcd = trim_pcd_by_z(ref_pcd, base_height)
    
    dist = np.linalg.norm(curr_pcd.get_center() - ref_pcd.get_center())
    return dist, curr_pcd

def get_largest_connected_component(mask):
    """Removes noise by keeping only the largest blob in a binary mask."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)

# def get_validated_mask(pipeline, img_bgr, color_path, prompt_points, biggest_area, args):
#     """
#     Tries SAM first. If SAM fails, tries VLM.
#     Returns: (mask, area, source_name) or (None, 0, None)
#     """
#     input_pts = np.median(prompt_points, axis=0)
    
#     # --- 1. TRY SAM ---
#     pipeline.sam_predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
#     masks, scores, _ = pipeline.sam_predictor.predict(
#         point_coords=np.expand_dims(input_pts, 0), 
#         point_labels=np.array([1]), 
#         multimask_output=False
#     )
#     sam_mask = masks[0].astype(np.uint8)
#     sam_area = np.sum(sam_mask > 0)

#     if (biggest_area * 0.5) <= sam_area <= (biggest_area * 1.5):
#         return sam_mask, sam_area, "SAM"

#     # --- 2. TRY VLM FALLBACK ---
#     res, v_det = pipeline.process_image(color_path, args.target_object, args.output_dir)
#     if v_det and len(v_det.mask) > 0:
#         v_mask = v_det.mask[0].astype(np.uint8)
#         v_area = np.sum(v_mask > 0)
        
#         # Validation checks
#         u, v = int(input_pts[0]), int(input_pts[1])
#         h, w = v_mask.shape
#         is_inside = (0 <= u < w and 0 <= v < h) and (v_mask[v, u] > 0)
#         area_ok = (biggest_area * 0.2) <= v_area <= (biggest_area * 2.0)

#         if is_inside and area_ok:
#             return v_mask, v_area, "VLM_Reset"

#     return None, 0, None

def get_validated_mask_vlm_first(v_det, pipeline, img_bgr, color_path, prompt_points, biggest_area, args):
    """
    1. Runs VLM first to get all candidate masks.
    2. Checks if the prompt point (median) is inside any VLM mask.
    3. If none match, falls back to SAM point prediction.
    """
    input_pts = np.median(prompt_points, axis=0)
    u, v = int(input_pts[0]), int(input_pts[1])
    
    # --- 1. TRY VLM FIRST ---
    if v_det and len(v_det.mask) > 0:
        for i in range(len(v_det.mask)):
            v_mask = v_det.mask[i].astype(np.uint8)
            v_area = np.sum(v_mask > 0)
            
            # Boundary check
            h, w = v_mask.shape
            if not (0 <= u < w and 0 <= v < h):
                continue
                
            # Alignment check: Is our projected point inside this specific mask?
            is_inside = v_mask[v, u] > 0
            
            # Area consistency check (Bypassed if --no_area_filter is set)
            if args.no_area_filter:
                area_ok = True
            else:
                area_ok = (biggest_area * 0.2) <= v_area <= (biggest_area * 2.5)

            if is_inside and area_ok:
                return v_mask, v_area, "VLM_Primary"

    # --- 2. FALLBACK TO SAM POINT PREDICTION ---
    pipeline.sam_predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = pipeline.sam_predictor.predict(
        point_coords=np.expand_dims(input_pts, 0), 
        point_labels=np.array([1]), 
        multimask_output=False
    )
    
    sam_mask = masks[0].astype(np.uint8)
    sam_area = np.sum(sam_mask > 0)

    # SAM validation (Bypassed if --no_area_filter is set)
    if args.no_area_filter:
        return sam_mask, sam_area, "SAM_Fallback"
    
    # Original stricter SAM logic
    if (biggest_area * 0.5) <= sam_area <= (biggest_area * 1.5):
        return sam_mask, sam_area, "SAM_Fallback"

    return None, 0, None

# def get_validated_mask_vlm_first(v_det, pipeline, img_bgr, color_path, prompt_points, biggest_area, args):
#     """
#     1. Runs VLM first to get all candidate masks.
#     2. Checks if the prompt point (median) is inside any VLM mask.
#     3. If none match, falls back to SAM point prediction.
#     """
#     input_pts = np.median(prompt_points, axis=0)
#     u, v = int(input_pts[0]), int(input_pts[1])
    
#     # --- 1. TRY VLM FIRST ---
#     # Assuming pipeline.process_image returns detections with multiple masks
#     #res, v_det = pipeline.process_image(color_path, args.target_object, args.output_dir)
    
#     if v_det and len(v_det.mask) > 0:
#         # Check every mask returned by VLM
#         for i in range(len(v_det.mask)):
#             v_mask = v_det.mask[i].astype(np.uint8)
#             v_area = np.sum(v_mask > 0)
            
#             # Boundary check
#             h, w = v_mask.shape
#             if not (0 <= u < w and 0 <= v < h):
#                 continue
                
#             # Alignment check: Is our projected point inside this specific mask?
#             is_inside = v_mask[v, u] > 0
#             # Area consistency check
#             area_ok = (biggest_area * 0.2) <= v_area <= (biggest_area * 2.5)

#             if is_inside and area_ok:
#                 return v_mask, v_area, "VLM_Primary"

#     # --- 2. FALLBACK TO SAM POINT PREDICTION ---
#     # Only runs if VLM didn't find a matching mask
#     pipeline.sam_predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
#     masks, scores, _ = pipeline.sam_predictor.predict(
#         point_coords=np.expand_dims(input_pts, 0), 
#         point_labels=np.array([1]), 
#         multimask_output=False
#     )
    
#     sam_mask = masks[0].astype(np.uint8)
#     sam_area = np.sum(sam_mask > 0)

#     # SAM validation (usually stricter on area than VLM)
#     if (biggest_area * 0.5) <= sam_area <= (biggest_area * 1.5):
#         return sam_mask, sam_area, "SAM_Fallback"

#     return None, 0, None


# Extract the numeric index from the filename (e.g., "color_0001.png" -> 1)
def get_num(path):
    return int(re.search(r'\d+', os.path.basename(path)).group())

def precompute_all_detections(pipeline, image_files, target_classes):
    """
    Runs DINO on all images once and stores results.
    target_classes: list of strings, e.g., ["cup", "hammer", "drill"]
    """
    global_detections = {} # frame_id -> {obj_name -> box}

    for img_path in tqdm(image_files, desc="Precomputing DINO Boxes"):
        image_pil = Image.open(img_path).convert("RGB")
        # DINO can handle all classes in a single text prompt
        inputs = pipeline.dino_processor(
            images=image_pil, 
            text=[target_classes], 
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = pipeline.dino_model(**inputs)
        
        results = pipeline.dino_processor.post_process_grounded_object_detection(
            outputs=outputs, 
            target_sizes=[image_pil.size[::-1]], 
            threshold=0.15
        )[0]

        frame_id = get_num(img_path)
        global_detections[frame_id] = {}

        # Parse DINO labels back to class names
        boxes = results["boxes"].cpu().numpy()
        labels = results.get("text_labels", [])
        scores = results["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            clean_label = label.strip().lower()
            # Store only the highest confidence box per class for this frame
            if clean_label not in global_detections[frame_id] or score > global_detections[frame_id][clean_label]['score']:
                global_detections[frame_id][clean_label] = {
                    "box": box,
                    "score": score
                }
                
    return global_detections

def load_target_detections(run_num, target_object):
    """
    Loads only the detections matching the target_object string from the seed.json.
    """
    # 1. Construct path and get directory context
    json_path = "/media/aurora/easystore/dataset/seed_anno_click/run{}/seed.json".format(run_num)
    base_dir = os.path.dirname(json_path)

    if not os.path.exists(json_path):
        print(f"Error: JSON not found at {json_path}")
        return sv.Detections.empty()

    with open(json_path, 'r') as f:
        data = json.load(f)

    bboxes, class_ids, masks, class_names = [], [], [], []

    # 2. Selective loading loop
    for obj in data['objects']:
        # Only proceed if the label matches exactly
        if obj['label'] == target_object:
            mask_full_path = os.path.join(base_dir, obj['mask_file'])
            
            # Load mask only for the target
            mask_img = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
            
            if mask_img is not None:
                bboxes.append(obj['bbox_xyxy'])
                class_ids.append(obj['object_id'])
                class_names.append(obj['label'])
                masks.append(mask_img > 0)
            else:
                print(f"Warning: Mask file missing for {target_object} at {mask_full_path}")

    # 3. Handle cases where no matches were found
    if not bboxes:
        return sv.Detections.empty()

    # 4. Wrap in Supervision Detections object
    return sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        mask=np.array(masks),
        class_id=np.array(class_ids),
       
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, help="Folder containing images (overrides default naming)")
    parser.add_argument("--tf_dir", type=str, help="Folder containing TF .npy files (overrides default naming)")
    parser.add_argument("--output_dir", type=str, default="saved_transfer_results")
    parser.add_argument("--target_object", type=str, default="glue bottle")
    parser.add_argument("--stop_vlm", action="store_true", default=False, help="Disable VLM (default: True)"  )
    parser.add_argument("--run_num", type=int, default=10, help="Run number identifier")
    # --- NEW FILTER TOGGLES ---
    parser.add_argument("--no_spatial_filter", action="store_true", help="Disable prompt point count check")
    parser.add_argument("--no_drift_filter", action="store_true", help="Disable spatial drift/distance check")
    parser.add_argument("--no_area_filter", action="store_true", help="Disable area check")
    args = parser.parse_args()
   
    det_0 =  load_target_detections(args.run_num, args.target_object)
    
    
    root = os.getcwd()
    args.images_dir = os.path.join(root, args.images_dir)
    args.tf_dir = os.path.join(root, args.tf_dir)

    # Initialize Pipeline ONCE
    pipeline = ModelPipeline()
    
   # 1. Get all image files first
    raw_files = glob(os.path.join(args.images_dir, "*.[jJ][pP][gG]")) + \
                glob(os.path.join(args.images_dir, "*.[pP][nN][gG]"))

    # 2. Filter for files that contain "color" in the filename
    # We use .lower() to ensure it catches "Color", "COLOR", or "color"
    # Apply the natural key to your sorting logic
   
    import random 
   
    image_files = sorted(
        [f for f in raw_files if "color" in os.path.basename(f).lower()],
        key=natural_key
    )
    idx = list(range(len(image_files)))
    random.shuffle(idx)
   
    depth_files = [f.replace("color_", "depth_") for f in image_files]
    all_results = []

    # TF file
    # 1. Get the filenames and sort them alphabetically/numerically
    tf_files = tf_files = sorted(
        [f for f in os.listdir(args.tf_dir) if f.endswith(".npy")], 
        key=natural_key
    )
    image_files = [image_files[i] for i in idx]
    depth_files = [depth_files[i] for i in idx] 
    tf_files = [tf_files[i] for i in idx]

    # 2. Load the matrices in that specific order
    TF_base_cam = [np.load(os.path.join(args.tf_dir, f)) for f in tf_files]
    


    if not image_files:
        print(f"No images containing 'color' found in {args.images_dir}")
        return
    # Use tqdm for a nice progress 
    master_json_path = os.path.join(args.output_dir, "all_detections.json")
   
    if not args.stop_vlm:
        if os.path.exists(args.output_dir):
            for filename in os.listdir(args.output_dir):
                file_path = os.path.join(args.output_dir, filename)

                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)          # delete file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)     # delete subfolder
                except Exception as e:
                    print(f'Failed to delete {file_path}: {e}')
            
            os.makedirs(args.output_dir, exist_ok=True)
        
        all_detections = {}
        for img_path in tqdm(image_files, desc="Processing Images"):
            res, detections = pipeline.process_image(img_path, args.target_object, args.output_dir)
            if res:
                all_results.append(res)
                all_detections[img_path] = detections
            
        # Save the master JSON
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(master_json_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nProcessing complete. Summary saved to: {master_json_path}")
    else: 
        master_json_path = os.path.join(args.output_dir, "all_detections.json")
        with open(master_json_path, 'r') as f:
            all_detections = json.load(f)
        for idx, results in enumerate(all_detections):
            all_detections[idx]['mask'] = [cv2.imread(results['mask_path'], cv2.IMREAD_UNCHANGED)]
    
   

    # --- STEP A: INITIALIZE ANCHOR ---
    # We use the very first detection to build our "Reference Object"
  
    
  
    anchor_idx = 0
    det_0 =  all_detections[image_files[anchor_idx]]
    T_anchor = TF_base_cam[anchor_idx]
    
    anchor_mask = det_0.mask[0].astype(np.uint8) if not args.stop_vlm else det_0['mask'][0].astype(np.uint8)
    anchor_pcd = rgbd_to_pcd_mask(image_files[0], depth_files[0], anchor_mask, T_anchor, intrinsic)

    

    

    print(f"Anchor initialized with {len(anchor_pcd.points)} points.")

    # --- STEP A: INITIALIZE ---
    anchor_idx = 0
    # # Extract the numeric index from the filename (e.g., "color_0001.png" -> 1)
    # def get_num(path):
    #     return int(re.search(r'\d+', os.path.basename(path)).group())

    # --- STEP B: TWO-POINTER VALIDATION (FOR LOOP) ---
    # Initial Reference (P1)
    p1_mask = all_detections[image_files[anchor_idx]].mask[0].astype(np.uint8) if not args.stop_vlm  else all_detections[image_files[anchor_idx]]['mask'][0].astype(np.uint8)
    p1_area = np.sum(p1_mask > 0)
    p1_pcd = rgbd_to_pcd_mask(image_files[0], depth_files[0], p1_mask, TF_base_cam[0], intrinsic)
    p1_num = get_num(image_files[0])
    biggest_area = p1_area

    all_json_results = []
    # Extract Mask
    anchor_mask = det_0.mask[0].astype(np.uint8) if not args.stop_vlm else np.array(det_0['mask'][0]).astype(np.uint8)
   

    # Calculate BBox for Frame 0
    y, x = np.where(anchor_mask)
    bbox_0 = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]

    # Save Seed Mask Image
    mask_filename_0 = f"mask_{0:05d}.png"
    mask_path_0 = os.path.join(args.output_dir, "masks", mask_filename_0)
    cv2.imwrite(mask_path_0, anchor_mask * 255)

    # Append Seed to Results
    all_json_results.append({
        "image_path": image_files[0],
        "mask_path": mask_path_0,
        "bbox": bbox_0,
        "obj_name": args.target_object,
        "source": "Initial_Seed",
        "frame_id": 0
    })

    total_tracking_opportunities = 0

    for i in tqdm(range(1, len(image_files))):
        # --- 1. SETUP ---
        color_path, depth_path = image_files[i], depth_files[i]
        T_curr, this_num = TF_base_cam[i], get_num(color_path)
        img_bgr = cv2.imread(color_path)
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        v_det = all_detections.get(color_path)

        # --- 2. SPATIAL PRE-FILTER ---
        prompt_points = project_pcd_to_pixels(p1_pcd, T_curr, intrinsic)
        # Check Spatial Filter (Point Count)
        if not args.no_spatial_filter:
            if len(prompt_points) < 10:
                print(f"Frame {this_num}: Too few prompt points ({len(prompt_points)}). Skipping.")
                continue
        elif len(prompt_points) == 0:
            # Hard safety: if there are literally 0 points, we can't segment even if filter is "off"
            continue

        # Check Drift Filter
        if not args.no_drift_filter:
            dist, _ = get_spatial_drift(p1_pcd, prompt_points, depth_map, T_curr, intrinsic, BASE_TABLE_HEIGHT)
            if dist > 0.05:
                print(f"Frame {this_num}: Drift too high ({dist:.4f}m). Skipping.")
                continue

        # --- 3. SEGMENTATION SELECTION ---
        final_mask, curr_area, source = get_validated_mask_vlm_first(
            v_det, pipeline, img_bgr, color_path, prompt_points, biggest_area, args
        )

        if final_mask is None:
            print(f"Frame {this_num}: Both SAM and VLM failed. P1 stays at {p1_num}.")
            continue
        total_tracking_opportunities += 1

        # --- 4. UPDATE STATE ---
        # Clean up the mask (largest component) and update the tracking reference
        final_mask = get_largest_connected_component(final_mask)
        if curr_area > biggest_area:
            biggest_area = curr_area
        
        p1_mask, p1_area, p1_num = final_mask, curr_area, this_num
        p1_pcd = rgbd_to_pcd_mask(color_path, depth_path, p1_mask, T_curr, intrinsic)
        
        print(f"Frame {this_num}: {source} Success. Area: {curr_area}")

        # --- 5. SAVE & ANNOTATE ---
        # Bbox and Annotations
        y, x = np.where(final_mask)
        bbox = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
        
        res_det = sv.Detections(
            xyxy=np.array([bbox]), mask=np.array([final_mask]), 
            class_id=np.array([0]), confidence=np.array([1.0])
        )
        
        # Visualization
        ann_img = pipeline.mask_annotator.annotate(scene=img_bgr.copy(), detections=res_det)
        ann_img = pipeline.box_annotator.annotate(scene=ann_img, detections=res_det, labels=[f"{args.target_object} ({source})"])
        
        # Draw point prompt
        input_pts = np.median(prompt_points, axis=0)
        cv2.circle(ann_img, (int(input_pts[0]), int(input_pts[1])), 5, (0, 255, 0), -1)

        # Write files
        cv2.imwrite(os.path.join(args.output_dir, f"{args.target_object}_{this_num:05d}.jpg"), ann_img)
        mask_path = os.path.join(args.output_dir, "masks", f"mask_{this_num:05d}.png")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, final_mask * 255)

        all_json_results.append({
            "image_path": color_path, "mask_path": mask_path, "bbox": bbox,
            "obj_name": args.target_object, "source": source, "frame_id": this_num
        })
        
    master_json_output = os.path.join(args.output_dir, "detections_transfer.json")

    with open(master_json_output, 'w') as f:
        json.dump(all_json_results, f, indent=4)

    print(f"Tracking complete. Results saved to {master_json_output}")
    print(f"Total tracking opportunities Success Rate: {total_tracking_opportunities / (len(image_files)-1) * 100:.2f}%")
   


if __name__ == "__main__":
    main()