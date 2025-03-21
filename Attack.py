import os
import torch
import argparse
import numpy as np
from torchvision import models, transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import requests
from ultralytics import YOLO
from torchvision.ops import nms  
import json
import timm
import sys
import pdb
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 
import random
import math
import scipy.linalg  

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, input):
                return self.attention(input, input, input)[0]  


def load_swin_model(device):
    swin_model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    swin_model.eval()  
    swin_model = swin_model.to(device)  

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),      
        transforms.CenterCrop(224),        
        transforms.ToTensor(),              
        transforms.Normalize(               
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return swin_model, preprocess



def register_hook(model, layer_name):
    features = []

    def hook(module, input, output):
   
        features.append(output.detach())   

   
    named_modules = dict([*model.named_modules()])   

   
    if layer_name in named_modules:
        layer = named_modules[layer_name]
        hook_handle = layer.register_forward_hook(hook)
    else:
        raise KeyError(f"Layer name '{layer_name}' not found. Available layers: {list(named_modules.keys())}")


    return features, hook_handle



def extract_swin_features(image, swin_model, preprocess, device, layer_name='features.7.1.mlp.3', boxes=None):
     
     if isinstance(image, Image.Image):  
         image = preprocess(image).unsqueeze(0).to(device)
     elif isinstance(image, torch.Tensor):  
         if image.dim() == 3:   
             image = image.unsqueeze(0).to(device)
     else:
         raise TypeError(f"Expected PIL Image or Tensor, but got {type(image)}")

     
     image = image.clone().detach().requires_grad_()

     swin_model = swin_model.to(device)

     
     features, hook_handle = register_hook(swin_model, layer_name)

     
     swin_model(image)

     
     full_features = features[0]
     hook_handle.remove()   
      
     full_features.requires_grad_()

           if boxes is not None:
         roi_features = []
         for box in boxes:
             x1, y1, x2, y2 = map(int, box[:4])
             roi_feature = full_features[:, :, y1:y2, x1:x2]
             roi_features.append(roi_feature)
         return roi_features   
     return full_features   



def get_clip_model(diffusion_model):
    print(f"diffusion_model.text_encoder type: {type(diffusion_model.text_encoder)}")
    print(f"diffusion_model.tokenizer type: {type(diffusion_model.tokenizer)}")
    return diffusion_model.text_encoder, diffusion_model.tokenizer


file_path = 'imagenet_class_index.json'
 
with open(file_path, 'r') as f:
    labels = json.load(f)


 
def get_class_name(class_idx):
    return labels[str(class_idx)][2]   

 
def plot_boxes_on_image(image, boxes):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image.squeeze(0).cpu())
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    
    for box in boxes:
        x1, y1, x2, y2, score = box
        width = x2 - x1
        height = y2 - y1

        
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

         
        ax.text(x1, y1, f'{score:.2f}', fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()


def get_yolov5_boxes(ref_image, yolo_model, iou_threshold=0.5):
   
    if isinstance(ref_image, Image.Image):
        pass   
    elif isinstance(ref_image, torch.Tensor):
        ref_image = transforms.ToPILImage()(ref_image.squeeze(0).cpu())   
       elif isinstance(ref_image, np.ndarray):
        ref_image = Image.fromarray(ref_image)
    else:
        raise ValueError("Expected ref_image to be a PIL.Image object")
         
    print(f"Converted ref_image type: {type(ref_image)}")

     
    transform = transforms.ToTensor()
    ref_image = transform(ref_image).unsqueeze(0)

     
    device = torch.device("cpu")
    ref_image = ref_image.to(device)
    yolo_model.to(device).eval()

     
    with torch.no_grad():
        results = yolo_model(ref_image)

    if isinstance(results, list):
        results = results[0]

    boxes = results.boxes

    if boxes is not None and len(boxes) > 0:
        coordinates = boxes.xyxy[0]
        scores = boxes.conf[0] if boxes.conf is not None else torch.ones(coordinates.shape[0], device=device)

         
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        if coordinates.dim() == 1:
            coordinates = coordinates.unsqueeze(0)

        print(f"Adjusted coordinates shape: {coordinates.shape}")
        print(f"Adjusted scores shape: {scores.shape}")

        
        boxes_and_scores = torch.cat((coordinates, scores.unsqueeze(1)), dim=1)

      
        selected_indices = nms(boxes_and_scores[:, :4], boxes_and_scores[:, 4], iou_threshold=iou_threshold)
        selected_boxes = boxes_and_scores[selected_indices]

    
        plot_boxes_on_image(ref_image, selected_boxes.cpu().numpy())

        return selected_boxes.cpu().numpy()
    else:
        print("No bounding boxes detected.")
        return None

def perturbation_on_bounding_boxes(generated_image, boxes, perturb_values, epsilon, device):
    print(f"boxes type: {type(boxes)}")
    print(f"boxes value: {boxes}")
    
    
    if not isinstance(boxes, (list, torch.Tensor, np.ndarray)):
        raise ValueError(f"Expected boxes to be a list, tensor, or ndarray, but got {type(boxes)}")

  
    if not isinstance(epsilon, float):
        print(f"epsilon type before correction: {type(epsilon)}")
        epsilon = float(epsilon)
        print(f"epsilon corrected to: {epsilon}")

    perturbed_image = generated_image.clone()   
    print(f"Starting loop over boxes with {len(boxes)} elements.")

    for i, box in enumerate(boxes):
        print(f"Processing box {i}: {box}", flush=True)  

        
        if isinstance(box, float):
            print(f"Skipping box {i} because it is a float: {box}")
            continue

         
        if isinstance(box, torch.Tensor):
            box = box.tolist()   

        
        if not isinstance(box, (list, tuple, np.ndarray)):
            raise TypeError(f"Expected box to be a list, tuple, or ndarray, but got {type(box)}: {box}")
        
        if len(box) != 4:
            print(f"Skipping box {i} due to invalid length: {len(box)}")
            continue

        
        x1, y1, x2, y2 = map(int, box[:4])

         
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(generated_image.size(3), x2), min(generated_image.size(2), y2)

         
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid box: ({x1}, {y1}, {x2}, {y2}), skipping.")
            continue

         
        region = generated_image[:, :, y1:y2, x1:x2]
        
         
        if region.numel() == 0:
            print(f"Skipping empty region for box {i}: ({x1}, {y1}, {x2}, {y2})")
            continue

         
        if isinstance(perturb_values, torch.Tensor):
             
            perturb_region = perturb_values[:, :, y1:y2, x1:x2]
            if perturb_region.shape != region.shape:
                raise ValueError(f"Perturbation values shape mismatch: {perturb_region.shape} vs {region.shape}")
        else:
             
            perturb_region = torch.full_like(region, fill_value=perturb_values, device=device)

         
        print(f"Box coordinates: ({x1}, {y1}, {x2}, {y2}), Region mean value: {region.mean().item()}")

        
        region = region + epsilon * perturb_region   
        region = region.clamp(0, 1)   

        
        perturbed_image[:, :, y1:y2, x1:x2] = region

    return perturbed_image


def adjust_channels(feature, target_channels):
    current_channels = feature.size(1)
    if current_channels < target_channels:
         
        padding = target_channels - current_channels
        padding_tensor = torch.zeros(feature.size(0), padding, feature.size(2), feature.size(3), device=feature.device)
        feature = torch.cat([feature, padding_tensor], dim=1)
    elif current_channels > target_channels:
         
        feature = feature[:, :target_channels, :, :]
    return feature

    


def resize_feature_map(feature, target_height, target_width):
    return F.interpolate(feature, size=(target_height, target_width), mode='bilinear', align_corners=False)

 
def ensure_4d(feature):
    if len(feature.shape) == 3:
        feature = feature.unsqueeze(0) 



device = torch.device("cpu")
maskrcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn.eval()  
maskrcnn = maskrcnn.to(device)  

import torchvision.ops.boxes as box_ops

def preprocess_image(image, device):
    if isinstance(image, Image.Image):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image).to(device)
    elif isinstance(image, torch.Tensor):
        return image.to(device)
    elif isinstance(image, list):
         
        return [img.to(device) for img in image]
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")




def get_mask_from_boxes(image, model, score_threshold, device='cpu'):

    print(f"Inside get_mask_from_boxes - score_threshold: {score_threshold} (type: {type(score_threshold)})")
    print(f"Inside get_mask_from_boxes - device: {device} (type: {type(device)})")

     
    if isinstance(image, list):
        image = [img.cpu() for img in image]
    else:
        image = image.cpu()

     
    model = model.cpu()

     
    if isinstance(image, list):
        for idx, img in enumerate(image):
            print(f"image {idx} device: {img.device}")
    else:
        print(f"image device: {image.device}")

     
    model_device = next(model.parameters()).device
    print(f"model device: {model_device}")  

     
    if isinstance(score_threshold, str):
        if score_threshold.isdigit() or score_threshold.replace('.', '', 1).isdigit():
            score_threshold = float(score_threshold)
        else:
            raise ValueError(f"Invalid score_threshold value: {score_threshold}. It must be a number.")
    
    if not isinstance(score_threshold, (float, int)):
        raise TypeError(f"Expected score_threshold to be a float or int, but got {type(score_threshold)}")

     
    if isinstance(image, torch.Tensor) and len(image.shape) == 3:   [C, H, W]
        image = image.unsqueeze(0)   
       with torch.no_grad():
        predictions = model(image)   
    
    boxes = predictions[0]['boxes']  
    scores = predictions[0]['scores']   
    masks = predictions[0]['masks']   

     
    boxes_cpu = boxes.cpu()
    scores_cpu = scores.cpu()
    masks_cpu = masks.cpu()
    print(f"Prediction done. Boxes: {boxes.shape}, Scores: {scores.shape}, Masks: {masks.shape}")

     
    keep = scores_cpu > score_threshold   

     
    if keep.sum() == 0:
        print("No boxes passed the score threshold.")
        return []

     
    boxes_cpu = boxes_cpu[keep]
    masks_cpu = masks_cpu[keep]

     
    mask_list = []
    for idx in range(len(boxes_cpu)):
        mask = masks_cpu[idx, 0]  
        mask = torch.sigmoid(mask)   
        mask = mask > 0.4   
        mask = mask.unsqueeze(0).unsqueeze(0)  
         
        target_height, target_width = image.shape[2:]   
        mask = F.interpolate(
            mask.float(),
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )
        mask = mask.squeeze(0).squeeze(0)   
        mask_list.append(mask)

    print(f"Number of masks returned: {len(mask_list)}")
    return mask_list


def box_count(image, box_size):
    
    count = 0
    for y in range(0, image.shape[0], box_size):
        for x in range(0, image.shape[1], box_size):
            
            if np.any(image[y:y + box_size, x:x + box_size]):
                count += 1
    return count



def calculate_fractal_dimension(image, box_sizes=None):
 
    if box_sizes is None:
        box_sizes = [2**i for i in range(1, int(np.log2(min(image.shape))) + 1)]
    
    counts = []
    for box_size in box_sizes:
        counts.append(box_count(image, box_size))
    
    box_sizes = np.array(box_sizes)
    counts = np.array(counts)

    box_sizes = box_sizes[box_sizes > 0]
    counts = counts[counts > 0]

    if len(box_sizes) == 0 or len(counts) == 0:
        print("Error: box_sizes or counts have no valid values.")
        return 0

   
    log_box_sizes = np.log(box_sizes).astype(np.float64)
    log_counts = np.log(counts).astype(np.float64)

    
    valid_idx = np.isfinite(log_box_sizes) & np.isfinite(log_counts)
    log_box_sizes = log_box_sizes[valid_idx]
    log_counts = log_counts[valid_idx]

    if len(log_box_sizes) == 0 or len(log_counts) == 0:
        print("Error: No valid data points after filtering.")
        return 0 

     
    log_box_sizes_std = np.std(log_box_sizes)
    log_counts_std = np.std(log_counts)

     
    if log_box_sizes_std > 0:
        log_box_sizes = (log_box_sizes - np.mean(log_box_sizes)) / log_box_sizes_std
    else:
        log_box_sizes = log_box_sizes - np.mean(log_box_sizes)

    if log_counts_std > 0:
        log_counts = (log_counts - np.mean(log_counts)) / log_counts_std
    else:
        log_counts = log_counts - np.mean(log_counts)

     
    try:
        p = np.polyfit(log_box_sizes, log_counts, 1)   
        return p[0]   
    except np.linalg.LinAlgError:
        print("SVD did not converge. Trying with regularization...")
        
        A = np.vstack([log_box_sizes, np.ones_like(log_box_sizes)]).T
       
        valid_idx = np.isfinite(A).all(axis=1) & np.isfinite(log_counts)
        A = A[valid_idx]
        log_counts = log_counts[valid_idx]

        if len(A) == 0 or len(log_counts) == 0:
            print("Error: No valid data points for lstsq.")
            return 0   

        m, c = scipy.linalg.lstsq(A, log_counts)[0]   使用 scipy.linalg.lstsq
        return m   


def add_perturbation_to_brightness_in_masked_area(
    generated_image, perturb_values, original_boxes, epsilon, device, mask_list,   
    brightness_weight=1.0, high_freq_weight=1.0, fractal_weight=1.0, style_weight=1.0  
):  
    if len(generated_image.shape) == 4:  
        generated_image = generated_image[0]  

    generated_image_np = generated_image.cpu().detach().numpy().transpose(1, 2, 0)  
    generated_image_hsv = cv2.cvtColor((generated_image_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)  
    h_channel, s_channel, value_channel = cv2.split(generated_image_hsv)  

    if len(perturb_values) < len(mask_list):  
        perturb_values = [0.05] * len(mask_list)    
    elif len(perturb_values) > len(mask_list):  
        perturb_values = perturb_values[:len(mask_list)]   

     
    if not perturb_values:  
        print("Warning: perturb_values is empty, using default perturbation value.")  
        perturb_values = [0.001] * len(mask_list)   
    
    def get_high_frequency_mask(image):  
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   转为灰度图  
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  
        grad_mag = cv2.magnitude(grad_x, grad_y)   
         
        high_freq_mask = grad_mag > np.percentile(grad_mag, 10)  
        return high_freq_mask.astype(np.uint8)  

    high_freq_mask = get_high_frequency_mask(generated_image_np)  

    value_channel_int = value_channel.astype(np.int32)  
    max_brightness_value = 190   
    perturbation_value = int(epsilon * perturb_values[0] * 255)  

    for idx, mask in enumerate(mask_list):  
        mask = mask.bool()  

           
        if len(mask.shape) == 2:    
            mask = mask.unsqueeze(0).unsqueeze(0)   
        elif len(mask.shape) == 3:   
            mask = mask.unsqueeze(0)   

         
        mask_resized = torch.nn.functional.interpolate(  
            mask.float(),   
            size=(value_channel.shape[0], value_channel.shape[1]),  
            mode='bilinear',  
            align_corners=False  
        ).squeeze(0).squeeze(0) > 0.6   

        if mask_resized.sum().item() == 0:  
            print(f"Mask {idx} is empty, skipping...")  
            continue  

          
        mask_np = mask_resized.cpu().numpy().astype(bool)  
        value_channel_int[mask_np] = np.clip(value_channel_int[mask_np] + perturbation_value * brightness_weight, 0, max_brightness_value)  

        
        high_freq_area_in_mask = high_freq_mask.astype(bool) & mask_np   
        value_channel_int[high_freq_area_in_mask] = np.clip(value_channel_int[high_freq_area_in_mask] + perturbation_value * high_freq_weight, 0, max_brightness_value)  

        
        fractal_dimension = calculate_fractal_dimension(generated_image_np)  
        if fractal_dimension is not None:  
            fractal_mask = mask_np & (fractal_dimension > 1.5)  
            h_channel[fractal_mask] = (h_channel[fractal_mask] + perturbation_value * fractal_weight) % 180     

           
        if style_weight > 0:  
            style_s_channel = calculate_gram_matrix(s_channel)    
            s_channel[mask_np] = np.clip(  
                (1 - style_weight) * s_channel[mask_np] + style_weight * style_s_channel[mask_np],  
                0, 255  
            )  

      
    value_channel = value_channel_int.astype(np.uint8)  
    h_channel = h_channel.astype(np.uint8)  
    s_channel = s_channel.astype(np.uint8)  
    generated_image_hsv_perturbed = cv2.merge([h_channel, s_channel, value_channel])  
    generated_image_rgb_perturbed = cv2.cvtColor(generated_image_hsv_perturbed, cv2.COLOR_HSV2RGB)  

    generated_image_rgb_perturbed = np.clip(generated_image_rgb_perturbed, 0, 255).astype(np.uint8)  

    generated_image = torch.tensor(generated_image_rgb_perturbed).permute(2, 0, 1).float() / 255.0  
    generated_image = generated_image.to(device)  

    return generated_image     



def scale_invariant_target_image(ref_image, scales=[0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]):
    scaled_images = []
    
    if isinstance(ref_image, Image.Image):
     
        ref_image = transforms.ToTensor()(ref_image).unsqueeze(0)
    

    original_size = ref_image.shape[2:]


    for scale in scales:

        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        resized_image = torch.nn.functional.interpolate(ref_image, size=new_size, mode='bilinear', align_corners=False)
        scaled_images.append(resized_image)
    
    return scaled_images

def calculate_perturbation(generated_image, scaled_image, epsilon, device):
 
    generated_image = generated_image.to(device)
    scaled_image = scaled_image.to(device)


    if len(generated_image.shape) == 3: 
        generated_image = generated_image.unsqueeze(0)  
    
    
    if len(scaled_image.shape) == 3:  
        scaled_image = scaled_image.unsqueeze(0)  


    scaled_image_resized = F.interpolate(
        scaled_image, 
        size=(generated_image.shape[2], generated_image.shape[3]),
        mode='bilinear', 
        align_corners=False
    )

    
    l2_loss = F.mse_loss(generated_image, scaled_image_resized)
    
     
    perturbation = epsilon * l2_loss.item()
    
    return perturbation

def apply_scaled_perturbations(generated_image, perturb_values, mask_list, brightness_weight, high_freq_weight,  fractal_weight):
    for mask_idx, mask in enumerate(mask_list):
        generated_image = add_perturbation_to_brightness_in_masked_area(
            generated_image, perturb_values, None, epsilon=0.002, device=generated_image.device,
            mask_list=[mask], brightness_weight=brightness_weight, high_freq_weight=high_freq_weight, fractal_weight = fractal_weight
        )
    return generated_image



def gram_matrix(x):
    b, c, h, w = x.size()
    x = x.view(b, c, -1)   
    gram = torch.bmm(x, x.transpose(1, 2))   
    return gram / (c * h * w)   

def calculate_loss(generated_image, ref_image, gen_box, ref_box, lambda_inf, lambda_cosine, lambda_style=1.0, layers=None):
  
    if not generated_image.requires_grad:
        generated_image = generated_image.requires_grad_(True) 

  
    if isinstance(ref_image, Image.Image):
        ref_image = transforms.ToTensor()(ref_image).unsqueeze(0).to(generated_image.device)  

 
    if generated_image.dim() == 3:  
        generated_image = generated_image.unsqueeze(0)   

        if ref_image.dim() == 2:   
        ref_image = ref_image.unsqueeze(0).unsqueeze(0)   [1, 1, height, width]
    elif ref_image.dim() == 3: 
        ref_image = ref_image.unsqueeze(0)   
    ref_image_resized = F.interpolate(ref_image, size=generated_image.shape[2:], mode='bilinear', align_corners=False)

 


     
    generated_image = generated_image.to(device)  
    ref_image = ref_image.to(device)  
    ref_image_resized = ref_image_resized.to(device)  

    l_inf_loss = torch.max(torch.abs(generated_image - ref_image_resized))

     
    generated_image_flat = generated_image.reshape(generated_image.size(0), -1)  
    ref_image_flat = ref_image_resized.reshape(ref_image_resized.size(0), -1)   
    cosine_similarity = F.cosine_similarity(generated_image_flat, ref_image_flat, dim=1)
    cosine_loss = 1 - cosine_similarity.mean()   

    style_loss = 0
    if layers is not None:
        for layer in layers:

            gen_features = layer(generated_image)
            ref_features = layer(ref_image_resized)

            gen_gram = gram_matrix(gen_features)
            ref_gram = gram_matrix(ref_features)
            

            style_loss += torch.mean((gen_gram - ref_gram) ** 2)

    total_loss = lambda_inf * l_inf_loss + lambda_cosine * cosine_loss + lambda_style * style_loss

    return l_inf_loss, cosine_loss, style_loss, total_loss




def perturbation_until_target_confidence(
    generated_image,
    target_text,
    target_confidence,
    perturb_steps,
    clip_model,
    clip_processor,
    swin_model,
    preprocess,
    epsilon,
    lambda_inf,
    lambda_cosine,
    lambda_style,
    perturb_start_step,
    transparency_factor,
    diffusion_steps,
    sd_pipe,
    args,
    yolo_model,
    ref_image,
    brightness_weight=0.5,   
    high_freq_weight=0.8,
    fractal_weight=0.8      
):

    if isinstance(generated_image, Image.Image):
        generated_image = transforms.ToTensor()(generated_image).unsqueeze(0).to("cpu")
    else:
        generated_image = generated_image.to("cpu")   

    clean_image_pil = transforms.ToPILImage()(generated_image.squeeze(0).cpu())
    clean_image_pil.save("clean_image.png")

    original_clean_image = generated_image.clone()

    generated_image.requires_grad_()   

    device = 'cpu'   
    swin_model, preprocess = load_swin_model(device=device)

    optimizer = torch.optim.Adam([generated_image], lr=0.01)

    
    ref_boxes = get_yolov5_boxes(ref_image, yolo_model)
    original_boxes = get_yolov5_boxes(generated_image, yolo_model)

     
    scaled_images = scale_invariant_target_image(ref_image, scales=[0.3, 0.5, 0.8, 1.0])

    
    diffusion_steps = 50
    lambda_inf = 20
    lambda_cosine = 20
    epsilon = 0.03
    current_confidence = 0.0  
    step = 0
    perturb_start_step = int(diffusion_steps * 0.9)
    lambda_style=20

    while current_confidence < target_confidence and step < diffusion_steps:
        step += 1
        print(f"Entering diffusion steps loop")
        print(f"Current Confidence: {current_confidence}, Target Confidence: {target_confidence}")
        print(f"Diffusion Step: {step}/{diffusion_steps}")
        
        if current_confidence < 0.7:
            target_confidence = 0.7
        
        current_confidence += 0.000001
        generated_image.requires_grad_()   Ensure gradient tracking is enabled for the generated image
        
        if step >= perturb_start_step:
          
            score_threshold = 0.3  
            print(f"score_threshold: {score_threshold} (type: {type(score_threshold)})")
            print(f"device: {device} (type: {type(device)})")
            
            generated_image = generated_image.to(device)   
            mask_list = get_mask_from_boxes(generated_image, maskrcnn, score_threshold=score_threshold, device=device) 
            perturb_values = [random.uniform(1, 5) for _ in mask_list]
            
             Add perturbation to brightness in the masked area
            generated_image = add_perturbation_to_brightness_in_masked_area(
                generated_image, perturb_values, original_boxes, epsilon, device, mask_list, brightness_weight=brightness_weight, fractal_weight=fractal_weight, high_freq_weight=high_freq_weight
            )
            
            generated_image = generated_image.clamp(0, 255)
            
             Show perturbed image every 10 steps
            if step % 5 == 0:
                 Ensure generated image is on CPU and properly formatted
                generated_image_np = generated_image.detach().cpu().numpy()
                if len(generated_image_np.shape) == 4:   [B, C, H, W]
                    generated_image_np = generated_image_np[0]
                generated_image_np = np.transpose(generated_image_np, (1, 2, 0))
                if generated_image_np.max() <= 1.0:
                    generated_image_np = (generated_image_np * 255).astype(np.uint8)
                else:
                    generated_image_np = np.clip(generated_image_np, 0, 255).astype(np.uint8)
                
                 Display the perturbed image
                plt.figure(figsize=(6, 6))
                plt.imshow(generated_image_np)
                plt.axis('off')
                plt.title(f"Step {step}: Perturbed Image")
                plt.show()
            
             使用不同尺度的参考图像与生成图像计算扰动值
            perturb_values = []
            for scaled_image in scaled_images:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                perturb_values.append(calculate_perturbation(generated_image, scaled_image, epsilon, device))
    
            
             
            generated_image = apply_scaled_perturbations(generated_image, perturb_values, mask_list, brightness_weight, high_freq_weight, fractal_weight)
            generated_image = generated_image.clamp(0, 255)
            
                         
            if isinstance(ref_image, Image.Image):
                transform = transforms.ToTensor()   
                ref_image = transform(ref_image)
            
           
            ref_image = ref_image.to(device)
   
            if isinstance(generated_image, torch.Tensor):
          
                images = [generated_image.to(device)]
            elif isinstance(generated_image, list):
               
                images = [img.to(device) for img in generated_image]
            else:
                raise TypeError(f"Unsupported type for generated_image: {type(generated_image)}")
            
                     for i in range(len(images)):
                if len(images[i].shape) == 2:   
                    print(f"Converting image {i} from gray to RGB format.")
                    images[i] = images[i].unsqueeze(0)   
                    images[i] = images[i].repeat(3, 1, 1)   
            

            for img in images:
                if len(img.shape) != 3 or img.shape[0] != 3:
                    raise ValueError(f"Each image must have shape [3, H, W], but got {img.shape}")
            

            mask_list = get_mask_from_boxes(images, maskrcnn, device)
            
            mask_list = get_mask_from_boxes(generated_image, maskrcnn, score_threshold=score_threshold, device=device) 
 
            if not mask_list:
                print("Debug Info: mask_list is empty.")
                print(f"Generated images shape: {[img.shape for img in images]}")
                
   
                detections = maskrcnn(images) 
                print("Detections:", detections)
            

                default_mask = torch.zeros_like(images[0][0], device=device)   
                mask_list = [default_mask]
                print("Using default mask as fallback.")
            else:
           
                mask = mask_list[0]
            
  
            mask = mask.to('cpu')
            generated_image = generated_image.to('cpu')
            mask = mask.unsqueeze(0).repeat(3, 1, 1)   [3, H, W]
                        
            masked_generated_image = generated_image * mask
            

            ref_image = ref_image.to('cpu')
            masked_ref_image = ref_image * mask
            
            
            
    
            perturbation_signal = masked_generated_image - masked_ref_image              
            
            
            perturbation_signal = perturbation_signal.detach().cpu().numpy()   
            if len(perturbation_signal.shape) == 4:  
                perturbation_signal = perturbation_signal[0]
            perturbation_signal = np.transpose(perturbation_signal, (1, 2, 0))  
            
             
            perturbation_signal = (perturbation_signal - perturbation_signal.min()) / (perturbation_signal.max() - perturbation_signal.min())
            
             
            plt.figure(figsize=(6, 6))
            plt.imshow(perturbation_signal)
            plt.axis('off')
            plt.title("Perturbation Signal in Masked Area")
            plt.show()

    
             
            if step == 1:   Only set once at the start
                generated_image.requires_grad_()
            
             
            total_loss = torch.zeros((), device=device)
            
            for gen_box, ref_box in zip(original_boxes, ref_boxes):
                 
                
                l_inf_loss, cosine_loss, total_loss, style_loss = calculate_loss(generated_image, ref_image, gen_box, ref_box, lambda_inf, lambda_cosine, layers=None)
    
                total_loss += l_inf_loss + cosine_loss
                total_loss = lambda_inf * l_inf_loss + lambda_cosine * cosine_loss + lambda_style * style_loss
    
            
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
             
            current_confidence += min(0.001, total_loss / 100000.0)
    
            print(f"Step {step}/{diffusion_steps}, current confidence: {current_confidence:.4f}")
                
        if current_confidence < target_confidence:
            print(f"Confidence is too low, continuing to next iteration.")
        else:
            print(f"Target confidence reached, stopping.")


    if isinstance(generated_image, Image.Image):
        generated_image.requires_grad_()
        
        generated_image = transforms.ToTensor()(generated_image).unsqueeze(0).to(device)
        generated_image.requires_grad_()
       
    elif not isinstance(generated_image, torch.Tensor):
        raise TypeError(f"Unsupported type for generated_image: {type(generated_image)}")
       

       if isinstance(original_clean_image, Image.Image):
        original_clean_image = transforms.ToTensor()(original_clean_image).unsqueeze(0).to(device)
        generated_image.requires_grad_()
    generated_image_tensor = generated_image.squeeze(0).cpu().clamp(0, 1) 
    generated_image.requires_grad_()
    

    original_clean_image_normalized = (original_clean_image - original_clean_image.min()) / (original_clean_image.max() - original_clean_image.min() + 1e-8)
    generated_image.requires_grad_()

    generated_image_normalized = (generated_image_tensor - generated_image_tensor.min()) / (generated_image_tensor.max() - generated_image_tensor.min() + 1e-8)
    generated_image.requires_grad_()

    
    generated_image_pil = transforms.ToPILImage()(generated_image_tensor.clamp(0, 1).squeeze(0).cpu())
    

    generated_image_tensor = transforms.ToTensor()(generated_image_pil).unsqueeze(0).to("cuda")
    

    yolo_model.conf = 0.5  
    
 
    yolo_model.eval()
    

    with torch.no_grad():
        yolo_results = yolo_model(generated_image_tensor)
    
    

    detected_boxes = None
    

    if hasattr(yolo_results, "boxes") and yolo_results.boxes is not None:
        detected_boxes = yolo_results.boxes.xyxy.cpu().numpy()   
        boxes = yolo_results.boxes
        confidences = boxes.conf   
        
    elif hasattr(yolo_results, "pred") and len(yolo_results.pred) > 0:
        detected_boxes = yolo_results.pred[0].cpu().numpy()
    
    
    if detected_boxes is not None and len(detected_boxes) > 0:
        print("Detected boxes:", detected_boxes)   输出检测框信息
        formatted_boxes = []
        
                for box in boxes:
            x1, y1, x2, y2, conf, cls = box   
            class_name = names[int(cls)]   
            
        
            print(f"Box coordinates: ({x1}, {y1}), ({x2}, {y2}) with confidence {conf} for class {class_name}")
            
                      formatted_boxes.append([x1, y1, x2, y2, conf, class_name])
    
        plot_boxes_with_matplotlib(generated_image_pil, formatted_boxes)

    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    clean_image_pil = transforms.ToPILImage()(original_clean_image.squeeze(0).cpu())

   

    if isinstance(ref_image, torch.Tensor):

        ref_image_tensor = ref_image.unsqueeze(0)   形状变为 [1, C, H, W]
    else:
                ref_image_tensor = transforms.ToTensor()(ref_image).unsqueeze(0)       

    print("ref_image_tensor dimensions:", ref_image_tensor.shape)
    
  
    if ref_image_tensor.ndimension() == 5:
        ref_image_tensor = ref_image_tensor.squeeze(0)   
        ref_image_tensor = ref_image_tensor.squeeze(0)   
    
     
    elif ref_image_tensor.ndimension() == 4:
        ref_image_tensor = ref_image_tensor.squeeze(0)   
    
     检查维度
    if ref_image_tensor.ndimension() not in [2, 3]:
        raise ValueError(f"Invalid tensor dimension: {ref_image_tensor.ndimension()}. Expected 2 or 3 dimensions.")

    
    ref_image_pil = transforms.ToPILImage()(ref_image_tensor)
    
    
    
    axes[0].imshow(clean_image_pil)   
    axes[0].set_title("Clean Image")
    
axes[1].imshow(np.array(ref_image_pil))   
axes[1].set_title("Reference Image")
    
    axes[2].imshow(np.array(generated_image_pil))   
    axes[2].set_title("Generated Image")
    
    
    plt.show()

    return  current_confidence


 设置种子
def set_seed(seed):
    torch.cuda.manual_seed(seed)   
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False   
 主函数
def main():
     解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', default=512, type=int, help='Input image resized resolution')
    parser.add_argument('--save_dir', default="output", type=str, help='Where to save the generated images')
    parser.add_argument('--pretrained_diffusion_path', default="pretrained/", type=str, help="Path to the pretrained diffusion model")
    parser.add_argument('--diffusion_steps', default=50, type=int, help='Total DDIM sampling steps')
    parser.add_argument('--guidance_scale', default=17, type=float, help='Guidance scale of diffusion models')
    parser.add_argument('--prompt', type=str, required=True, help="Text prompt to generate the image")
    parser.add_argument('--target_text', type=str, required=True, help="Target text description for the perturbation")
    parser.add_argument('--perturb_steps', default=400, type=int, help='Number of perturbation steps')  
    parser.add_argument('--target_confidence', default=0.8, type=float, help="Target confidence threshold")
    parser.add_argument('--epsilon', default=0.01, type=float, help="Perturbation strength")   
    parser.add_argument('--perturb_start_step', default=100, type=int, help="Starting step for applying perturbations")   
    parser.add_argument('--transparency_factor', default=1, type=float, help="Transparency factor for blending")  
    parser.add_argument('--lambda_inf', default=20, type=float, help="Weight for L2 loss in the perturbation")   
    parser.add_argument('--lambda_cosine', default=20, type=float, help="Weight for cosine similarity loss in the perturbation")   
    parser.add_argument('--lambda_style', default=20, type=float, help=" Weight for style loss in the perturbation")   
    parser.add_argument('--A', default=200, type=float, help="Weight for jiaocha loss in the perturbation")   
    parser.add_argument('--brightness_weight', default=0.5, type=float, help="Weight loss in the perturbation")   
    parser.add_argument('--high_freq_weight', default=0.5, type=float, help="Weight loss in the perturbation")   
    parser.add_argument('--seed', default=10, type=int, help="Random seed for reproducibility")   添加随机种子参数
    args = parser.parse_args()   
    set_seed(args.seed)

     加载模型并设置调度器
    sd_pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_diffusion_path, torch_dtype=torch.float16).to("cuda")
    sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(11)  45cat    

    generated_image = sd_pipe(prompt=args.prompt, guidance_scale=args.guidance_scale, num_inference_steps=int(args.diffusion_steps)).images[0]

    set_seed(50)   50dog 18eagle 45 fish 5chicken 5A colorful butterfly spread wings

    ref_image = sd_pipe(prompt=args.target_text, guidance_scale=args.guidance_scale, num_inference_steps=int(args.diffusion_steps)).images[0]  

    
    clip_model, clip_processor = get_clip_model(sd_pipe)   
    yolo_model = YOLO('yolo5/yolov5su.pt')   


   
    swin_model, preprocess = load_swin_model(device='cuda')


    final_confidence = perturbation_until_target_confidence(
        generated_image,
        args.target_text,
        args.target_confidence,   
        args.perturb_steps,   
        args.diffusion_steps,
        clip_model,
        clip_processor,
        swin_model,   
        preprocess,
        args.epsilon,   
        args.lambda_inf,   
        args.lambda_cosine,   
        args.lambda_style,
        args.perturb_start_step,   
        args.transparency_factor,   
        args,   
        sd_pipe,  
        yolo_model,
        ref_image,
    )
    

if __name__ == "__main__":
    main()
