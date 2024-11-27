'''
Given input images folders, generate data loader to process inference in batch
Author: Xiao Wang
Data: 11.21.2024
'''

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F
import logging
import json
from src.smirk_generator import SmirkGenerator
from torch.utils.data.dataloader import default_collate
from pdb import set_trace as st
mediapipe_logger = logging.getLogger('mediapipe_logger')
mediapipe_logger.setLevel(logging.ERROR)
file_handler = logging.FileHandler('mediapipe_fails_log_0719.txt', mode='a')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
mediapipe_logger.addHandler(file_handler)
def process_image_paths(input_folder, output_folder):
    os.makedirs( output_folder, exist_ok=True)
    image_paths = []
    for root, dir, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
               
                input_path = os.path.join(root, file)
                dir_name = os.path.basename(root)   
                image_name = file
                output_dir = os.path.join(output_folder, dir_name)
                output_path = os.path.join(output_folder, dir_name, image_name)
                image_info = (input_path, output_dir , output_path) # 确保包含三个值
                image_paths.append(image_info)                

        for _, output_dir, _ in image_paths:
            os.makedirs(output_dir, exist_ok=True)        
        
    #write all the paths information
    to_save= True
    if to_save:
        with open("temp_image_paths.txt", "w") as f:
            for input_path, output_dir, output_path in image_paths:
                f.write(f"{input_path} {output_dir} {output_path}\n")  # 确保每行写入三个值
    # st()
        
    return image_paths
        


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], 
                        [center[0] - size / 2, center[1] + size / 2], 
                        [center[0] + size / 2, center[1] - size / 2]])
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform

class FaceDataset(Dataset):
    def __init__(self, image_paths, crop=False, image_size=224):
        self.image_paths = image_paths
        self.crop = crop
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path, output_dir, output_path = self.image_paths[idx]
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Error reading image {image_path}")
            orig_image_height, orig_image_width, _ = image.shape
            kpt_mediapipe = run_mediapipe(image)
            if kpt_mediapipe is None:
                mediapipe_logger.error(f"{image_path}")
                raise ValueError("Could not find landmarks for the image using mediapipe and cannot crop the face.")
            if self.crop:
                kpt_mediapipe = kpt_mediapipe[..., :2]
                tform = crop_face(image, kpt_mediapipe, scale=1.4, image_size=self.image_size)
                cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size), preserve_range=True).astype(np.uint8)
                cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T).T[:, :2]
            else:
                cropped_image, cropped_kpt_mediapipe = image, kpt_mediapipe
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = cv2.resize(cropped_image, (self.image_size, self.image_size))
            cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).float() / 255.0
            return cropped_image, cropped_kpt_mediapipe, (orig_image_height, orig_image_width), output_path
        except Exception as e:
            return None, None, None, None
        
def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with None values.
    """
    filtered_batch = []
    for item in batch:
        if item is not None and all(sub_item is not None for sub_item in item):
            filtered_batch.append(item)
    
    if not filtered_batch:
        return None  
    
    return default_collate(filtered_batch) 
def initialize_smirk_encoder(checkpoint_path, device):
    smirk_encoder = SmirkEncoder().to(device)
    checkpoint = torch.load(checkpoint_path)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()
    return smirk_encoder
def initialize_smirk_generator(checkpoint_path, device):
    from src.smirk_generator import SmirkGenerator
    smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(device)
    checkpoint = torch.load(checkpoint_path)
    checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k}
    smirk_generator.load_state_dict(checkpoint_generator)
    smirk_generator.eval()
    return smirk_generator

def save_image(output_path, grid):
    grid_numpy = grid.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
    grid_numpy = grid_numpy.astype(np.uint8)
    grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, grid_numpy)

def save_FLAME_parameters_json_batch(outputs, output_paths):
    """
    Save a batch of FLAME parameters to JSON files.

    Parameters:
    outputs (dict): The dictionary containing lists of FLAME parameters for each batch item.
    output_paths (list): A list of paths where each batch output file will be saved.
    """
    required_keys = ['pose_params', 'cam', 'shape_params', 'expression_params', 'eyelid_params', 'jaw_params']
    
    batch_size = len(output_paths)
    
    for i in range(batch_size):
        # Ensure the outputs contain the necessary keys for each item in the batch
        for key in required_keys:
            if key not in outputs or len(outputs[key]) != batch_size:
                raise ValueError(f"Key '{key}' not found in outputs or batch size mismatch")
        
        # Move the data to CPU, convert to numpy arrays, and then to lists for the i-th batch item
        cpu_outputs = {key: outputs[key][i].detach().cpu().numpy().tolist() for key in required_keys}

        # Save the outputs to a JSON file
        output_path = output_paths[i]
        with open(f"{output_path}.json", 'w') as outfile:
            json.dump(cpu_outputs, outfile, indent=4)


def main(args):
    device = args.device
    image_size = 224
    
    # Initialize models
    smirk_encoder = initialize_smirk_encoder(args.checkpoint, device)
    if args.use_smirk_generator:
        smirk_generator = initialize_smirk_generator(args.checkpoint, device)
    flame = FLAME().to(device)
    renderer = Renderer().to(device)

    #st()
    read_from_txt = True
    print(f"Processing images...")
    if read_from_txt:
    # Processing images
        all_image_paths= process_image_paths(args.input_path, args.output_path)
    else: 
    # # Read image paths and output paths from file
        with open(args.input_path, 'r') as f:
            all_image_paths = [line.strip().split() for line in f] #等于dataloader
        
    
    # Create dataset and dataloader
    dataset = FaceDataset(all_image_paths, crop=args.crop, image_size=image_size)
    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, collate_fn=custom_collate_fn)
    for batch in dataloader:
        if not batch:
            continue
        cropped_images, cropped_kpt_mediapipe_batch, orig_image_shapes, output_paths = batch
        cropped_images = cropped_images.to(device)
        outputs = smirk_encoder(cropped_images)

        # save the parameters
        save_FLAME_parameters_json_batch(outputs, output_paths)

        flame_outputs = flame(outputs)
        renderer_outputs = renderer(flame_outputs['vertices'], outputs['cam'],
                                    landmarks_fan=flame_outputs['landmarks_fan'], 
                                    landmarks_mp=flame_outputs['landmarks_mp'])

        rendered_imgs = renderer_outputs['rendered_img']
        grid = rendered_imgs if args.only_render else torch.cat([cropped_images, rendered_imgs], dim=3)

        if args.render_orig:
            grid = []
            for i in range(cropped_images.size(0)):
                orig_image_height, orig_image_width = orig_image_shapes[i]
                if args.crop:
                    rendered_img_numpy = (rendered_imgs[i].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                    rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(orig_image_height, orig_image_width), preserve_range=True).astype(np.uint8)
                    rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                else:
                    rendered_img_orig = F.interpolate(rendered_imgs[i].unsqueeze(0), (orig_image_height, orig_image_width), mode='bilinear').cpu()
                grid.append(torch.cat([cropped_images[i].unsqueeze(0), rendered_img_orig], dim=3))
            grid = torch.cat(grid, dim=0)

        if args.use_smirk_generator:
            if cropped_kpt_mediapipe_batch is None:
                raise ValueError("Could not find landmarks for the image using mediapipe and cannot create the hull mask for the smirk generator.")

            hull_masks = [create_mask(kpt, (image_size, image_size)) for kpt in cropped_kpt_mediapipe_batch]
            hull_masks = torch.stack([torch.from_numpy(mask).type(torch.float32).unsqueeze(0) for mask in hull_masks]).to(device)

            face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()
            rendered_mask = 1 - (rendered_imgs == 0).all(dim=1, keepdim=True).float()
            mask_ratio_mul, mask_ratio, mask_dilation_radius = 5, 0.01, 10
            tmask_ratio = mask_ratio * mask_ratio_mul

            npoints_batch, _ = masking_utils.mesh_based_mask_uniform_faces(renderer_outputs['transformed_vertices'], flame.faces_tensor, face_probabilities, tmask_ratio)

            pmask_batch = torch.zeros_like(rendered_mask)
            for i in range(npoints_batch.size(0)):
                npoints = npoints_batch[i]
                rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
                rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
                rbound = (npoints.size(1) * (1 / mask_ratio_mul) * (rscale ** rsing)).long()

                for bi in range(npoints.size(0)):
                    pmask_batch[i, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1

            extra_points = cropped_images * pmask_batch
            masked_imgs = [masking_utils.masking(cropped_images[i].unsqueeze(0), hull_masks[i].unsqueeze(0), extra_points[i].unsqueeze(0), mask_dilation_radius, rendered_mask=rendered_mask[i].unsqueeze(0)) for i in range(len(hull_masks))]
            masked_imgs = torch.cat(masked_imgs, dim=0)

            smirk_generator_input = torch.cat([rendered_imgs, masked_imgs], dim=1)
            reconstructed_imgs = smirk_generator(smirk_generator_input)

            if args.only_render:
                grid = reconstructed_imgs
            else:
                if args.render_orig:
                    grid = []
                    for i in range(cropped_images.size(0)):
                        orig_image_height, orig_image_width = orig_image_shapes[i]
                        if args.crop:
                            reconstructed_img_numpy = (reconstructed_imgs[i].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                            reconstructed_img_orig = warp(reconstructed_img_numpy, tform, output_shape=(orig_image_height, orig_image_width), preserve_range=True).astype(np.uint8)
                            reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        else:
                            reconstructed_img_orig = F.interpolate(reconstructed_imgs[i].unsqueeze(0), (orig_image_height, orig_image_width), mode='bilinear').cpu()
                        grid.append(torch.cat([grid[i].unsqueeze(0), reconstructed_img_orig], dim=3))
                    grid = torch.cat(grid, dim=0)
                else:
                    grid = torch.cat([grid, reconstructed_imgs], dim=3)

        for i in range(cropped_images.size(0)):
            output_path = output_paths[i]
            save_image(output_path, grid[i])
          



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='temp_image_paths.txt', help='Path to the input image paths file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--output_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')
    parser.add_argument('--only_render', action='store_true', help='Only render the 3D image without original image')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for processing images')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading') # only 0 works for no bug

    args = parser.parse_args()
    main(args)