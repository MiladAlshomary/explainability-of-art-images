from PIL import Image
import numpy as np
import os
import argparse # Added for command-line argument parsing
import json # Added for saving results to JSON
import torch

import torch
import requests
from PIL import Image
from lavis.models import load_model_and_preprocess

def image_captioning(folder_path, model_name, model_type="caption_coco_flant5xl", num_captions=5):
    """
    Loads images from a folder, generates captions for each using a specified BLIP model.

    Args:
        folder_path (str): Path to the folder containing images.
        model_name (str): Name of the BLIP model to use (e.g., "blip2_t5", "blip_caption").
        model_type (str): Type of the model (e.g., "caption_coco_flant5xl", "base_coco").
        num_captions (int): Number of captions to generate for nucleus sampling.

    Returns:
        dict: A dictionary where keys are image file paths and values are
              dictionaries containing 'beam_search' and 'nucleus_sampling' captions.
    """
    # Set the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captions_dict = {}

    # Load the pre-trained BLIP-2 captioning model and its associated image processor
    model, vis_processors, _ = load_model_and_preprocess(
        name=model_name,
        model_type=model_type,
        is_eval=True,  # Set to evaluation mode for inference
        device=device  # Specify the device
    )

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return captions_dict

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                raw_image = Image.open(image_path).convert('RGB')
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

                caption_beam = model.generate({"image": image})
                caption_nucleus = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=num_captions)

                captions_dict[image_path] = {"beam_search": caption_beam, "nucleus_sampling": caption_nucleus}
                print(f"--- Image: {filename} ---")
                print(f"Caption (Beam Search): {caption_beam}")
                print(f"Captions (Nucleus Sampling): {caption_nucleus}\n")
            except Exception as e:
                print(f"Could not process image {filename}: {e}")
    return captions_dict


def get_sharegpt4v_image_representations(folder_path, processor, model, device='cuda:0'):
    """
    Loads images from a folder, processes them using the ShareGPT4V processor
    to obtain pixel values, and then obtains their latent representations
    using the vision encoder part of the ShareGPT4V model.

    Args:
        folder_path (str): Path to the folder containing images.
        processor: The loaded ShareGPT4V processor.
        model: The loaded ShareGPT4V model.

    Returns:
        dict: A dictionary where keys are image file paths and values are
              the corresponding latent representations (PyTorch tensors on CPU).
    """
    representations = {}
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return representations

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image = Image.open(image_path).convert("RGB") # PIL Image
            
            # Process the image using the ShareGPT4V processor
            # This typically handles normalization and conversion to pixel_values
            # The processor might be a LlavaProcessor or similar
            image_tensor = processor(image, return_tensors='pt')['pixel_values']
            pixel_values = image_tensor.to(device)

            with torch.no_grad():
                # ShareGPT4V models (LLaVA-based) usually have an 'encode_images' method
                # or access vision features through a 'vision_tower' or 'get_vision_tower()'
                if hasattr(model, 'encode_images'):
                    # This is a common pattern in LLaVA-style models
                    image_features = model.encode_images(pixel_values)
                elif hasattr(model, 'vision_tower') and callable(model.vision_tower):
                    # If vision_tower is the encoder itself
                    # Output might need selection, e.g., .last_hidden_state or specific pooler output
                    vision_tower_output = model.vision_tower(pixel_values)
                    if hasattr(vision_tower_output, 'last_hidden_state'):
                        image_features = vision_tower_output.last_hidden_state
                    elif isinstance(vision_tower_output, torch.Tensor): # Direct tensor output
                        image_features = vision_tower_output
                    else:
                        print(f"Warning: 'model.vision_tower' output type not directly usable for {filename}. Output: {type(vision_tower_output)}")
                        continue
                elif hasattr(model, 'get_vision_tower') and callable(model.get_vision_tower):
                    vision_tower = model.get_vision_tower()
                    vision_tower_output = vision_tower(pixel_values)
                    if hasattr(vision_tower_output, 'last_hidden_state'):
                        image_features = vision_tower_output.last_hidden_state
                    elif isinstance(vision_tower_output, torch.Tensor):
                            image_features = vision_tower_output
                    else:
                        print(f"Warning: 'model.get_vision_tower()(...)' output type not directly usable for {filename}. Output: {type(vision_tower_output)}")
                        continue
                else:
                    print(f"Error: Could not find a suitable method (encode_images, vision_tower) to extract image features for {filename}")
                    continue
            
            # The output of encode_images or vision_tower might be [batch_size, num_tokens, hidden_size]
            # or [batch_size, hidden_size] if pooled.
            # For consistency, let's assume we want the features before any projection LLaVA might do.
            # If image_features has 3 dims (B, N, D) and you need a pooled version (B,D),
            # you might need to average or take a specific token (e.g., CLS token if applicable).
            # For now, we store what the vision encoder returns.
            representations[image_path] = image_features.cpu() # Store on CPU
    return representations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image captions using a BLIP model.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("model_name", type=str, help="Name of the BLIP model (e.g., 'blip2_t5', 'blip_caption').")
    parser.add_argument("--model_type", type=str, default="caption_coco_flant5xl",
                        help="Type of the model (e.g., 'caption_coco_flant5xl', 'base_coco'). Default: 'caption_coco_flant5xl'.")
    parser.add_argument("--num_captions", type=int, default=5,
                        help="Number of captions to generate with nucleus sampling. Default: 5.")
    parser.add_argument("--output_json", type=str, default="caption_results.json",
                        help="Path to save the captioning results as a JSON file. Default: 'caption_results.json'.")

    args = parser.parse_args()

    print(f"Starting image captioning for folder: {args.folder_path}")
    print(f"Using model: {args.model_name}, type: {args.model_type}")
    print(f"Number of nucleus sampling captions: {args.num_captions}")

    results = image_captioning(
        folder_path=args.folder_path,
        model_name=args.model_name,
        model_type=args.model_type,
        num_captions=args.num_captions
    )

    if results:
        print("\n--- Captioning Results ---")
        for img_path, captions_data in results.items():
            print(f"Image: {os.path.basename(img_path)}")
            print(f"  Beam Search: {captions_data['beam_search']}")
            print(f"  Nucleus Sampling: {captions_data['nucleus_sampling']}")
            print("-" * 20)

        # Save results to JSON file
        try:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nResults saved to {args.output_json}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")

    print("Image captioning finished.")
