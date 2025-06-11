from PIL import Image
import numpy as np
import os
import torch

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
