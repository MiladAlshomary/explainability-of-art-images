from PIL import Image
import numpy as np
import os
import argparse # Added for command-line argument parsing
import json # Added for saving results to JSON
import torch

# Imports for clustering and dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def image_captioning(folder_path, model_name, model_type="caption_coco_flant5xl", num_captions=5):
    from lavis.models import load_model_and_preprocess
    
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
                raw_image_pil = Image.open(image_path).convert('RGB')

                # Speculative: Reconstruct PIL image via NumPy array.
                # This is an attempt to normalize the PIL object, in case subtle
                # internal differences contribute to downstream type issues.
                # This is unlikely to fix a fundamental type identity check error
                # (expected np.ndarray got numpy.ndarray) in a library.
                image_to_process = raw_image_pil # Default to original
                try:
                    np_array_from_pil = np.array(raw_image_pil)
                    if np_array_from_pil.dtype != np.uint8: # Ensure uint8 for RGB
                        np_array_from_pil = np_array_from_pil.astype(np.uint8)
                    image_to_process = Image.fromarray(np_array_from_pil)
                except Exception as recon_e:
                    print(f"Warning: Could not reconstruct PIL image {filename} via NumPy, using original. Error: {recon_e}")

                image = vis_processors["eval"](image_to_process).unsqueeze(0).to(device)

                caption_beam = model.generate({"image": image})
                caption_nucleus = model.generate(
                    {"image": image},
                    use_nucleus_sampling=True,
                    num_captions=num_captions,
                    temperature=1.2,  # Increased temperature for more randomness
                    top_p=0.9  # Adjust top_p for nucleus sampling diversity
                )

                captions_dict[image_path] = {"beam_search": caption_beam, "nucleus_sampling": caption_nucleus}
                # Print statements moved to main section for cleaner function output if used as a library
            except Exception as e:
                print(f"Could not process image {filename}: {e}")
    return captions_dict

def generate_image_descriptions(folder_path, processor, model, system_pormpt, device='cuda:0') -> dict:
    """
    Loads images from a folder, and for each image, generates a description
    or answers a question based on the system_pormpt using the provided model.
    This function assumes the model and processor are compatible with the LAVIS library's API,
    similar to the `image_captioning` function.

    Args:
        folder_path (str): Path to the folder containing images.
        processor: The visual processor (e.g., vis_processors["eval"] from lavis).
                   This is expected to be a callable that takes a PIL image and returns a tensor.
        model: The pre-trained model (e.g., a lavis VQA or instruction-following model)
               with a `generate` method.
        system_pormpt (str): The prompt, question, or instruction to apply to each image.
                             The model's `generate` method will receive this as the "prompt" argument.
                             Users should format this string appropriately for the specific model
                             (e.g., "Question: What is in this image? Answer:").
        device (str): The device to run the model on (e.g., 'cuda:0' or 'cpu').
                      The input image tensor will be moved to this device.

    Returns:
        dict: A dictionary where keys are image file paths and values are
              lists of generated textual outputs (strings) from the model.
    """
    descriptions_dict = {}

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return descriptions_dict

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                raw_image_pil = Image.open(image_path).convert('RGB')
                
                # Process image using the provided processor (e.g., lavis vis_processors["eval"])
                image_tensor = processor(raw_image_pil).unsqueeze(0).to(device)

                # Generate description/answer using the model's generate method
                with torch.no_grad(): # Ensure no gradients are computed during inference
                    generated_outputs = model.generate({"image": image_tensor, "prompt": system_pormpt})
                
                descriptions_dict[image_path] = generated_outputs # model.generate typically returns a list of strings
            except Exception as e:
                print(f"Could not process image {filename} for description generation: {e}")
    return descriptions_dict


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


def cluster_visual_representations(visual_representations, pca_n_components=50, eps_values=None, min_samples=5):
    """
    Cluster visual representations using DBSCAN after dimensionality reduction (PCA).
    Finds the best eps parameter based on the silhouette score.

    Args:
        visual_representations (dict): A dictionary where keys are image file paths
                                       and values are the corresponding latent representations
                                       (PyTorch tensors or NumPy arrays).
        pca_n_components (int): Number of components for PCA dimensionality reduction.
                                Set to None or 0 to skip PCA.
        eps_values (list or None): A list of eps values to try for DBSCAN.
                                   If None, a default range will be used.
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point (DBSCAN parameter).

    Returns:
        dict: A dictionary where keys are image file paths and values are the
              corresponding cluster labels (integers). Returns an empty dict
              if no valid clustering is found.
        float: The best silhouette score found. Returns -1.0 if no valid clustering.
        float: The best eps value found. Returns None if no valid clustering.
    """
    if not visual_representations:
        print("No visual representations provided for clustering.")
        return {}, -1.0, None

    # 1. Extract representations and image paths
    image_paths = list(visual_representations.keys())
    # Ensure representations are NumPy arrays and stack them
    representations_list = [rep.squeeze().cpu().numpy() if isinstance(rep, torch.Tensor) else rep.squeeze() for rep in visual_representations.values()]

    # Check if representations are valid (e.g., not empty or None)
    if not representations_list or any(rep is None or rep.size == 0 for rep in representations_list):
         print("Invalid or empty representations found.")
         return {}, -1.0, None

    # Ensure all representations have the same shape for stacking
    try:
        X = np.vstack(representations_list)
    except ValueError as e:
        print(f"Error stacking representations. Ensure all representations have the same shape. Error: {e}")
        return {}, -1.0, None

    print(f"Original representation shape: {X.shape}")

    # 2. Apply Dimensionality Reduction (PCA)
    if pca_n_components is not None and pca_n_components > 0 and pca_n_components < X.shape[1]:
        print(f"Applying PCA with {pca_n_components} components.")
        try:
            pca = PCA(n_components=pca_n_components)
            X_reduced = pca.fit_transform(X)
            print(f"Reduced representation shape: {X_reduced.shape}")
            X_to_cluster = X_reduced
        except Exception as e:
            print(f"Error during PCA: {e}")
            return {}, -1.0, None
    else:
        print("Skipping PCA.")
        X_to_cluster = X

    # 3. Define eps values to test
    if eps_values is None:
        # Default range - this is highly dependent on the data scale after PCA/no PCA
        # This range might need tuning based on your specific data and model outputs.
        eps_values = np.linspace(0.1, 2.0, 20) # Example range and steps
        print(f"Using default eps range: {eps_values}")
    else:
         print(f"Using provided eps values: {eps_values}")

    best_score = -1.0
    best_eps = None
    best_labels = None

    # 4. Iterate through eps values and evaluate
    print(f"Testing {len(eps_values)} eps values for DBSCAN...")
    for eps in eps_values:
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_to_cluster)

            # Check if clustering is valid for silhouette score calculation
            # Need at least 2 clusters, and not all points in one cluster (excluding noise -1)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0) # Count non-noise clusters

            if n_clusters >= 2 and len(X_to_cluster) > n_clusters:
                 score = silhouette_score(X_to_cluster, labels)
                 # print(f"  eps={eps:.4f}, Clusters={n_clusters}, Silhouette Score={score:.4f}")
                 if score > best_score:
                     best_score = score
                     best_eps = eps
                     best_labels = labels

        except Exception as e:
            print(f"Error during DBSCAN for eps={eps}: {e}")
            # Continue to the next eps value

    # 5. Return results
    if best_labels is not None:
        print(f"\nBest clustering found with eps={best_eps:.4f}, Silhouette Score={best_score:.4f}")
        # Map labels back to image paths
        cluster_results = {image_paths[i]: int(best_labels[i]) for i in range(len(image_paths))}
        return cluster_results, best_score, best_eps
    else:
        print("\nNo valid clustering found for the given eps range and parameters.")
        # Return empty results if no meaningful clustering was found
        return {}, -1.0, None

    

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
