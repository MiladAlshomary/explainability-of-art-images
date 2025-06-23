import argparse
import json
from share4v.model.builder import load_pretrained_model
from utils import generate_descriptions_with_share4v, get_model_name_from_path

def image_analysis_with_gallery_gpt(image_path, model_path, model_base=None):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        get_model_name_from_path,
    )

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )

    descriptions = generate_descriptions_with_share4v(
        folder_path=image_path,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        prompt="Compose a short paragraph of formal analysis for this painting",
        conv_mode="llava_v1",  # LLaVA models typically use 'llava_v1'
        device="cuda:0",
        # You can add other generation arguments here
        max_new_tokens=512
    )

    return descriptions

def image_analysis_with_share4v(image_path, model_path):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device="cuda:0"
    )
    
    descriptions = generate_descriptions_with_share4v(
        folder_path=image_path,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        prompt="Compose a short paragraph of formal analysis for this painting",
        conv_mode="share4v_v1" ,
        device="cuda:0",
        # You can add other generation arguments here
        max_new_tokens=512
    )

    return descriptions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform image style analysis using a specified model.")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the folder containing images for analysis."
    )
    parser.add_argument(
        "--analyzer",
        type=str,
        required=True,
        choices=['share4v', 'gallery_gpt'],
        help="The type of analyzer model to use."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model weights (Hugging Face repo or local path)."
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
        help="Path to the base model, if required (e.g., for LLaVA delta weights)."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default=None,
        help="Path to save the generated descriptions as a JSON file. If not provided, prints to console."
    )
    args = parser.parse_args()

    descriptions = {}
    if args.analyzer == 'share4v':
        print(f"Running analysis with ShareGPT4V model: {args.model_path}")
        descriptions = image_analysis_with_share4v(args.image_path, args.model_path)
    elif args.analyzer == 'gallery_gpt':
        print(f"Running analysis with Gallery-GPT (LLaVA) model: {args.model_path}")
        descriptions = image_analysis_with_gallery_gpt(args.image_path, args.model_path, args.model_base)

    if args.output_path:
        try:
            with open(args.output_path, 'w') as f:
                json.dump(descriptions, f, indent=4)
            print(f"Descriptions successfully saved to {args.output_path}")
        except Exception as e:
            print(f"Error saving results to JSON file: {e}")
    else:
        print("\n--- Generated Descriptions ---")
        print(json.dumps(descriptions, indent=4))