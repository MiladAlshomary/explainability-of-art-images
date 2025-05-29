from PIL import Image
import numpy as np
from IPython.display import display # Used to explicitly display images
# For Method 4 (displaying from bytes), you might also need:
# from IPython.display import Image as IPImage 

# --- IMPORTANT ---
# Replace the line below with your actual VisImage object.
# For example:
# vis_image_instance = my_function_that_returns_a_vis_image()
# or
# vis_image_instance = MyVisImageClass(parameters...)

def display_image(vis_image_instance):
    if vis_image_instance is None:
        print("Please assign your actual VisImage object to the 'vis_image_instance' variable.")
    else:
        pil_image_to_display = None
        print(f"Attempting to display VisImage of type: {type(vis_image_instance)}")

        # Option 1: Check for a .to_pil() method (or similar)
        # Common method names: to_pil(), as_pil(), get_pil_image(), etc.
        # Check your VisImage object's documentation for the correct method name.
        if hasattr(vis_image_instance, 'to_pil'):
            try:
                print("Attempting to use .to_pil() method...")
                pil_image_to_display = vis_image_instance.to_pil()
                if not isinstance(pil_image_to_display, Image.Image):
                    print(f".to_pil() did not return a PIL Image. Got: {type(pil_image_to_display)}")
                    pil_image_to_display = None # Reset if not a PIL Image
            except Exception as e:
                print(f"Error calling .to_pil(): {e}")
        elif hasattr(vis_image_instance, 'as_pil'): # Another common name
            try:
                print("Attempting to use .as_pil() method...")
                pil_image_to_display = vis_image_instance.as_pil()
                if not isinstance(pil_image_to_display, Image.Image):
                    print(f".as_pil() did not return a PIL Image. Got: {type(pil_image_to_display)}")
                    pil_image_to_display = None
            except Exception as e:
                print(f"Error calling .as_pil(): {e}")
        
        # Option 2: Check for a .data attribute that is a NumPy array
        if pil_image_to_display is None and hasattr(vis_image_instance, 'data'):
            if isinstance(vis_image_instance.data, np.ndarray):
                try:
                    print("Attempting to use .data attribute (NumPy array)...")
                    image_array = vis_image_instance.data
                    
                    # Important: Convert array to uint8 if necessary.
                    # PIL.Image.fromarray typically expects uint8 data.
                    # Adjust the conditions below based on your array's properties.
                    if image_array.dtype != np.uint8:
                        print(f"NumPy array dtype is {image_array.dtype}. Attempting conversion to uint8.")
                        if np.issubdtype(image_array.dtype, np.floating):
                            # Common case: float array in range [0.0, 1.0]
                            if image_array.min() >= 0.0 and image_array.max() <= 1.0:
                                image_array = (image_array * 255).astype(np.uint8)
                            # Common case: float array in range [0.0, 255.0]
                            elif image_array.min() >= 0.0 and image_array.max() <= 255.0:
                                image_array = image_array.astype(np.uint8)
                            else:
                                # For other float ranges, you might need specific normalization
                                print(f"Warning: Float array range is [{image_array.min()}, {image_array.max()}]. Simple cast to uint8 may not be ideal.")
                                image_array = image_array.astype(np.uint8)
                        elif image_array.dtype == np.uint16: # Example for uint16
                            # Scale 16-bit (0-65535) to 8-bit (0-255)
                            image_array = (image_array / 256).astype(np.uint8)
                        # Add more specific conversions if needed for other dtypes
                        else: 
                            image_array = image_array.astype(np.uint8)
                    
                    pil_image_to_display = Image.fromarray(image_array)
                except Exception as e:
                    print(f"Error converting .data (NumPy array) to PIL Image: {e}")
            elif vis_image_instance.data is not None:
                print(f".data attribute is not a NumPy array. Found type: {type(vis_image_instance.data)}")


        # Option 3: Check if the instance itself is already a PIL Image
        if pil_image_to_display is None and isinstance(vis_image_instance, Image.Image):
            print("The instance itself is already a PIL Image.")
            pil_image_to_display = vis_image_instance

        # Now, display the image if we successfully obtained a PIL Image
        if pil_image_to_display:
            print("Displaying image...")
            display(pil_image_to_display)
            # Note: If `display(pil_image_to_display)` is the last line in a Jupyter cell,
            # simply having `pil_image_to_display` by itself might also render it.
        else:
            print("\nCould not automatically convert VisImage to a displayable PIL Image using common methods.")
            print("Please consider the following:")
            print("1. Check the documentation for your specific VisImage class:")
            print("   - It might have a different method name for PIL conversion (e.g., `get_image()`, `render()`).")
            print("   - It might have a specific attribute for the NumPy array (e.g., `_data`, `array`).")
            print("   - It might have a dedicated display function for Jupyter (e.g., `vis_image_instance.show_notebook()`).")
            print("2. If your VisImage can provide raw image bytes (e.g., PNG, JPEG):")
            print("   You can use: ")
            print("   from IPython.display import Image as IPImage")
            print("   # Assuming vis_image_instance.get_bytes() returns image bytes")
            print("   # image_bytes = vis_image_instance.get_bytes()")
            print("   # display(IPImage(data=image_bytes))")
            print("3. Ensure the NumPy array (if used) has the correct shape (Height, Width, Channels) and data type for `Image.fromarray`.")

