from flask import Flask, request, jsonify  # Add jsonify import here
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from skimage.measure import find_contours
import cv2

# Define the class labels (adjust these based on your dataset)
class_labels = {
    0: 'benign keratosis-like lesions (bkl)',
    1: 'melanocytic nevi (nv)',
    2: 'dermatofibroma (df)',
    3: 'melanoma (mel)',
    4: 'vascular lesions (vasc)',
    5: 'basal cell carcinoma (bcc)',
    6: 'actinic keratoses and intraepithelial carcinoma (akiec)'
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Change the resize size to 256x256 to match the model's expected input shape
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

def load_image(filepath):
    """Load and preprocess the image for models."""
    image = Image.open(filepath).convert('RGB')  # Ensure image is RGB
    image = transform(image).unsqueeze(0).to(device)  # Apply transform and add batch dimension
    return image

def preprocess_image(img_path):
    """Preprocess image for model inference."""
    # Load the image and apply transformations
    image = load_image(img_path)
    return image

def classify_image(model, img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Check the type of image and output
    print(f"Image type: {type(image)}")  # Should be a torch tensor

    with torch.no_grad():
        output = model(image)

    # Check the type of the output
    print(f"Model output type: {type(output)}")  # Should be a tensor

    _, predicted = torch.max(output, 1)
    print(f"Predicted: {predicted}")  # This should be a tensor

    predicted_class = class_labels[predicted.item()]  # Convert index to class label
    return predicted_class




def check_and_process_mask(mask):
    """
    Ensure that the mask is binary (contains only 0 and 255).
    
    Parameters:
    - mask: The mask to check and process.
    
    Returns:
    - processed_mask: A binary mask (values 0 or 255).
    """
    print(f"Mask min: {mask.min()}, Mask max: {mask.max()}")
    
    # Convert mask to binary (0 and 255) if it's not already
    processed_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    return processed_mask

def extract_and_overlay_boundaries(image, mask, boundary_color=(0, 0, 0), thickness=2):
    """
    Extract boundaries from a binary mask and overlay them on the original image.
    
    Parameters:
    - image: The original image as a NumPy array.
    - mask: The binary mask as a NumPy array.
    - boundary_color: The color of the boundary to overlay (default is black).
    - thickness: The thickness of the boundary line.
    
    Returns:
    - overlay_image: The image with overlaid boundaries.
    - lesion_mask: The binary lesion mask.
    """
    # Step 1: Process the mask to ensure it's binary
    mask = check_and_process_mask(mask)

    # Step 2: Find contours in the binary mask
    contours = find_contours(mask, level=0.5)
    print(f"Contours found: {len(contours)}")
    
    if len(contours) == 0:
        print("Warning: No contours detected!")

    # Step 3: Create a white image to overlay the boundaries (255 for white)
    overlay_image = np.ones_like(image, dtype=np.uint8) * 255  # White background

    # Step 4: Overlay the contours on the overlay_image
    for contour in contours:
        contour = np.round(contour).astype(np.int32)
        cv2.polylines(overlay_image, [contour], isClosed=True, color=boundary_color, thickness=thickness)

    # Step 5: Generate the lesion mask (binary mask)
    lesion_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)  # Black and white lesion mask

    # Return both the lesion mask and the overlay image with boundaries
    return lesion_mask, overlay_image


def detect_lesion(model, image, output_path="static/lesion_output.jpg"):
    """
    Run lesion detection model on the image and save the output with dynamic thresholding.
    """
    # Convert PyTorch tensor to NumPy array for compatibility with Keras
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Normalize pixel values for Keras model input
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # Expand dimensions to add batch size (Keras expects 4D tensors)
    image_np = np.expand_dims(image_np, axis=0)

    # Perform inference with the Keras model
    lesion_output = model.predict(image_np)

    # Assuming the model outputs an image-like tensor
    lesion_image = lesion_output.squeeze().astype(np.float32)

    # Normalize lesion image to 0-255 for thresholding
    lesion_image = (lesion_image - lesion_image.min()) / (lesion_image.max() - lesion_image.min()) * 255
    lesion_image = lesion_image.astype(np.uint8)

    # Apply dynamic thresholding (Otsu's method)
    _, lesion_mask = cv2.threshold(lesion_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the lesion mask as an image
    lesion_output_image = Image.fromarray(lesion_mask)
    lesion_output_image.save(output_path)

    return output_path  # Return the saved image path

import matplotlib.pyplot as plt

def display_with_extracted_boundaries(image, mask):
    """
    Display the original image, lesion mask (black and white), and image with boundaries overlaid.
    
    Parameters:
    - image: The original input image.
    - mask: The predicted binary lesion mask.
    """
    # Extract the lesion mask and overlay the boundaries
    lesion_mask, overlay_image = extract_and_overlay_boundaries(image, mask, boundary_color=(0, 0, 0), thickness=2)

    if lesion_mask is None or overlay_image is None:
        print("No valid contours or mask found.")
        return

    # Display results
    plt.figure(figsize=(15, 15))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Lesion Mask (Black and White)
    plt.subplot(1, 3, 2)
    plt.title('Lesion Mask (Black and White)')
    plt.imshow(lesion_mask, cmap='gray')
    plt.axis('off')

    # Image with Extracted Boundaries
    plt.subplot(1, 3, 3)
    plt.title('Image with Boundaries')
    plt.imshow(overlay_image, cmap='gray')
    plt.axis('off')

    plt.show()


def visualize_histogram_and_threshold(image, threshold_value):
    plt.figure(figsize=(10, 5))

    # Plot the histogram
    plt.hist(image.ravel(), bins=256, color='blue', alpha=0.7)
    plt.axvline(x=threshold_value, color='red', linestyle='--', label=f'Threshold = {threshold_value}')
    plt.title('Pixel Intensity Histogram and Threshold')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Assuming the model is already loaded as `model`
    img_path = "path_to_your_image.jpg"  # Replace with actual image path
    model_output = classify_image(model, img_path)
    print(f"Predicted class: {model_output}")
