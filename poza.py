import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont
import argparse
import cv2
import numpy as np

def take_photo(output_path="captured_photo.jpg", width=1920, height=1080):
    """
    Take a photo using libcamera-still command
    """
    try:
        cmd = [
            "libcamera-still",
            "-o", output_path,
            "--width", str(width),
            "--height", str(height),
            "--timeout", "2000"  # 2 second delay
        ]
        
        print(f"Taking photo with resolution {width}x{height}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Photo saved successfully to {output_path}")
            return True
        else:
            print(f"Error taking photo: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Error: libcamera-still not found. Please install libcamera tools.")
        return False

def get_photo_resolution(image_path):
    """
    Get the resolution of the captured photo
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Photo resolution: {width}x{height}")
            return width, height
    except Exception as e:
        print(f"Error reading image: {e}")
        return None, None

def detect_paper_corners(image_path):
    """
    Detect the corners of a rectangular paper in the image using computer vision
    """
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not load image")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (assuming it's the paper)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we found a quadrilateral, use it as the paper
        if len(approx) == 4:
            corners = approx.reshape(-1, 2)
            # Sort corners: top-left, top-right, bottom-right, bottom-left
            corners = sort_corners(corners)
            
            print("Paper corners detected:")
            corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]
            corner_dict = {}
            for i, name in enumerate(corner_names):
                corner_dict[name] = tuple(corners[i])
                print(f"{name}: {corner_dict[name]}")
            
            return corner_dict
        else:
            print(f"Could not detect rectangular paper. Found {len(approx)} corners instead of 4.")
            return None
            
    except Exception as e:
        print(f"Error detecting paper corners: {e}")
        return None

def sort_corners(corners):
    """
    Sort corners in order: top-left, top-right, bottom-right, bottom-left
    """
    # Calculate centroid
    centroid = np.mean(corners, axis=0)
    
    # Sort by angle from centroid
    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
    
    # Sort corners by angle (starting from top-left, going clockwise)
    corners_with_angles = [(corner, angle_from_centroid(corner)) for corner in corners]
    corners_with_angles.sort(key=lambda x: x[1])
    
    # Rearrange to start from top-left
    sorted_corners = [corner for corner, _ in corners_with_angles]
    
    # Find top-left (smallest x+y sum)
    top_left_idx = np.argmin([pt[0] + pt[1] for pt in sorted_corners])
    
    # Reorder starting from top-left, going clockwise
    reordered = sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx]
    
    return np.array(reordered)

def calculate_paper_center(corners):
    """
    Calculate the center coordinates of the rectangular paper
    """
    if corners and len(corners) == 4:
        # Method 1: Simple average of all corners
        corner_coords = np.array(list(corners.values()))
        center_x = np.mean(corner_coords[:, 0])
        center_y = np.mean(corner_coords[:, 1])
        
        # Method 2: Intersection of diagonals (more accurate for rectangles)
        top_left = corners["top_left"]
        top_right = corners["top_right"]
        bottom_left = corners["bottom_left"]
        bottom_right = corners["bottom_right"]
        
        # Calculate intersection of diagonals
        diagonal1_center_x = (top_left[0] + bottom_right[0]) / 2
        diagonal1_center_y = (top_left[1] + bottom_right[1]) / 2
        
        diagonal2_center_x = (top_right[0] + bottom_left[0]) / 2
        diagonal2_center_y = (top_right[1] + bottom_left[1]) / 2
        
        # Average both methods for better accuracy
        final_center_x = (diagonal1_center_x + diagonal2_center_x) / 2
        final_center_y = (diagonal1_center_y + diagonal2_center_y) / 2
        
        center = (int(final_center_x), int(final_center_y))
        
        print(f"Paper center coordinates: {center}")
        print(f"  Method 1 (average): ({center_x:.1f}, {center_y:.1f})")
        print(f"  Method 2 (diagonals): ({final_center_x:.1f}, {final_center_y:.1f})")
        
        return center
    return None

def calculate_paper_perimeter(corners):
    """
    Calculate the perimeter of the detected paper
    """
    if corners and len(corners) == 4:
        corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]
        perimeter = 0
        
        for i in range(4):
            current = corners[corner_names[i]]
            next_corner = corners[corner_names[(i + 1) % 4]]
            
            # Calculate distance between consecutive corners
            distance = np.sqrt((next_corner[0] - current[0])**2 + (next_corner[1] - current[1])**2)
            perimeter += distance
        
        print(f"Paper perimeter: {perimeter:.2f} pixels")
        return perimeter
    return None

def get_corner_areas_of_interest(corners, corner_size=100):
    """
    Define areas of interest around each detected paper corner
    """
    if not corners:
        return None
    
    corner_areas = {}
    for corner_name, (x, y) in corners.items():
        # Create rectangular area around each corner
        area = {
            "center": (int(x), int(y)),
            "bounds": {
                "left": max(0, int(x - corner_size//2)),
                "top": max(0, int(y - corner_size//2)),
                "right": int(x + corner_size//2),
                "bottom": int(y + corner_size//2)
            }
        }
        corner_areas[corner_name] = area
    
    print(f"\nPaper corner areas of interest:")
    for corner, data in corner_areas.items():
        center = data["center"]
        bounds = data["bounds"]
        print(f"{corner}:")
        print(f"  Center: {center}")
        print(f"  Area bounds: ({bounds['left']}, {bounds['top']}) to ({bounds['right']}, {bounds['bottom']})")
        print(f"  Area size: {bounds['right'] - bounds['left']}x{bounds['bottom'] - bounds['top']}")
    
    return corner_areas

def create_paper_corner_visualization(image_path, output_path, corners, corner_areas, paper_center, corner_size=100):
    """
    Create a visualization focusing on the detected paper corners and center
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Create a copy to draw on
            img_with_corners = img.copy()
            draw = ImageDraw.Draw(img_with_corners)
            
            if corners and corner_areas and paper_center:
                # Draw paper outline
                corner_names = ["top_left", "top_right", "bottom_right", "bottom_left", "top_left"]
                paper_coords = [corners[name] for name in corner_names]
                draw.line(paper_coords, fill="yellow", width=4)
                
                # Draw center point
                center_x, center_y = paper_center
                draw.ellipse([
                    center_x - 15, center_y - 15,
                    center_x + 15, center_y + 15
                ], fill="purple", outline="white", width=4)
                
                # Draw cross at center
                draw.line([center_x - 25, center_y, center_x + 25, center_y], fill="purple", width=3)
                draw.line([center_x, center_y - 25, center_x, center_y + 25], fill="purple", width=3)
                
                # Highlight corner areas of interest
                colors = {
                    "top_left": "red",
                    "top_right": "green", 
                    "bottom_left": "blue",
                    "bottom_right": "orange"
                }
                
                for corner_name, area_data in corner_areas.items():
                    bounds = area_data["bounds"]
                    center = area_data["center"]
                    color = colors[corner_name]
                    
                    # Draw corner area rectangle
                    draw.rectangle([
                        bounds["left"], bounds["top"],
                        bounds["right"], bounds["bottom"]
                    ], outline=color, width=4)
                    
                    # Fill corner area with semi-transparent effect (using stipple pattern)
                    for i in range(bounds["left"], bounds["right"], 4):
                        for j in range(bounds["top"], bounds["bottom"], 4):
                            if (i + j) % 8 == 0:  # Create pattern
                                draw.point((i, j), fill=color)
                    
                    # Draw center point (actual paper corner)
                    draw.ellipse([
                        center[0] - 8, center[1] - 8,
                        center[0] + 8, center[1] + 8
                    ], fill=color, outline="white", width=3)
                    
                    # Add corner labels
                    try:
                        font = ImageFont.load_default()
                        label_x = bounds["left"] + 5
                        label_y = bounds["top"] + 5
                        
                        # Ensure label is within image bounds
                        if label_x < width - 100 and label_y < height - 40:
                            draw.rectangle([label_x-2, label_y-2, label_x+98, label_y+38], fill="black", outline=color)
                            draw.text((label_x, label_y), f"{corner_name.upper()}\n{center}", fill="white", font=font)
                    except:
                        pass
                
                # Add center label
                try:
                    font = ImageFont.load_default()
                    center_label_x = center_x + 30
                    center_label_y = center_y - 20
                    
                    if center_label_x < width - 100 and center_label_y > 0:
                        draw.rectangle([center_label_x-2, center_label_y-2, center_label_x+98, center_label_y+38], fill="black", outline="purple")
                        draw.text((center_label_x, center_label_y), f"CENTER\n{paper_center}", fill="white", font=font)
                except:
                    pass
                
                # Add overall information
                try:
                    font = ImageFont.load_default()
                    perimeter = calculate_paper_perimeter(corners)
                    info_text = f"PAPER DETECTION\nCenter: {paper_center}\nPerimeter: {perimeter:.0f}px\nCorner Size: {corner_size}px\nResolution: {width}x{height}\nPaper Corners: 4"
                    
                    # Draw info background
                    draw.rectangle([5, 5, 250, 130], fill="black", outline="white")
                    draw.text((10, 10), info_text, fill="yellow", font=font)
                except:
                    pass
                
                # Save the visualization
                img_with_corners.save(output_path)
                print(f"Paper corner visualization saved to {output_path}")
                return True
            
    except Exception as e:
        print(f"Error creating paper corner visualization: {e}")
        return False

def extract_paper_corner_regions(image_path, corner_areas, output_dir="paper_corners"):
    """
    Extract and save individual paper corner regions as separate images
    """
    import os
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        with Image.open(image_path) as img:
            for corner_name, area_data in corner_areas.items():
                bounds = area_data["bounds"]
                
                # Extract corner region
                corner_img = img.crop((
                    bounds["left"], bounds["top"],
                    bounds["right"], bounds["bottom"]
                ))
                
                # Save individual corner
                corner_path = os.path.join(output_dir, f"paper_{corner_name}.jpg")
                corner_img.save(corner_path)
                print(f"Paper corner region saved: {corner_path}")
        
        return True
    except Exception as e:
        print(f"Error extracting paper corner regions: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Take photo and detect paper corners")
    parser.add_argument("--width", type=int, default=3840, help="Photo width (default: 1500)")
    parser.add_argument("--height", type=int, default=2160, help="Photo height (default: 1024)")
    parser.add_argument("--corner-size", type=int, default=100, help="Corner area size in pixels (default: 100)")
    parser.add_argument("--output", type=str, default="captured_photo.jpg", help="Output photo filename")
    parser.add_argument("--extract-corners", action="store_true", help="Extract paper corner regions as separate images")
    parser.add_argument("--image", type=str, help="Use existing image instead of taking photo")
    
    args = parser.parse_args()
    
    image_path = args.image if args.image else args.output
    
    # Take photo if no existing image provided
    if not args.image:
        if not take_photo(args.output, args.width, args.height):
            print("Failed to take photo")
            return
    
    # Get photo resolution
    width, height = get_photo_resolution(image_path)
    if not width or not height:
        print("Failed to get photo resolution")
        return
    
    # Detect paper corners
    paper_corners = detect_paper_corners(image_path)
    if not paper_corners:
        print("Failed to detect paper corners")
        return
    
    # Calculate paper center
    paper_center = calculate_paper_center(paper_corners)
    
    # Calculate paper perimeter
    perimeter = calculate_paper_perimeter(paper_corners)
    
    # Get corner areas of interest
    corner_areas = get_corner_areas_of_interest(paper_corners, args.corner_size)
    
    if paper_corners and corner_areas and paper_center:
        print(f"\nFinal Results:")
        print(f"Photo: {image_path}")
        print(f"Resolution: {width}x{height}")
        print(f"Paper center: {paper_center}")
        print(f"Paper perimeter: {perimeter:.2f} pixels")
        print(f"Corner area size: {args.corner_size}x{args.corner_size} pixels")
        
        # Create paper corner visualization
        visualization_path = f"paper_corners_{args.output}"
        if create_paper_corner_visualization(image_path, visualization_path, paper_corners, corner_areas, paper_center, args.corner_size):
            print(f"Paper corner visualization created: {visualization_path}")
        
        # Extract individual corner regions if requested
        if args.extract_corners:
            extract_paper_corner_regions(image_path, corner_areas)

if __name__ == "__main__":
    main()