"""
Defect Heatmap Visualization Test Script
Provides multiple depth perception visualization options for defect detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class DefectHeatmapVisualizer:
    """
    Creates various depth perception heatmaps to highlight defects in images
    """
    
    def __init__(self, image_path):
        """
        Initialize with an image path
        
        Args:
            image_path: Path to the defect image
        """
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.image.shape[:2]
    
    def edge_based_heatmap(self, blur_size=5, canny_low=50, canny_high=150):
        """
        Option 1: Edge-based depth heatmap
        Highlights edges and contours where defects typically appear
        """
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (blur_size, blur_size), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Create distance transform (depth from edges)
        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply colormap for depth perception
        heatmap = cv2.applyColorMap(255 - dist_transform.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original
        overlay = cv2.addWeighted(self.image, 0.6, heatmap, 0.4, 0)
        
        return {
            'heatmap': cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            'edges': edges,
            'name': 'Edge-Based Depth Heatmap'
        }
    
    def gradient_magnitude_heatmap(self, ksize=3):
        """
        Option 2: Gradient magnitude heatmap
        Shows intensity changes (defects have high gradients)
        """
        # Compute gradients
        grad_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Compute magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(magnitude, cv2.COLORMAP_HOT)
        
        # Blend with original
        overlay = cv2.addWeighted(self.image, 0.5, heatmap, 0.5, 0)
        
        return {
            'heatmap': cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            'magnitude': magnitude,
            'name': 'Gradient Magnitude Heatmap'
        }
    
    def laplacian_heatmap(self, ksize=3):
        """
        Option 3: Laplacian-based heatmap
        Detects rapid intensity changes (dents, scratches, defects)
        """
        # Compute Laplacian
        laplacian = cv2.Laplacian(self.gray, cv2.CV_64F, ksize=ksize)
        laplacian = np.abs(laplacian)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(laplacian, cv2.COLORMAP_VIRIDIS)
        
        # Blend with original
        overlay = cv2.addWeighted(self.image, 0.6, heatmap, 0.4, 0)
        
        return {
            'heatmap': cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            'laplacian': laplacian,
            'name': 'Laplacian Depth Heatmap'
        }
    
    def morphological_gradient_heatmap(self, kernel_size=5):
        """
        Option 4: Morphological gradient heatmap
        Good for detecting surface irregularities and defects
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Morphological gradient
        gradient = cv2.morphologyEx(self.gray, cv2.MORPH_GRADIENT, kernel)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        
        # Blend with original
        overlay = cv2.addWeighted(self.image, 0.5, heatmap, 0.5, 0)
        
        return {
            'heatmap': cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            'gradient': gradient,
            'name': 'Morphological Gradient Heatmap'
        }
    
    def texture_based_heatmap(self, window_size=15):
        """
        Option 5: Texture variance heatmap
        Highlights areas with abnormal texture (common in defects)
        """
        # Compute local standard deviation
        mean = cv2.blur(self.gray.astype(np.float32), (window_size, window_size))
        sqr_mean = cv2.blur(self.gray.astype(np.float32)**2, (window_size, window_size))
        variance = sqr_mean - mean**2
        variance = np.maximum(variance, 0)  # Handle numerical errors
        std_dev = np.sqrt(variance)
        
        # Normalize
        std_dev = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(std_dev, cv2.COLORMAP_PLASMA)
        
        # Blend with original
        overlay = cv2.addWeighted(self.image, 0.6, heatmap, 0.4, 0)
        
        return {
            'heatmap': cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            'variance': std_dev,
            'name': 'Texture Variance Heatmap'
        }
    
    def combined_multi_scale_heatmap(self):
        """
        Option 6: Multi-scale combined heatmap (RECOMMENDED)
        Combines multiple features for robust defect highlighting
        """
        # 1. Edges at fine scale
        edges_fine = cv2.Canny(cv2.GaussianBlur(self.gray, (3, 3), 0), 50, 150)
        
        # 2. Edges at coarse scale
        edges_coarse = cv2.Canny(cv2.GaussianBlur(self.gray, (9, 9), 0), 30, 100)
        
        # 3. Gradient magnitude
        grad_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 4. Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph_grad = cv2.morphologyEx(self.gray, cv2.MORPH_GRADIENT, kernel)
        
        # Combine features
        combined = (
            edges_fine.astype(np.float32) * 0.3 +
            edges_coarse.astype(np.float32) * 0.2 +
            magnitude.astype(np.float32) * 0.3 +
            morph_grad.astype(np.float32) * 0.2
        )
        
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_INFERNO)
        
        # Create enhanced overlay with contours
        overlay = cv2.addWeighted(self.image, 0.6, heatmap, 0.4, 0)
        
        # Add contour highlights
        _, thresh = cv2.threshold(combined, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_with_contours = overlay.copy()
        cv2.drawContours(overlay_with_contours, contours, -1, (0, 255, 255), 2)
        
        return {
            'heatmap': cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            'overlay_contours': cv2.cvtColor(overlay_with_contours, cv2.COLOR_BGR2RGB),
            'combined': combined,
            'name': 'Multi-Scale Combined Heatmap (RECOMMENDED)'
        }
    
    def saliency_heatmap(self):
        """
        Option 7: Saliency-based heatmap
        Highlights visually salient regions (often defects)
        """
        # Create saliency detector
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(self.image)
        
        if not success:
            return None
        
        # Normalize and convert
        saliency_map = (saliency_map * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        
        # Blend with original
        overlay = cv2.addWeighted(self.image, 0.6, heatmap, 0.4, 0)
        
        return {
            'heatmap': cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            'saliency': saliency_map,
            'name': 'Saliency-Based Heatmap'
        }


def create_comparison_figure(image_path, output_dir="heatmap_outputs"):
    """
    Generate all heatmap visualizations and create a comparison figure
    
    Args:
        image_path: Path to the defect image
        output_dir: Directory to save outputs
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize visualizer
    viz = DefectHeatmapVisualizer(image_path)
    
    # Generate all visualizations
    results = []
    
    print("Generating visualizations...")
    print("1. Edge-based heatmap...")
    results.append(viz.edge_based_heatmap())
    
    print("2. Gradient magnitude heatmap...")
    results.append(viz.gradient_magnitude_heatmap())
    
    print("3. Laplacian heatmap...")
    results.append(viz.laplacian_heatmap())
    
    print("4. Morphological gradient heatmap...")
    results.append(viz.morphological_gradient_heatmap())
    
    print("5. Texture variance heatmap...")
    results.append(viz.texture_based_heatmap())
    
    print("6. Multi-scale combined heatmap (RECOMMENDED)...")
    results.append(viz.combined_multi_scale_heatmap())
    
    print("7. Saliency heatmap...")
    saliency_result = viz.saliency_heatmap()
    if saliency_result:
        results.append(saliency_result)
    
    # Create comparison figure
    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 3, figsize=(18, 6*n_methods))
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results):
        # Original image
        axes[idx, 0].imshow(viz.image_rgb)
        axes[idx, 0].set_title(f'{result["name"]}\nOriginal Image', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Heatmap only
        axes[idx, 1].imshow(result['heatmap'])
        axes[idx, 1].set_title('Depth Heatmap', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Overlay
        overlay_key = 'overlay_contours' if 'overlay_contours' in result else 'overlay'
        axes[idx, 2].imshow(result[overlay_key])
        axes[idx, 2].set_title('Overlay on Original', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    # Save comparison figure
    comparison_path = output_path / f"comparison_{Path(image_path).stem}.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison saved to: {comparison_path}")
    
    # Save individual heatmaps
    for idx, result in enumerate(results):
        method_name = result['name'].replace(' ', '_').replace('(', '').replace(')', '').lower()
        
        # Save heatmap
        heatmap_path = output_path / f"{method_name}_heatmap_{Path(image_path).stem}.png"
        plt.imsave(heatmap_path, result['heatmap'])
        
        # Save overlay
        overlay_key = 'overlay_contours' if 'overlay_contours' in result else 'overlay'
        overlay_path = output_path / f"{method_name}_overlay_{Path(image_path).stem}.png"
        plt.imsave(overlay_path, result[overlay_key])
    
    print(f"✅ Individual heatmaps saved to: {output_path}")
    
    plt.show()
    
    return results


def test_on_sample_images():
    """
    Test function - looks for sample images in common locations
    """
    # Check for sample images
    test_paths = [
        "6.jpeg"
    ]
    
    found_images = []
    for path in test_paths:
        if os.path.exists(path):
            found_images.append(path)
    
    if not found_images:
        print("❌ No test images found!")
        print("\nPlease provide a defect image path, for example:")
        print("  create_comparison_figure('path/to/your/defect_image.jpg')")
        print("\nOr place a test image in one of these locations:")
        for path in test_paths:
            print(f"  - {path}")
        return
    
    print(f"✅ Found {len(found_images)} test image(s)")
    
    for img_path in found_images:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path}")
        print('='*60)
        create_comparison_figure(img_path)


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("DEFECT HEATMAP VISUALIZATION TEST")
    print("="*60)
    print("\nThis script generates 7 different depth perception heatmaps:")
    print("  1. Edge-Based Depth Heatmap")
    print("  2. Gradient Magnitude Heatmap")
    print("  3. Laplacian Depth Heatmap")
    print("  4. Morphological Gradient Heatmap")
    print("  5. Texture Variance Heatmap")
    print("  6. Multi-Scale Combined Heatmap ⭐ RECOMMENDED")
    print("  7. Saliency-Based Heatmap")
    print("\n" + "="*60 + "\n")
    
    if len(sys.argv) > 1:
        # User provided image path
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ Error: Image not found at {image_path}")
            sys.exit(1)
        
        create_comparison_figure(image_path)
    else:
        # Auto-detect test images
        test_on_sample_images()