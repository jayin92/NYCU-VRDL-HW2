import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from tqdm import tqdm
import seaborn as sns
import pandas as pd

def analyze_bbox_dimensions(json_file):
    """
    Analyze the dimensions of bounding boxes in a COCO JSON file.
    Also analyzes image dimensions.
    
    Args:
        json_file: Path to COCO format JSON file
    
    Returns:
        DataFrame with width, height, and aspect ratio information,
        DataFrame with image dimensions,
        Dictionary of category names
    """
    print(f"Loading annotations from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract annotations and image info
    annotations = data['annotations']
    images_dict = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    print(f"Found {len(annotations)} annotations across {len(images_dict)} images")
    print(f"Categories: {categories}")
    
    # Calculate dimensions for bounding boxes
    widths = []
    heights = []
    aspect_ratios = []  # aspect_ratio = width / height (w/h)
    areas = []
    category_ids = []
    relative_widths = []  # width / image_width
    relative_heights = []  # height / image_height
    
    for ann in tqdm(annotations, desc="Processing annotations"):
        x, y, w, h = ann['bbox']  # COCO format is [x, y, width, height]
        img = images_dict[ann['image_id']]
        img_width, img_height = img['width'], img['height']
        
        widths.append(w)
        heights.append(h)
        # aspect_ratio = width/height (a value > 1 means wider than tall, < 1 means taller than wide)
        aspect_ratios.append(w/h if h > 0 else 0)
        areas.append(w*h)
        category_ids.append(ann['category_id'])
        relative_widths.append(w / img_width)
        relative_heights.append(h / img_height)
    
    # Create DataFrame for bbox analysis
    df = pd.DataFrame({
        'width': widths,
        'height': heights,
        'aspect_ratio': aspect_ratios,
        'area': areas,
        'category_id': category_ids,
        'relative_width': relative_widths,
        'relative_height': relative_heights
    })
    
    # Calculate image dimensions
    image_widths = []
    image_heights = []
    image_aspect_ratios = []
    image_ids = []
    
    for img in tqdm(data['images'], desc="Processing images"):
        img_width, img_height = img['width'], img['height']
        image_widths.append(img_width)
        image_heights.append(img_height)
        image_aspect_ratios.append(img_width / img_height if img_height > 0 else 0)
        image_ids.append(img['id'])
    
    # Create DataFrame for image analysis
    img_df = pd.DataFrame({
        'image_id': image_ids,
        'image_width': image_widths,
        'image_height': image_heights,
        'image_aspect_ratio': image_aspect_ratios
    })
    
    return df, img_df, categories

def print_statistics(df, img_df, by_category=False, categories=None):
    """Print statistical information about bounding box and image dimensions"""
    
    # Overall bbox statistics
    print("\n=== Overall Bounding Box Statistics ===")
    stats = df[['width', 'height', 'aspect_ratio', 'area']].describe(
        percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]
    )
    print(stats)
    
    # Image dimension statistics
    print("\n=== Image Dimension Statistics ===")
    img_stats = img_df[['image_width', 'image_height', 'image_aspect_ratio']].describe(
        percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]
    )
    print(img_stats)
    
    # Most common image dimensions
    print("\n=== Most Common Image Dimensions ===")
    dimension_counts = img_df.groupby(['image_width', 'image_height']).size().reset_index(name='count')
    dimension_counts = dimension_counts.sort_values('count', ascending=False)
    print(dimension_counts.head(10))
    
    # Key statistics for anchor box design
    print("\n=== Key Statistics for Anchor Design ===")
    for dim in ['width', 'height']:
        percentiles = np.percentile(df[dim], [10, 20, 30, 40, 50, 60, 70, 80, 90])
        print(f"{dim.capitalize()} percentiles: {percentiles}")
    
    # Common aspect ratios
    aspect_ratios = df['aspect_ratio'].dropna()
    common_ratios = [0.5, 0.75, 1.0, 1.33, 1.5, 2.0]
    print("\nClosest aspect ratios to common values:")
    for target in common_ratios:
        count = np.sum((aspect_ratios >= target*0.9) & (aspect_ratios <= target*1.1))
        pct = (count / len(aspect_ratios)) * 100
        print(f"  Aspect ratio ~{target}: {count} boxes ({pct:.1f}%)")
    
    # Min-size and max-size suggestions based on image dimensions
    print("\n=== Suggested min_size and max_size Parameters ===")
    img_sizes = np.array([img_df['image_width'], img_df['image_height']]).min(axis=0)
    # Round to multiples of 50
    min_size = 50 * round(np.percentile(img_sizes, 10) / 50)
    max_size = 50 * round(np.percentile(img_sizes, 90) / 50)
    print(f"Suggested min_size: {min_size}")
    print(f"Suggested max_size: {max_size}")
    
    # Statistics by category
    if by_category and categories:
        print("\n=== Statistics by Category ===")
        for cat_id, name in categories.items():
            cat_df = df[df['category_id'] == cat_id]
            if len(cat_df) > 0:
                print(f"\nCategory: {name} (ID: {cat_id}), Count: {len(cat_df)}")
                cat_stats = cat_df[['width', 'height']].describe([.1, .5, .9])
                print(cat_stats)

def visualize_dimensions(df, img_df, output_dir, categories=None):
    """Create visualizations of bounding box and image dimensions"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # --- Bounding Box Visualizations ---
    plt.figure(figsize=(15, 10))
    
    # Distribution of widths and heights
    plt.subplot(2, 2, 1)
    sns.histplot(df['width'], kde=True, label='Width')
    sns.histplot(df['height'], kde=True, label='Height')
    plt.title('Distribution of Bounding Box Dimensions')
    plt.xlabel('Pixels')
    plt.ylabel('Count')
    plt.legend()
    
    # Aspect ratio distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['aspect_ratio'].clip(0, 3), bins=30, kde=True)
    plt.title('Distribution of Aspect Ratios (Width/Height)')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Count')
    
    # Width vs. Height scatter plot
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        x='width', 
        y='height', 
        hue='category_id' if categories and len(categories) < 10 else None,
        data=df, 
        alpha=0.3
    )
    plt.title('Width vs. Height')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    
    # Relative width vs. relative height
    plt.subplot(2, 2, 4)
    sns.scatterplot(
        x='relative_width', 
        y='relative_height', 
        hue='category_id' if categories and len(categories) < 10 else None,
        data=df, 
        alpha=0.3
    )
    plt.title('Relative Width vs. Relative Height')
    plt.xlabel('Width / Image Width')
    plt.ylabel('Height / Image Height')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_dimensions.png'))
    print(f"Saved bbox visualization to {os.path.join(output_dir, 'bbox_dimensions.png')}")
    
    # --- Image Dimension Visualizations ---
    plt.figure(figsize=(15, 10))
    
    # Distribution of image widths and heights
    plt.subplot(2, 2, 1)
    sns.histplot(img_df['image_width'], kde=True, label='Width')
    sns.histplot(img_df['image_height'], kde=True, label='Height')
    plt.title('Distribution of Image Dimensions')
    plt.xlabel('Pixels')
    plt.ylabel('Count')
    plt.legend()
    
    # Image aspect ratio distribution
    plt.subplot(2, 2, 2)
    sns.histplot(img_df['image_aspect_ratio'].clip(0, 3), bins=30, kde=True)
    plt.title('Distribution of Image Aspect Ratios (Width/Height)')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Count')
    
    # Image width vs. height scatter plot
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        x='image_width', 
        y='image_height', 
        data=img_df, 
        alpha=0.5
    )
    plt.title('Image Width vs. Height')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    
    # Hexbin plot for common dimensions
    plt.subplot(2, 2, 4)
    plt.hexbin(img_df['image_width'], img_df['image_height'], gridsize=20, cmap='viridis')
    plt.colorbar(label='Count')
    plt.title('Common Image Dimensions')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_dimensions.png'))
    print(f"Saved image dimension visualization to {os.path.join(output_dir, 'image_dimensions.png')}")
    
    # Box plot for width and height by category
    if categories and len(categories) < 10:
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x='category_id', y='width', data=df)
        plt.title('Width by Category')
        plt.xlabel('Category ID')
        plt.ylabel('Width (pixels)')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='category_id', y='height', data=df)
        plt.title('Height by Category')
        plt.xlabel('Category ID')
        plt.ylabel('Height (pixels)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bbox_dimensions_by_category.png'))
        print(f"Saved category visualization to {os.path.join(output_dir, 'bbox_dimensions_by_category.png')}")
    
    # --- Integrated visualization (bboxes in context of images) ---
    plt.figure(figsize=(10, 6))
    
    # Density plot: bbox size relative to image size
    # Updated to work with newer seaborn versions
    sns.kdeplot(x=df['width'] / df['relative_width'], 
                y=df['height'] / df['relative_height'], 
                cmap="viridis", 
                fill=True)
    plt.title('Bounding Boxes in Context of Images')
    plt.xlabel('Image Width (pixels)')
    plt.ylabel('Image Height (pixels)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxes_in_images.png'))
    print(f"Saved integrated visualization to {os.path.join(output_dir, 'boxes_in_images.png')}")
    
    plt.close('all')

def suggest_anchor_settings(df):
    """Suggest anchor settings based on the data analysis"""
    
    widths = np.array(df['width'])
    heights = np.array(df['height'])
    
    # Calculate representative sizes using k-means clustering
    from sklearn.cluster import KMeans
    
    # Convert to 2D array of (width, height)
    X = np.vstack((widths, heights)).T
    
    # Determine number of clusters (anchor sizes)
    n_clusters = min(5, len(X))
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    
    # Sort centers by area (width * height)
    areas = centers[:, 0] * centers[:, 1]
    sorted_indices = np.argsort(areas)
    centers = centers[sorted_indices]
    
    # Get representative areas and aspect ratios
    rep_heights = centers[:, 1] 
    rep_widths = centers[:, 0]
    rep_sizes = np.sqrt(rep_widths * rep_heights)  # geometric mean as representative size
    rep_aspect_ratios = rep_widths / rep_heights
    
    # Simplify aspect ratios to common values
    common_ratios = [0.5, 0.75, 1.0, 1.5, 2.0]
    simplified_ratios = []
    
    for ratio in rep_aspect_ratios:
        closest = min(common_ratios, key=lambda x: abs(x - ratio))
        simplified_ratios.append(closest)
    
    # Remove duplicates from simplified_ratios while preserving order
    unique_ratios = []
    for r in simplified_ratios:
        if r not in unique_ratios:
            unique_ratios.append(r)
    
    # If we have less than 3 unique ratios, add more from common_ratios
    while len(unique_ratios) < 3:
        for r in common_ratios:
            if r not in unique_ratios:
                unique_ratios.append(r)
                break
    
    # Round sizes to nearest multiple of 4 for efficiency
    rounded_sizes = [max(4, 4 * round(s/4)) for s in rep_sizes]
    
    # Prepare suggestions
    print("\n=== Suggested Anchor Settings ===")
    print(f"Suggested anchor sizes: {rounded_sizes}")
    print(f"Suggested aspect ratios: {unique_ratios[:3]}")
    
    # Code snippet for model definition
    code_snippet = f"""
# Replace your current anchor generator with this:
def _digit_anchorgen():
    anchor_sizes = {tuple((s,) for s in rounded_sizes)}
    aspect_ratios = {tuple(unique_ratios[:3])} * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)
"""
    print("\nCode snippet to use in your model:")
    print(code_snippet)
    
    return rounded_sizes, unique_ratios[:3]

def main():
    parser = argparse.ArgumentParser(description='Analyze bounding box and image dimensions in COCO format dataset')
    parser.add_argument('--json_file', type=str, required=True, help='Path to COCO format JSON file')
    parser.add_argument('--output_dir', type=str, default='./bbox_analysis', help='Directory to save visualizations')
    parser.add_argument('--by_category', action='store_true', help='Show statistics by category')
    
    args = parser.parse_args()
    
    df, img_df, categories = analyze_bbox_dimensions(args.json_file)
    print_statistics(df, img_df, args.by_category, categories)
    visualize_dimensions(df, img_df, args.output_dir, categories)
    suggest_anchor_settings(df)
    
    # Save processed data for further analysis if needed
    df.to_csv(os.path.join(args.output_dir, 'bbox_stats.csv'), index=False)
    img_df.to_csv(os.path.join(args.output_dir, 'image_stats.csv'), index=False)
    print(f"Saved statistics to {os.path.join(args.output_dir, 'bbox_stats.csv')} and {os.path.join(args.output_dir, 'image_stats.csv')}")

if __name__ == "__main__":
    main()