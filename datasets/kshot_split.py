import json
import random
import argparse

def filter_k_shot_json(input_file, output_file, k_shot):
    # Read the COCO FORMAT JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # Build a mapping from image_id to image information
    image_id_to_image = {image['id']: image for image in images}

    category_to_image_annotations = {}
    
    # Group annotations by category and image
    for annotation in annotations:
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        
        if category_id not in category_to_image_annotations:
            category_to_image_annotations[category_id] = {}
        
        if image_id not in category_to_image_annotations[category_id]:
            category_to_image_annotations[category_id][image_id] = annotation
    
    selected_annotations = []
    selected_image_ids = set()
    
    # For each category, randomly select k images and ensure 1 annotation per image
    for category_id, image_annotations in category_to_image_annotations.items():
        image_ids = list(image_annotations.keys())

        if len(image_ids) >= k_shot:
            chosen_ids = random.sample(image_ids, k_shot)
        else:
            # not enough images, sample with replacement
            chosen_ids = image_ids + random.choices(image_ids, k=k_shot - len(image_ids))

        selected_image_ids.update(chosen_ids)
        selected_annotations.extend(image_annotations[img_id] for img_id in chosen_ids)
    
    # Filter the images based on the selected image IDs
    selected_images = [image_id_to_image[image_id] for image_id in selected_image_ids]

    # Construct the new JSON data structure
    kshot_data = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": categories  # Keep all categories unchanged
    }
    
    # Write the filtered data to the new JSON file
    with open(output_file, 'w') as f:
        json.dump(kshot_data, f, indent=4)
    
    print(f"{k_shot}-shot JSON data has been saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a k-shot subset from a COCO-style dataset")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path to save the k-shot JSON")
    parser.add_argument("k", type=int, help="Number of shots per category")

    args = parser.parse_args()
    filter_k_shot_json(args.input, args.output, args.k)
