import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def create_coco_json(image_dir, mask_dir, output_file):
    images = []
    annotations = []
    categories = [{"id": 1, "name": "defect", "supercategory": "none"}]

    image_id = 1
    annotation_id = 1

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)

        if not os.path.exists(mask_path):
            continue

        image = Image.open(image_path)
        width, height = image.size

        images.append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)

        # Generate bbox and segmentation from mask
        if np.any(mask_np > 0):
            segmentation = coco_mask.encode(np.asfortranarray(mask_np))
            bbox = coco_mask.toBbox(segmentation)

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": coco_mask.decode(segmentation).tolist(),
                "area": int(np.sum(mask_np > 0)),
                "bbox": bbox.tolist(),
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    coco_format = {
        "info": {"description": "Defect Dataset", "version": "1.0", "year": 2024},
        "licenses": [{"id": 1, "name": "Attribution-NonCommercial 4.0 International",
                      "url": "http://creativecommons.org/licenses/by-nc/4.0/"}],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)


# Usage
create_coco_json('../data\crack_segmentation_dataset/train/images', '../data\crack_segmentation_dataset/train/masks', 'output.json')
