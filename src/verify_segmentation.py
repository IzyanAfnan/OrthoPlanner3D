# src/verify_segmentation.py
# Run this after every TotalSegmentator segmentation to confirm output quality.

import nibabel as nib
import numpy as np

def verify_segmentation(seg_path: str, expected_bones: dict) -> bool:
    """
    Loads a segmentation file and checks that  epected bone labels are present.

    Args:
        seg_path: Path to the segmentation NIfTI file
        
        expected_bones: dict of {label_id : bone_name} you expect to find (e.g. {5: "liver", 115: "L5 vertebra"})
    
    Returns:
        True if all expected labels found, Flase otherwise
    """

    seg_img= nib.load(seg_path)
    seg_data= seg_img.get_fdata().astype(int)
    found_labels= np.unique(seg_data)

    print(f"\nVerifying: {seg_path}")
    print(f"All label IDs present: {found_labels}")
    print(f"Total non-background voxels: {np.sum(seg_data > 0)}")

    all_found= True
    for label_id, bone_name in expected_bones.items():
        count= np.sum(seg_data == label_id)
        status= "✓" if  count > 0 else "x MISSING"
        print(f" {status} Label {label_id} ({bone_name}): {count} voxels")
        if count == 0:
            all_found= False

    return all_found

if __name__ == "__main__":
    # Fill in the label IDs you discovered on Day 5 from np.unique()
    # These are placeholders — replace with your actual observed label IDs
    total_expected = {
        75 : "femur_left",
        76 : "femur_right",

    }

    appendicular_expected = {
        1: "patella",        
        2: "tibia",          
        3: "fibula",
        4: "tarsal", 
    }
    
    verify_segmentation(
        "./data/segmentations/patient_01_total.nii.gz",
        total_expected
    )

    verify_segmentation(
        "./data/segmentations/patient_01_appendicular.nii.gz",
        appendicular_expected
    )