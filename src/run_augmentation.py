#!/usr/bin/env python3
"""
Run data augmentation to create more training data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from augment_data import DataAugmentor

def main():
    project_root = Path(__file__).parent
    
    augmentor = DataAugmentor(project_root)
    augmentor.load_training_data()
    
    # Create 5x augmented data (1,561 -> 7,805 samples)
    print("\n" + "="*80)
    print("CREATING 5X AUGMENTED TRAINING DATA")
    print("="*80)
    print("\nThis will create synthetic training pairs using:")
    print("  - 50% Paraphrase augmentation")
    print("  - 30% Translation variations")
    print("  - 20% Segment combinations")
    print("\nCurrent: 1,561 samples")
    print("Target:  9,756 samples (6.25x multiplier)")
    print("\n" + "="*80)
    
    # Generate augmented data with 6.25x multiplier
    augmented_df = augmentor.augment_training_data(multiplier=6.25)
    
    # Save to file
    output_path = project_root / "data" / "processed" / "train_augmented_5x.csv"
    augmented_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ AUGMENTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"Total samples: {len(augmented_df)}")
    
    # Show breakdown
    if 'augmentation_type' in augmented_df.columns:
        print(f"\nAugmentation breakdown:")
        print(f"  Original samples: {len(augmentor.training_data)}")
        print(f"  Paraphrasing: {sum(augmented_df['augmentation_type'] == 'paraphrase')}")
        print(f"  Variations: {sum(augmented_df['augmentation_type'] == 'variation')}")
        print(f"  Segment combinations: {sum(augmented_df['augmentation_type'] == 'segment_combination')}")
    
    print(f"\n✓ Ready to use in training with:")
    print(f"  python train.py --data {output_path}")
    print(f"\n" + "="*80)

if __name__ == '__main__':
    main()
