"""
Complete data preprocessing pipeline for Akkadian-English translation.

Uses existing preprocessors from src/preprocessing.py:
- AkkadianPreprocessor: Handles scribal notations, Unicode normalization, etc.
- EnglishPreprocessor: Handles English translation cleanup

Applies all transformations from DatasetInstructions.md:
- Remove scribal notations
- Normalize Unicode characters
- Handle determinatives
- Standardize gaps/breaks
- Clean whitespace

Saves processed data to data/processed/train_clean.csv
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import sys

logger = logging.getLogger(__name__)


def load_preprocessors():
    """
    Load preprocessing classes from preprocessing.py
    
    Returns:
        Tuple of (AkkadianPreprocessor, EnglishPreprocessor)
    """
    sys.path.insert(0, str(Path(__file__).parent))
    
    from preprocessing import (
        AkkadianPreprocessor,
        EnglishPreprocessor,
        DeterminativeHandling
    )
    
    # Initialize preprocessors with recommended settings
    ak_processor = AkkadianPreprocessor(
        remove_scribal_marks=True,
        normalize_unicode=True,
        handle_determinatives=DeterminativeHandling.NORMALIZE,
        normalize_gaps=True,
        normalize_subscripts=True,
    )
    
    en_processor = EnglishPreprocessor(
        remove_scribal_marks=True,
        lowercase=False,  # Keep case information
    )
    
    return ak_processor, en_processor


def preprocess_training_data(
    input_file: str = "data/raw/train.csv",
    output_file: str = "data/processed/train_clean.csv",
    sample_limit: int = None,
) -> Tuple[int, int]:
    """
    Preprocess the complete training dataset.
    
    Args:
        input_file: Path to raw training CSV
        output_file: Path to save cleaned CSV
        sample_limit: Maximum number of samples to process (None = all)
        
    Returns:
        Tuple of (total_processed, total_dropped)
    """
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting preprocessing: {input_file} -> {output_file}")
    
    # Load preprocessors
    ak_processor, en_processor = load_preprocessors()
    
    try:
        # Load raw data
        logger.info(f"Loading raw training data from {input_file}...")
        df = pd.read_csv(
            input_file,
            encoding='utf-8',
            dtype={'oare_id': str, 'transliteration': str, 'translation': str}
        )
        
        logger.info(f"Loaded {len(df)} samples")
        
        if sample_limit:
            df = df.head(sample_limit)
            logger.info(f"Limited to {len(df)} samples")
        
        # Initialize processed lists
        processed_ids = []
        processed_sources = []
        processed_targets = []
        
        total_input = len(df)
        total_dropped = 0
        
        # Process each row
        logger.info("Processing samples...")
        for idx, (_, row) in enumerate(df.iterrows()):
            if (idx + 1) % 500 == 0:
                logger.info(f"  Processed {idx + 1} / {total_input} samples...")
            
            sample_id = row['oare_id']
            source_text = row['transliteration']
            target_text = row['translation']
            
            # Skip if missing
            if pd.isna(source_text) or pd.isna(target_text):
                total_dropped += 1
                logger.debug(f"Skipping sample {sample_id}: missing text")
                continue
            
            # Convert to string if needed
            source_text = str(source_text).strip()
            target_text = str(target_text).strip()
            
            # Skip if empty
            if not source_text or not target_text:
                total_dropped += 1
                logger.debug(f"Skipping sample {sample_id}: empty after stripping")
                continue
            
            try:
                # Apply preprocessing pipeline
                processed_source = ak_processor.preprocess(source_text)
                processed_target = en_processor.preprocess(target_text)
                
                # Skip if becomes empty after preprocessing
                if not processed_source or not processed_target:
                    total_dropped += 1
                    logger.debug(f"Skipping sample {sample_id}: empty after preprocessing")
                    continue
                
                # Store processed data
                processed_ids.append(sample_id)
                processed_sources.append(processed_source)
                processed_targets.append(processed_target)
                
            except Exception as e:
                total_dropped += 1
                logger.warning(f"Error processing sample {sample_id}: {str(e)}")
                continue
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'oare_id': processed_ids,
            'transliteration': processed_sources,
            'translation': processed_targets,
        })
        
        logger.info(f"Processed {len(output_df)} samples successfully")
        logger.info(f"Dropped {total_dropped} samples")
        
        # Save to CSV
        logger.info(f"Saving processed data to {output_file}...")
        output_df.to_csv(
            output_file,
            index=False,
            encoding='utf-8'
        )
        
        logger.info(f"Successfully saved {len(output_df)} samples to {output_file}")
        
        # Print statistics
        logger.info("\n=== Preprocessing Statistics ===")
        logger.info(f"Total input samples: {total_input}")
        logger.info(f"Total processed samples: {len(output_df)}")
        logger.info(f"Total dropped samples: {total_dropped}")
        logger.info(f"Keep rate: {100 * len(output_df) / total_input:.1f}%")
        
        return len(output_df), total_dropped
        
    except Exception as e:
        logger.error(f"Fatal error during preprocessing: {str(e)}", exc_info=True)
        raise


def generate_preprocessing_report(
    raw_file: str = "data/raw/train.csv",
    cleaned_file: str = "data/processed/train_clean.csv",
    report_file: str = "data/processed/preprocessing_report.txt",
) -> None:
    """
    Generate report comparing raw and cleaned data.
    
    Args:
        raw_file: Path to raw training data
        cleaned_file: Path to cleaned training data
        report_file: Path to save report
    """
    
    logger.info(f"Generating preprocessing report...")
    
    try:
        # Load both datasets
        raw_df = pd.read_csv(raw_file, encoding='utf-8')
        cleaned_df = pd.read_csv(cleaned_file, encoding='utf-8')
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PREPROCESSING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Basic statistics
        report_lines.append("DATASET STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Raw samples: {len(raw_df)}")
        report_lines.append(f"Cleaned samples: {len(cleaned_df)}")
        report_lines.append(f"Samples removed: {len(raw_df) - len(cleaned_df)}")
        report_lines.append(f"Keep rate: {100 * len(cleaned_df) / len(raw_df):.1f}%")
        report_lines.append("")
        
        # Length analysis
        raw_source_lengths = [len(str(s)) for s in raw_df['transliteration']]
        raw_target_lengths = [len(str(t)) for t in raw_df['translation']]
        cleaned_source_lengths = [len(str(s)) for s in cleaned_df['transliteration']]
        cleaned_target_lengths = [len(str(t)) for t in cleaned_df['translation']]
        
        report_lines.append("SOURCE TEXT LENGTH ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"Raw - Min: {min(raw_source_lengths)}, Max: {max(raw_source_lengths)}, Mean: {sum(raw_source_lengths)/len(raw_source_lengths):.1f}")
        report_lines.append(f"Cleaned - Min: {min(cleaned_source_lengths)}, Max: {max(cleaned_source_lengths)}, Mean: {sum(cleaned_source_lengths)/len(cleaned_source_lengths):.1f}")
        report_lines.append(f"Average reduction: {100 * (1 - sum(cleaned_source_lengths)/sum(raw_source_lengths)):.1f}%")
        report_lines.append("")
        
        report_lines.append("TARGET TEXT LENGTH ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"Raw - Min: {min(raw_target_lengths)}, Max: {max(raw_target_lengths)}, Mean: {sum(raw_target_lengths)/len(raw_target_lengths):.1f}")
        report_lines.append(f"Cleaned - Min: {min(cleaned_target_lengths)}, Max: {max(cleaned_target_lengths)}, Mean: {sum(cleaned_target_lengths)/len(cleaned_target_lengths):.1f}")
        report_lines.append("")
        
        # Save report
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved preprocessing report to {report_file}")
        
        # Print summary
        logger.info("\n" + "\n".join(report_lines))
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)


def main():
    """Main preprocessing pipeline."""
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        logger.info("Starting full preprocessing pipeline...")
        
        # Run preprocessing
        processed_count, dropped_count = preprocess_training_data(
            input_file="data/raw/train.csv",
            output_file="data/processed/train_clean.csv",
            sample_limit=None,  # Process all samples
        )
        
        # Generate report
        logger.info("Generating preprocessing report...")
        generate_preprocessing_report(
            raw_file="data/raw/train.csv",
            cleaned_file="data/processed/train_clean.csv",
            report_file="data/processed/preprocessing_report.txt",
        )
        
        logger.info("Preprocessing pipeline complete!")
        logger.info(f"Successfully processed {processed_count} samples, dropped {dropped_count} samples")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
