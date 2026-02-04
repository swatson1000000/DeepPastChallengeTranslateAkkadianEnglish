"""
Data loader module for Akkadian-English parallel corpus.

Handles loading and parsing CSV files from data/raw/ directory.
Provides DataLoader class for batch processing of training and test data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParallelPair:
    """Container for source-target translation pair."""
    source: str
    target: Optional[str] = None
    sample_id: Optional[str] = None


class DataLoader:
    """
    Data loader for Akkadian-English corpus.
    
    Loads training data (train.csv) and test data (test.csv) from data/raw/ directory.
    Provides methods for batch processing and data access.
    """
    
    def __init__(self, data_dir: str = "data/raw/"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing raw CSV files
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"Initialized DataLoader with data_dir: {self.data_dir}")
        
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Load training data from train.csv.
        
        Expected columns:
        - oare_id: Unique identifier
        - transliteration: Akkadian transliterated text (source)
        - translation: English translation (target)
        
        Returns:
            DataFrame with training data
            
        Raises:
            FileNotFoundError: If train.csv not found
            ValueError: If required columns missing
        """
        train_file = self.data_dir / "train.csv"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        
        try:
            self.train_data = pd.read_csv(
                train_file,
                encoding='utf-8',
                dtype={'oare_id': str, 'transliteration': str, 'translation': str}
            )
            
            # Validate required columns
            required_cols = ['oare_id', 'transliteration', 'translation']
            missing_cols = [col for col in required_cols if col not in self.train_data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Handle missing values
            initial_count = len(self.train_data)
            self.train_data = self.train_data.dropna(subset=['transliteration', 'translation'])
            dropped_count = initial_count - len(self.train_data)
            
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} rows with missing translations or transliteration")
            
            logger.info(f"Loaded {len(self.train_data)} training samples from {train_file}")
            
            return self.train_data
            
        except Exception as e:
            logger.error(f"Error loading train data: {str(e)}")
            raise
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data from test.csv.
        
        Expected columns:
        - id: Sample identifier (0-indexed or unique)
        - text_id: Text identifier
        - line_start: Starting line number
        - line_end: Ending line number
        - transliteration: Akkadian transliterated text (source, no target)
        
        Returns:
            DataFrame with test data
            
        Raises:
            FileNotFoundError: If test.csv not found
            ValueError: If required columns missing
        """
        test_file = self.data_dir / "test.csv"
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        try:
            self.test_data = pd.read_csv(
                test_file,
                encoding='utf-8',
                dtype={'id': str, 'transliteration': str}
            )
            
            # Validate required columns
            required_cols = ['id', 'transliteration']
            missing_cols = [col for col in required_cols if col not in self.test_data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Handle missing values
            initial_count = len(self.test_data)
            self.test_data = self.test_data.dropna(subset=['transliteration'])
            dropped_count = initial_count - len(self.test_data)
            
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} rows with missing transliteration in test data")
            
            logger.info(f"Loaded {len(self.test_data)} test samples from {test_file}")
            
            return self.test_data
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def get_train_pairs(self) -> List[ParallelPair]:
        """
        Get training data as list of ParallelPair objects.
        
        Returns:
            List of ParallelPair with source (transliteration) and target (translation)
        """
        if self.train_data is None:
            self.load_train_data()
        
        pairs = [
            ParallelPair(
                source=row['transliteration'],
                target=row['translation'],
                sample_id=row['oare_id']
            )
            for _, row in self.train_data.iterrows()
        ]
        
        return pairs
    
    def get_test_pairs(self) -> List[ParallelPair]:
        """
        Get test data as list of ParallelPair objects (no targets).
        
        Returns:
            List of ParallelPair with source (transliteration) only
        """
        if self.test_data is None:
            self.load_test_data()
        
        pairs = [
            ParallelPair(
                source=row['transliteration'],
                target=None,
                sample_id=row['id']
            )
            for _, row in self.test_data.iterrows()
        ]
        
        return pairs
    
    def get_train_batch(self, batch_size: int = 32, shuffle: bool = False) -> List[List[ParallelPair]]:
        """
        Get training data in batches.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data before batching
            
        Returns:
            List of batches, each batch is a list of ParallelPair objects
        """
        pairs = self.get_train_pairs()
        
        if shuffle:
            import random
            random.shuffle(pairs)
        
        batches = [
            pairs[i:i + batch_size]
            for i in range(0, len(pairs), batch_size)
        ]
        
        return batches
    
    def get_train_sources(self) -> List[str]:
        """Get all training source texts (Akkadian)."""
        if self.train_data is None:
            self.load_train_data()
        
        return self.train_data['transliteration'].tolist()
    
    def get_train_targets(self) -> List[str]:
        """Get all training target texts (English)."""
        if self.train_data is None:
            self.load_train_data()
        
        return self.train_data['translation'].tolist()
    
    def get_test_sources(self) -> List[str]:
        """Get all test source texts (Akkadian)."""
        if self.test_data is None:
            self.load_test_data()
        
        return self.test_data['transliteration'].tolist()
    
    def get_test_ids(self) -> List[str]:
        """Get all test sample IDs."""
        if self.test_data is None:
            self.load_test_data()
        
        return self.test_data['id'].tolist()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.train_data is None:
            self.load_train_data()
        
        if self.test_data is None:
            self.load_test_data()
        
        train_sources = self.train_data['transliteration'].tolist()
        train_targets = self.train_data['translation'].tolist()
        test_sources = self.test_data['transliteration'].tolist()
        
        stats = {
            'train_count': len(self.train_data),
            'test_count': len(self.test_data),
            'train_source_avg_length': sum(len(s) for s in train_sources) / len(train_sources) if train_sources else 0,
            'train_target_avg_length': sum(len(t) for t in train_targets) / len(train_targets) if train_targets else 0,
            'test_source_avg_length': sum(len(s) for s in test_sources) / len(test_sources) if test_sources else 0,
            'train_source_max_length': max(len(s) for s in train_sources) if train_sources else 0,
            'train_target_max_length': max(len(t) for t in train_targets) if train_targets else 0,
            'test_source_max_length': max(len(s) for s in test_sources) if test_sources else 0,
        }
        
        return stats


def main():
    """Example usage of DataLoader."""
    import json
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize loader
        loader = DataLoader(data_dir="data/raw/")
        
        # Load data
        train_df = loader.load_train_data()
        test_df = loader.load_test_data()
        
        # Get statistics
        stats = loader.get_statistics()
        print("\n=== Dataset Statistics ===")
        print(json.dumps(stats, indent=2))
        
        # Get first training example
        pairs = loader.get_train_pairs()
        if pairs:
            print("\n=== First Training Example ===")
            print(f"ID: {pairs[0].sample_id}")
            print(f"Source: {pairs[0].source[:100]}...")
            print(f"Target: {pairs[0].target[:100]}...")
        
        # Get batches
        batches = loader.get_train_batch(batch_size=32, shuffle=True)
        print(f"\n=== Batch Information ===")
        print(f"Total batches: {len(batches)}")
        print(f"Batch size: 32")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
