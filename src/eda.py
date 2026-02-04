"""
Exploratory Data Analysis (EDA) module for Akkadian-English translation project.

Analyzes:
- Text length distributions (source and target)
- Word count statistics
- Special character frequency
- Formatting patterns
- Proper noun coverage
- Gap/break patterns

Generates comprehensive report to data/processed/eda_report.txt
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import sys

logger = logging.getLogger(__name__)


class AkkadianEDA:
    """
    Exploratory Data Analysis for Akkadian-English corpus.
    """
    
    def __init__(self, output_dir: str = "data/processed/"):
        """
        Initialize EDA analyzer.
        
        Args:
            output_dir: Directory to save report
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_lines: List[str] = []
    
    def add_section(self, title: str):
        """Add a section header to report."""
        self.report_lines.append(f"\n{'=' * 80}")
        self.report_lines.append(f"{title:^80}")
        self.report_lines.append(f"{'=' * 80}\n")
    
    def add_line(self, text: str = ""):
        """Add a line to report."""
        self.report_lines.append(text)
    
    def analyze_text_lengths(self, sources: List[str], targets: List[str]) -> Dict[str, Any]:
        """
        Analyze text length distributions.
        
        Args:
            sources: List of source texts
            targets: List of target texts
            
        Returns:
            Dictionary with length statistics
        """
        source_lengths = [len(text) for text in sources]
        target_lengths = [len(text) for text in targets]
        
        source_word_counts = [len(text.split()) for text in sources]
        target_word_counts = [len(text.split()) for text in targets]
        
        stats = {
            'source': {
                'char_min': min(source_lengths),
                'char_max': max(source_lengths),
                'char_mean': sum(source_lengths) / len(source_lengths),
                'char_median': sorted(source_lengths)[len(source_lengths) // 2],
                'word_min': min(source_word_counts),
                'word_max': max(source_word_counts),
                'word_mean': sum(source_word_counts) / len(source_word_counts),
                'word_median': sorted(source_word_counts)[len(source_word_counts) // 2],
            },
            'target': {
                'char_min': min(target_lengths),
                'char_max': max(target_lengths),
                'char_mean': sum(target_lengths) / len(target_lengths),
                'char_median': sorted(target_lengths)[len(target_lengths) // 2],
                'word_min': min(target_word_counts),
                'word_max': max(target_word_counts),
                'word_mean': sum(target_word_counts) / len(target_word_counts),
                'word_median': sorted(target_word_counts)[len(target_word_counts) // 2],
            }
        }
        
        self.add_section("TEXT LENGTH ANALYSIS")
        self.add_line("Source (Akkadian) Texts:")
        self.add_line(f"  Characters: min={stats['source']['char_min']}, max={stats['source']['char_max']}, mean={stats['source']['char_mean']:.2f}, median={stats['source']['char_median']}")
        self.add_line(f"  Words: min={stats['source']['word_min']}, max={stats['source']['word_max']}, mean={stats['source']['word_mean']:.2f}, median={stats['source']['word_median']}")
        
        self.add_line("\nTarget (English) Texts:")
        self.add_line(f"  Characters: min={stats['target']['char_min']}, max={stats['target']['char_max']}, mean={stats['target']['char_mean']:.2f}, median={stats['target']['char_median']}")
        self.add_line(f"  Words: min={stats['target']['word_min']}, max={stats['target']['word_max']}, mean={stats['target']['word_mean']:.2f}, median={stats['target']['word_median']}")
        
        return stats
    
    def analyze_special_characters(self, sources: List[str]) -> Dict[str, int]:
        """
        Analyze frequency of special characters in Akkadian texts.
        
        Args:
            sources: List of source texts
            
        Returns:
            Dictionary mapping characters to frequency
        """
        char_freq = Counter()
        special_chars = {}
        
        for text in sources:
            for char in text:
                if not char.isalnum() and not char.isspace() and char != '-':
                    char_freq[char] += 1
                    special_chars[char] = char_freq[char]
        
        self.add_section("SPECIAL CHARACTER ANALYSIS")
        self.add_line(f"Total unique special characters: {len(char_freq)}\n")
        self.add_line("Top 30 special characters by frequency:")
        
        for i, (char, count) in enumerate(char_freq.most_common(30), 1):
            self.add_line(f"  {i:2d}. '{char}': {count:6d} occurrences")
        
        return dict(char_freq)
    
    def analyze_determinatives(self, sources: List[str]) -> Dict[str, int]:
        """
        Analyze determinative patterns in Akkadian texts.
        
        Args:
            sources: List of source texts
            
        Returns:
            Dictionary mapping determinatives to frequency
        """
        det_pattern = r'\{[^}]+\}'
        det_freq = Counter()
        
        for text in sources:
            dets = re.findall(det_pattern, text)
            for det in dets:
                det_freq[det] += 1
        
        self.add_section("DETERMINATIVE ANALYSIS")
        self.add_line(f"Total unique determinatives: {len(det_freq)}")
        self.add_line(f"Total determinative occurrences: {sum(det_freq.values())}\n")
        self.add_line("Determinatives by frequency:")
        
        for i, (det, count) in enumerate(det_freq.most_common(), 1):
            self.add_line(f"  {i:2d}. {det:20s}: {count:6d} occurrences")
        
        return dict(det_freq)
    
    def analyze_proper_nouns(self, sources: List[str]) -> Dict[str, int]:
        """
        Analyze proper nouns and capitalization patterns.
        
        Args:
            sources: List of source texts
            
        Returns:
            Dictionary mapping proper nouns to frequency
        """
        proper_nouns = Counter()
        sumerian_logograms = Counter()
        
        for text in sources:
            # Find capitalized words (proper nouns)
            cap_words = re.findall(r'\b[A-Z][a-zñ0-9-]*\b', text)
            for word in cap_words:
                if len(word) > 1:
                    proper_nouns[word] += 1
            
            # Find ALL CAPS (Sumerian logograms)
            caps_words = re.findall(r'\b[A-Z][A-Z0-9₀-₉.]*\b', text)
            for word in caps_words:
                if len(word) > 1 and word.isupper():
                    sumerian_logograms[word] += 1
        
        self.add_section("PROPER NOUNS AND LOGOGRAMS")
        self.add_line(f"Total unique proper nouns: {len(proper_nouns)}")
        self.add_line(f"Total proper noun occurrences: {sum(proper_nouns.values())}")
        self.add_line(f"Average occurrences per proper noun: {sum(proper_nouns.values()) / len(proper_nouns) if proper_nouns else 0:.2f}\n")
        
        self.add_line("Top 20 proper nouns:")
        for i, (noun, count) in enumerate(proper_nouns.most_common(20), 1):
            self.add_line(f"  {i:2d}. {noun:30s}: {count:6d} occurrences")
        
        self.add_line(f"\n\nTotal unique Sumerian logograms: {len(sumerian_logograms)}")
        self.add_line(f"Total logogram occurrences: {sum(sumerian_logograms.values())}\n")
        
        self.add_line("Top 20 Sumerian logograms:")
        for i, (logo, count) in enumerate(sumerian_logograms.most_common(20), 1):
            self.add_line(f"  {i:2d}. {logo:30s}: {count:6d} occurrences")
        
        return {'proper_nouns': dict(proper_nouns), 'logograms': dict(sumerian_logograms)}
    
    def analyze_gaps(self, sources: List[str]) -> Dict[str, int]:
        """
        Analyze gap and break patterns.
        
        Args:
            sources: List of source texts
            
        Returns:
            Dictionary with gap statistics
        """
        gap_pattern = r'<gap>|<big_gap>'
        gap_freq = Counter()
        
        for text in sources:
            gaps = re.findall(gap_pattern, text)
            for gap in gaps:
                gap_freq[gap] += 1
        
        texts_with_gaps = sum(1 for text in sources if re.search(gap_pattern, text))
        
        self.add_section("GAP/BREAK ANALYSIS")
        self.add_line(f"Texts with gaps/breaks: {texts_with_gaps} / {len(sources)} ({100*texts_with_gaps/len(sources):.1f}%)")
        self.add_line(f"Total gaps found: {sum(gap_freq.values())}\n")
        
        for gap_type, count in gap_freq.most_common():
            self.add_line(f"  {gap_type}: {count} occurrences")
        
        return dict(gap_freq)
    
    def analyze_abbreviations(self, targets: List[str]) -> Dict[str, int]:
        """
        Analyze abbreviations and patterns in English translations.
        
        Args:
            targets: List of target texts
            
        Returns:
            Dictionary with abbreviation statistics
        """
        abbrev_pattern = r'\b[A-Z]+\b'  # ALL CAPS words (likely abbreviations)
        abbrev_freq = Counter()
        
        for text in targets:
            abbrevs = re.findall(abbrev_pattern, text)
            for abbrev in abbrevs:
                if len(abbrev) > 1:
                    abbrev_freq[abbrev] += 1
        
        texts_with_abbrevs = sum(1 for text in targets if re.search(abbrev_pattern, text))
        
        self.add_section("ENGLISH TEXT ANALYSIS")
        self.add_line(f"Texts with abbreviations: {texts_with_abbrevs} / {len(targets)} ({100*texts_with_abbrevs/len(targets):.1f}%)")
        self.add_line(f"Total abbreviations found: {sum(abbrev_freq.values())}\n")
        
        self.add_line("Top 20 abbreviations:")
        for i, (abbrev, count) in enumerate(abbrev_freq.most_common(20), 1):
            self.add_line(f"  {i:2d}. {abbrev:15s}: {count:6d} occurrences")
        
        return dict(abbrev_freq)
    
    def save_report(self, filename: str = "eda_report.txt"):
        """
        Save report to file.
        
        Args:
            filename: Output filename
        """
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        
        logger.info(f"Saved EDA report to {output_file}")


def main():
    """Main function for EDA."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data_loader import DataLoader
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        logger.info("Starting EDA analysis...")
        
        # Load data
        logger.info("Loading training data...")
        loader = DataLoader(data_dir="data/raw/")
        train_sources = loader.get_train_sources()
        train_targets = loader.get_train_targets()
        
        logger.info(f"Loaded {len(train_sources)} training samples")
        
        # Initialize EDA analyzer
        eda = AkkadianEDA(output_dir="data/processed/")
        
        # Add header
        eda.add_section("AKKADIAN-ENGLISH TRANSLATION CORPUS ANALYSIS")
        eda.add_line(f"Analysis Date: from EDA script")
        eda.add_line(f"Total training samples: {len(train_sources)}\n")
        
        # Run analyses
        logger.info("Analyzing text lengths...")
        eda.analyze_text_lengths(train_sources, train_targets)
        
        logger.info("Analyzing special characters...")
        eda.analyze_special_characters(train_sources)
        
        logger.info("Analyzing determinatives...")
        eda.analyze_determinatives(train_sources)
        
        logger.info("Analyzing proper nouns...")
        eda.analyze_proper_nouns(train_sources)
        
        logger.info("Analyzing gaps and breaks...")
        eda.analyze_gaps(train_sources)
        
        logger.info("Analyzing English translations...")
        eda.analyze_abbreviations(train_targets)
        
        # Save report
        logger.info("Saving report...")
        eda.save_report("eda_report.txt")
        
        logger.info("EDA analysis complete!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
