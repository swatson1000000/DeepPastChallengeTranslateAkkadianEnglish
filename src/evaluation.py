"""
Evaluation metrics module for Akkadian-English translation.

Implements:
- BLEU score calculation
- chrF++ (character n-gram F-score) calculation
- Geometric mean (competition metric)
- Batch evaluation support

Uses sacrebleu library for standardized implementations.
"""

import logging
from typing import List, Tuple, Dict, Any
import numpy as np

try:
    from sacrebleu import BLEU, CHRF
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

logger = logging.getLogger(__name__)


class TranslationMetrics:
    """
    Evaluation metrics for machine translation.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        if SACREBLEU_AVAILABLE:
            self.bleu = BLEU()
            self.chrf = CHRF()
            logger.info("Initialized sacrebleu metrics")
        else:
            logger.warning("sacrebleu not available. Install with: pip install sacrebleu")
    
    def calculate_bleu(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> float:
        """
        Calculate BLEU score.
        
        BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between
        hypothesis and reference translations.
        
        Args:
            hypotheses: List of predicted translations
            references: List of reference translations
            
        Returns:
            BLEU score (0-100)
            
        Raises:
            ImportError: If sacrebleu not available
        """
        if not SACREBLEU_AVAILABLE:
            raise ImportError("sacrebleu required. Install with: pip install sacrebleu")
        
        if len(hypotheses) != len(references):
            raise ValueError("Hypotheses and references must have same length")
        
        # sacrebleu expects references as list of lists
        refs = [[ref] for ref in references]
        
        score = self.bleu.corpus_score(hypotheses, refs)
        
        return score.score
    
    def calculate_chrf(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> float:
        """
        Calculate chrF++ score.
        
        chrF++ (character n-gram F-score++) measures character-level n-gram overlap.
        More suitable for morphologically rich languages like Akkadian.
        
        Args:
            hypotheses: List of predicted translations
            references: List of reference translations
            
        Returns:
            chrF++ score (0-100)
            
        Raises:
            ImportError: If sacrebleu not available
        """
        if not SACREBLEU_AVAILABLE:
            raise ImportError("sacrebleu required. Install with: pip install sacrebleu")
        
        if len(hypotheses) != len(references):
            raise ValueError("Hypotheses and references must have same length")
        
        # sacrebleu expects references as list of lists
        refs = [[ref] for ref in references]
        
        score = self.chrf.corpus_score(hypotheses, refs)
        
        return score.score
    
    def calculate_geometric_mean(
        self,
        bleu_score: float,
        chrf_score: float,
    ) -> float:
        """
        Calculate geometric mean of BLEU and chrF++ scores.
        
        This is the competition metric for the Deep Past Challenge.
        
        Geometric mean = sqrt(BLEU * chrF++)
        
        Args:
            bleu_score: BLEU score (0-100)
            chrf_score: chrF++ score (0-100)
            
        Returns:
            Geometric mean score
        """
        # Convert percentages to 0-1 range
        bleu_normalized = bleu_score / 100.0
        chrf_normalized = chrf_score / 100.0
        
        # Calculate geometric mean
        geom_mean = np.sqrt(bleu_normalized * chrf_normalized)
        
        # Convert back to 0-100 range
        return geom_mean * 100.0
    
    def evaluate_batch(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate a batch of translations with all metrics.
        
        Args:
            hypotheses: List of predicted translations
            references: List of reference translations
            
        Returns:
            Dictionary with metrics:
                - bleu: BLEU score
                - chrf: chrF++ score
                - geometric_mean: Geometric mean (competition metric)
        """
        if not SACREBLEU_AVAILABLE:
            raise ImportError("sacrebleu required. Install with: pip install sacrebleu")
        
        bleu = self.calculate_bleu(hypotheses, references)
        chrf = self.calculate_chrf(hypotheses, references)
        geom_mean = self.calculate_geometric_mean(bleu, chrf)
        
        results = {
            'bleu': bleu,
            'chrf': chrf,
            'geometric_mean': geom_mean,
        }
        
        return results
    
    def evaluate_sample(
        self,
        hypothesis: str,
        reference: str,
    ) -> Dict[str, float]:
        """
        Evaluate a single translation pair.
        
        Args:
            hypothesis: Predicted translation
            reference: Reference translation
            
        Returns:
            Dictionary with metrics
        """
        return self.evaluate_batch([hypothesis], [reference])
    
    @staticmethod
    def simple_bleu(
        hypotheses: List[str],
        references: List[str],
        max_n: int = 4,
    ) -> float:
        """
        Calculate simple BLEU score without sacrebleu (fallback).
        
        Args:
            hypotheses: List of predicted translations
            references: List of reference translations
            max_n: Maximum n-gram size
            
        Returns:
            BLEU score
        """
        from collections import Counter
        
        def get_ngrams(text: str, n: int) -> Counter:
            """Extract n-grams from text."""
            tokens = text.split()
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(tuple(tokens[i:i+n]))
            return Counter(ngrams)
        
        total_score = 0.0
        
        for hyp, ref in zip(hypotheses, references):
            hyp_tokens = hyp.split()
            ref_tokens = ref.split()
            
            # Calculate brevity penalty
            if len(hyp_tokens) == 0:
                score = 0.0
            elif len(hyp_tokens) > len(ref_tokens):
                bp = 1.0
            else:
                bp = np.exp(1 - len(ref_tokens) / len(hyp_tokens))
            
            # Calculate n-gram precision
            precisions = []
            for n in range(1, max_n + 1):
                hyp_ngrams = get_ngrams(hyp, n)
                ref_ngrams = get_ngrams(ref, n)
                
                if len(hyp_ngrams) == 0:
                    precision = 0.0
                else:
                    matches = sum((hyp_ngrams & ref_ngrams).values())
                    precision = matches / sum(hyp_ngrams.values())
                
                precisions.append(precision)
            
            # Geometric mean of precisions
            if all(p > 0 for p in precisions):
                geo_mean = np.exp(sum(np.log(p) for p in precisions) / len(precisions))
            else:
                geo_mean = 0.0
            
            score = bp * geo_mean
            total_score += score
        
        # Average across batch
        avg_score = total_score / len(hypotheses)
        
        # Convert to 0-100 scale
        return avg_score * 100.0


class EvaluationReport:
    """
    Generate evaluation reports.
    """
    
    def __init__(self):
        """Initialize report generator."""
        self.metrics = TranslationMetrics()
    
    def generate_report(
        self,
        hypotheses: List[str],
        references: List[str],
        output_file: str = "evaluation_report.txt",
    ) -> Dict[str, float]:
        """
        Generate evaluation report and save to file.
        
        Args:
            hypotheses: List of predicted translations
            references: List of reference translations
            output_file: Path to save report
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = self.metrics.evaluate_batch(hypotheses, references)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TRANSLATION EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Total samples evaluated: {len(hypotheses)}")
        report_lines.append("")
        
        report_lines.append("RESULTS")
        report_lines.append("-" * 80)
        report_lines.append(f"BLEU Score:          {results['bleu']:6.2f}")
        report_lines.append(f"chrF++ Score:        {results['chrf']:6.2f}")
        report_lines.append(f"Geometric Mean:      {results['geometric_mean']:6.2f}  (Competition Metric)")
        report_lines.append("")
        
        report_lines.append("NOTES")
        report_lines.append("-" * 80)
        report_lines.append("BLEU: Measures n-gram overlap (word-level)")
        report_lines.append("chrF++: Measures character-level n-gram overlap (better for morphology)")
        report_lines.append("Geometric Mean: sqrt(BLEU * chrF++) - used for competition ranking")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved evaluation report to {output_file}")
        logger.info("\n" + "\n".join(report_lines))
        
        return results


def main():
    """Example usage."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example translations
    references = [
        "The man went to the market",
        "She has a beautiful house",
        "They are playing in the garden",
    ]
    
    hypotheses = [
        "The man goes to market",
        "She has beautiful house",
        "They play in garden",
    ]
    
    try:
        metrics = TranslationMetrics()
        results = metrics.evaluate_batch(hypotheses, references)
        
        print("\nEvaluation Results:")
        print(f"BLEU:              {results['bleu']:.2f}")
        print(f"chrF++:            {results['chrf']:.2f}")
        print(f"Geometric Mean:    {results['geometric_mean']:.2f}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.info("Trying simple BLEU calculation...")
        
        bleu = TranslationMetrics.simple_bleu(hypotheses, references)
        print(f"Simple BLEU Score: {bleu:.2f}")


if __name__ == '__main__':
    main()
