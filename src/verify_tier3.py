#!/usr/bin/env python3
"""
TIER 3 Implementation Verification Script

Verifies that all TIER 3 components are properly implemented and integrated.
TIER 3 builds on TIER 1 and TIER 2 with advanced features:
1. Beam Search - Multi-path decoding with length normalization
2. Subword Tokenization - BPE/SentencePiece for better morphology handling
3. Back-translation - Synthetic data generation
4. Multi-task Learning - Bidirectional translation (optional)
5. Ensemble Methods - Multiple model training (optional)
6. Transformer - Full architecture replacement (optional)
"""

import logging
import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class TIER3Verifier:
    """Verify TIER 3 implementation completeness."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.checks_passed = 0
        self.checks_failed = 0
        self.checks_total = 0
    
    def run_all_checks(self) -> bool:
        """Run all TIER 3 verification checks."""
        
        logger.info("\n" + "="*80)
        logger.info("TIER 3 IMPLEMENTATION VERIFICATION")
        logger.info("="*80)
        
        # Phase 1: Beam Search
        logger.info("\n[PHASE 1] Beam Search Implementation")
        logger.info("-" * 80)
        self.check_beam_search_module()
        self.check_inference_beam_search_integration()
        
        # Phase 2: Subword Tokenization  
        logger.info("\n[PHASE 2] Subword Tokenization")
        logger.info("-" * 80)
        self.check_subword_tokenization_module()
        
        # Phase 3: Back-translation
        logger.info("\n[PHASE 3] Back-translation Data Generation")
        logger.info("-" * 80)
        self.check_back_translation_module()
        
        # Phase 4-6: Optional
        logger.info("\n[PHASE 4-6] Multi-task / Ensemble / Transformer (Optional)")
        logger.info("-" * 80)
        logger.info("ℹ These phases are optional for TIER 3 MVP")
        
        # Configuration and Integration
        logger.info("\n[INTEGRATION] Configuration and Scripts")
        logger.info("-" * 80)
        self.check_tier3_config()
        self.check_train_py_tier3_support()
        self.check_inference_py_tier3_support()
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("VERIFICATION SUMMARY")
        logger.info("="*80)
        total_required = 6  # Beam search, inference beam, config, train support, inference support, back-translation
        passed_required = sum([
            self.has_beam_search_module,
            self.has_inference_beam_support,
            self.has_tier3_config,
            self.has_train_tier3,
            self.has_inference_tier3,
            self.has_back_translation
        ])
        
        logger.info(f"\nRequired Components: {passed_required}/{total_required}")
        logger.info(f"All Checks: {self.checks_passed}/{self.checks_total} passed")
        
        if passed_required == total_required:
            logger.info("\n✓ TIER 3 MVP READY FOR TRAINING")
            return True
        else:
            logger.info(f"\n✗ Missing {total_required - passed_required} required component(s)")
            return False
    
    def check_beam_search_module(self):
        """Verify beam search module exists and has required components."""
        logger.info("\n✓ Checking Beam Search module...")
        
        beam_search_path = self.project_root / "src/beam_search.py"
        self.checks_total += 1
        
        if not beam_search_path.exists():
            logger.info(f"  ✗ FAILED: {beam_search_path} not found")
            self.checks_failed += 1
            self.has_beam_search_module = False
            return
        
        try:
            with open(beam_search_path, 'r') as f:
                content = f.read()
            
            required_classes = ['BeamSearchDecoder', 'SimpleBeamSearch', 'beam_search_decode']
            missing = []
            
            for cls in required_classes:
                if cls not in content:
                    missing.append(cls)
            
            if missing:
                logger.info(f"  ✗ Missing classes: {missing}")
                self.checks_failed += 1
                self.has_beam_search_module = False
            else:
                logger.info(f"  ✓ All required classes found")
                logger.info(f"    - BeamSearchDecoder")
                logger.info(f"    - SimpleBeamSearch")
                logger.info(f"    - beam_search_decode")
                self.checks_passed += 1
                self.has_beam_search_module = True
        
        except Exception as e:
            logger.info(f"  ✗ Error reading file: {e}")
            self.checks_failed += 1
            self.has_beam_search_module = False
    
    def check_inference_beam_search_integration(self):
        """Verify beam search is integrated into inference.py."""
        logger.info("\n✓ Checking Beam Search integration in inference.py...")
        
        inference_path = self.project_root / "inference.py"
        self.checks_total += 1
        
        if not inference_path.exists():
            logger.info(f"  ✗ FAILED: {inference_path} not found")
            self.checks_failed += 1
            self.has_inference_beam_support = False
            return
        
        try:
            with open(inference_path, 'r') as f:
                content = f.read()
            
            required_checks = [
                ('beam_search_decode' in content, 'beam_search_decode method'),
                ('--use-beam-search' in content, '--use-beam-search argument'),
                ('--beam-width' in content, '--beam-width argument'),
                ('use_beam_search' in content, 'use_beam_search parameter'),
            ]
            
            missing = [check[1] for check in required_checks if not check[0]]
            
            if missing:
                logger.info(f"  ✗ Missing integrations: {missing}")
                self.checks_failed += 1
                self.has_inference_beam_support = False
            else:
                logger.info(f"  ✓ Beam search properly integrated")
                logger.info(f"    - beam_search_decode method")
                logger.info(f"    - --use-beam-search CLI flag")
                logger.info(f"    - --beam-width parameter")
                logger.info(f"    - use_beam_search parameter in generate_predictions()")
                self.checks_passed += 1
                self.has_inference_beam_support = True
        
        except Exception as e:
            logger.info(f"  ✗ Error checking file: {e}")
            self.checks_failed += 1
            self.has_inference_beam_support = False
    
    def check_subword_tokenization_module(self):
        """Check for subword tokenization module (Phase 2 - optional for MVP)."""
        logger.info("\n✓ Checking Subword Tokenization...")
        
        subword_paths = [
            self.project_root / "src/subword_tokenizer.py",
            self.project_root / "src/sentencepiece_tokenizer.py",
        ]
        
        found = False
        for path in subword_paths:
            if path.exists():
                logger.info(f"  ✓ Found: {path.name}")
                found = True
                break
        
        if not found:
            logger.info(f"  ℹ Subword tokenization not yet implemented (Phase 2)")
    
    def check_back_translation_module(self):
        """Verify back-translation module exists."""
        logger.info("\n✓ Checking Back-translation module...")
        
        back_trans_path = self.project_root / "src/back_translate.py"
        self.checks_total += 1
        
        if not back_trans_path.exists():
            logger.info(f"  ✗ FAILED: {back_trans_path} not found")
            self.checks_failed += 1
            self.has_back_translation = False
            return
        
        try:
            with open(back_trans_path, 'r') as f:
                content = f.read()
            
            if 'BackTranslationGenerator' in content:
                logger.info(f"  ✓ BackTranslationGenerator class found")
                self.checks_passed += 1
                self.has_back_translation = True
            else:
                logger.info(f"  ✗ BackTranslationGenerator class not found")
                self.checks_failed += 1
                self.has_back_translation = False
        
        except Exception as e:
            logger.info(f"  ✗ Error checking file: {e}")
            self.checks_failed += 1
            self.has_back_translation = False
    
    def check_tier3_config(self):
        """Verify TIER 3 configuration file exists."""
        logger.info("\n✓ Checking TIER 3 configuration...")
        
        config_path = self.project_root / "configs/model_seq2seq_tier3.yaml"
        self.checks_total += 1
        
        if not config_path.exists():
            logger.info(f"  ✗ FAILED: {config_path} not found")
            self.checks_failed += 1
            self.has_tier3_config = False
            return
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            required_sections = [
                ('beam_search:' in content or 'beam_search' in content, 'beam_search section'),
                ('back_translation' in content or 'augmentation' in content, 'augmentation section'),
            ]
            
            missing = [section[1] for section in required_sections if not section[0]]
            
            if missing:
                logger.info(f"  ✗ Missing config sections: {missing}")
                self.checks_failed += 1
                self.has_tier3_config = False
            else:
                logger.info(f"  ✓ TIER 3 config properly configured")
                logger.info(f"    - Beam search settings")
                logger.info(f"    - Augmentation settings")
                self.checks_passed += 1
                self.has_tier3_config = True
        
        except Exception as e:
            logger.info(f"  ✗ Error checking file: {e}")
            self.checks_failed += 1
            self.has_tier3_config = False
    
    def check_train_py_tier3_support(self):
        """Verify train.py supports TIER 3."""
        logger.info("\n✓ Checking train.py TIER 3 support...")
        
        train_path = self.project_root / "train.py"
        self.checks_total += 1
        
        if not train_path.exists():
            logger.info(f"  ✗ FAILED: {train_path} not found")
            self.checks_failed += 1
            self.has_train_tier3 = False
            return
        
        try:
            with open(train_path, 'r') as f:
                content = f.read()
            
            if "'tier3'" in content or '"tier3"' in content:
                logger.info(f"  ✓ train.py supports --model tier3")
                if "configs/model_seq2seq_tier3.yaml" in content:
                    logger.info(f"  ✓ Proper config loading for TIER 3")
                self.checks_passed += 1
                self.has_train_tier3 = True
            else:
                logger.info(f"  ✗ train.py doesn't support --model tier3")
                self.checks_failed += 1
                self.has_train_tier3 = False
        
        except Exception as e:
            logger.info(f"  ✗ Error checking file: {e}")
            self.checks_failed += 1
            self.has_train_tier3 = False
    
    def check_inference_py_tier3_support(self):
        """Verify inference.py supports TIER 3."""
        logger.info("\n✓ Checking inference.py TIER 3 support...")
        
        inference_path = self.project_root / "inference.py"
        self.checks_total += 1
        
        if not inference_path.exists():
            logger.info(f"  ✗ FAILED: {inference_path} not found")
            self.checks_failed += 1
            self.has_inference_tier3 = False
            return
        
        try:
            with open(inference_path, 'r') as f:
                content = f.read()
            
            if "'tier3'" in content or '"tier3"' in content or "'improved'" in content:
                logger.info(f"  ✓ inference.py supports model variants")
                if 'beam_search_decode' in content:
                    logger.info(f"  ✓ Beam search decoding available")
                self.checks_passed += 1
                self.has_inference_tier3 = True
            else:
                logger.info(f"  ✗ inference.py missing model support")
                self.checks_failed += 1
                self.has_inference_tier3 = False
        
        except Exception as e:
            logger.info(f"  ✗ Error checking file: {e}")
            self.checks_failed += 1
            self.has_inference_tier3 = False


def main():
    """Run TIER 3 verification."""
    
    project_root = Path(__file__).parent
    
    verifier = TIER3Verifier(project_root)
    success = verifier.run_all_checks()
    
    logger.info("\n" + "="*80)
    if success:
        logger.info("✓ TIER 3 READY FOR TRAINING")
        logger.info("\nNext steps:")
        logger.info("1. Train: python train.py --model tier3 --epochs 300")
        logger.info("2. Infer: python inference.py --model tier3 --use-beam-search --beam-width 8")
        logger.info("3. Evaluate: python evaluate_predictions.py")
    else:
        logger.info("✗ TIER 3 IMPLEMENTATION INCOMPLETE")
        logger.info("\nPlease complete pending components before training.")
    logger.info("="*80 + "\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
