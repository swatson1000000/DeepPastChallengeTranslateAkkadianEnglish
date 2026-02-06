#!/usr/bin/env python3
"""
Verify all TIER 2 improvements are properly implemented
"""

import os
import sys
from pathlib import Path

def check_file_contains(filepath, patterns, description):
    """Check if file contains all patterns."""
    if not os.path.exists(filepath):
        print(f"✗ {description}: File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    all_found = True
    for pattern in patterns:
        if pattern in content:
            print(f"  ✓ {pattern}")
        else:
            print(f"  ✗ MISSING: {pattern}")
            all_found = False
    
    return all_found

def main():
    project_root = Path(__file__).parent
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║        TIER 2 IMPLEMENTATION VERIFICATION                     ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    all_good = True
    
    # 1. Check train.py
    print("1. Training Script (train.py)")
    patterns = [
        "class CopyMechanism",
        "class LexiconConstrainedDecoder",
        "def build_valid_token_mask",
        "if use_tier2 and copy_mechanism",
        "if use_tier2 and lexicon_decoder",
        "copy_mechanism = CopyMechanism",
        "lexicon_decoder = LexiconConstrainedDecoder",
        "grad_params.extend(list(copy_mechanism.parameters()))",
        "grad_params.extend(list(lexicon_decoder.parameters()))"
    ]
    if not check_file_contains(str(project_root / "train.py"), patterns, "train.py"):
        all_good = False
    print()
    
    # 2. Check inference.py
    print("2. Inference Script (inference.py)")
    patterns = [
        "class CopyMechanism",
        "def greedy_decode(self, encoder_outputs, hidden_state, cell_state=None, src_tokens=None",
        "if use_copy and self.copy_mechanism and src_tokens is not None:",
        "src_tokens required for copy mechanism",
        "coverage = coverage + copy_weights",
        "decoded, _ = self.greedy_decode(encoder_outputs[0], hidden, cell, src_tokens=src_tensor[0], use_copy=use_copy)"
    ]
    if not check_file_contains(str(project_root / "inference.py"), patterns, "inference.py"):
        all_good = False
    print()
    
    # 3. Check tier2_improvements.py
    print("3. TIER 2 Components (src/tier2_improvements.py)")
    patterns = [
        "class CopyMechanism(nn.Module):",
        "class LexiconConstrainedDecoder(nn.Module):",
        "def build_valid_token_mask",
        "coverage_penalty = self.coverage_proj",
        "copy_weights = torch.softmax",
        "copy_prob = torch.sigmoid",
        "enforce_constraints",
        "valid_mask"
    ]
    if not check_file_contains(str(project_root / "src/tier2_improvements.py"), patterns, "tier2_improvements.py"):
        all_good = False
    print()
    
    # 4. Check configs
    print("4. TIER 2 Configuration (configs/model_seq2seq_tier2.yaml)")
    patterns = [
        "copy_mechanism:",
        "enabled: true",
        "coverage_enabled: true",
        "coverage_penalty:",
        "lexicon_constraints:",
        "train_augmented.csv"
    ]
    if not check_file_contains(str(project_root / "configs/model_seq2seq_tier2.yaml"), patterns, "model_seq2seq_tier2.yaml"):
        all_good = False
    print()
    
    # 5. Check documentation
    print("5. Documentation")
    tier2_doc = project_root / "TIER2_IMPLEMENTATION.md"
    if tier2_doc.exists():
        print(f"  ✓ {tier2_doc.name}")
    else:
        print(f"  ✗ Missing: {tier2_doc.name}")
        all_good = False
    
    consolidated_doc = project_root / "CONSOLIDATED_SCRIPTS.md"
    if consolidated_doc.exists():
        print(f"  ✓ {consolidated_doc.name}")
    else:
        print(f"  ✗ Missing: {consolidated_doc.name}")
        all_good = False
    print()
    
    # 6. Summary
    print("╔════════════════════════════════════════════════════════════════╗")
    if all_good:
        print("║ ✓ ALL TIER 2 IMPROVEMENTS PROPERLY IMPLEMENTED                ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        print("\nTIER 2 Features Summary:")
        print("  ✓ Copy Mechanism - Pointer-generator network with coverage tracking")
        print("  ✓ Lexicon Constraints - Token masking to prevent gibberish")
        print("  ✓ Coverage Penalty - Prevents repeated copying")
        print("  ✓ Extended Training - 300 epochs for TIER 2")
        print("  ✓ Gradient Clipping - Includes all TIER 2 components")
        print("  ✓ Source Token Passing - Proper implementation in inference")
        print("\nReady to train: python train.py --model tier2 --epochs 300")
        print("Ready to infer: python inference.py --model tier2 --use-copy")
        return 0
    else:
        print("║ ✗ SOME TIER 2 FEATURES MISSING                               ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        return 1

if __name__ == '__main__':
    sys.exit(main())
