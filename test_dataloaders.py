"""
Test script for F1 dataloaders.
Tests both StintDataloader and AutoregressiveLapDataloader.
"""

import sys
from pathlib import Path

# Add project and code directories to path
project_root = Path(__file__).parent
code_root = project_root / "code"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(code_root))

from dataloaders import StintDataloader, AutoregressiveLapDataloader, LapTimeNormalizer
import torch
from torch.utils.data import DataLoader


def test_stint_dataloader_single_year():
    """Test StintDataloader with a single year."""
    print("\n" + "="*60)
    print("TEST 1: StintDataloader - Single Year (2019)")
    print("="*60)
    
    try:
        ds = StintDataloader(year=2019, window_size=20, augment_prob=0.0)
        print(f"✓ Successfully loaded StintDataloader")
        print(f"  - Total stints: {len(ds)}")
        
        if len(ds) > 0:
            features, target, metadata = ds[0]
            print(f"  - First sample shapes:")
            print(f"    - Features: {features.shape}")
            print(f"    - Target: {target.shape}")
            print(f"  - Metadata: {metadata}")
            print(f"  - Target lap time: {target.item():.2f} ms")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stint_dataloader_multi_year():
    """Test StintDataloader with multiple years."""
    print("\n" + "="*60)
    print("TEST 2: StintDataloader - Multiple Years (2019-2021)")
    print("="*60)
    
    try:
        ds = StintDataloader(year=[2019, 2020, 2021], window_size=15, augment_prob=0.0)
        print(f"✓ Successfully loaded StintDataloader with multiple years")
        print(f"  - Total stints: {len(ds)}")
        
        if len(ds) > 10:
            features, target, metadata = ds[10]
            print(f"  - Sample 10 (checking year={metadata['year']}):")
            print(f"    - Features shape: {features.shape}")
            print(f"    - Target: {target.item():.2f} ms")
            print(f"    - Driver: {metadata['driver']}, Year: {metadata['year']}, Circuit: {metadata['circuit']}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autoregressive_dataloader_single_year():
    """Test AutoregressiveLapDataloader with a single year."""
    print("\n" + "="*60)
    print("TEST 3: AutoregressiveLapDataloader - Single Year (2019)")
    print("="*60)
    
    try:
        ds = AutoregressiveLapDataloader(year=2019, context_window=5, augment_prob=0.0)
        print(f"✓ Successfully loaded AutoregressiveLapDataloader")
        print(f"  - Total lap pairs: {len(ds)}")
        
        if len(ds) > 0:
            context, target, metadata = ds[0]
            print(f"  - First sample shapes:")
            print(f"    - Context: {context.shape}")
            print(f"    - Target: {target.shape}")
            print(f"  - Metadata: {metadata}")
            print(f"  - Target lap time: {target.item():.2f} ms")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autoregressive_dataloader_multi_year():
    """Test AutoregressiveLapDataloader with multiple years."""
    print("\n" + "="*60)
    print("TEST 4: AutoregressiveLapDataloader - Multiple Years (2020-2021)")
    print("="*60)
    
    try:
        ds = AutoregressiveLapDataloader(
            year=[2020, 2021],
            context_window=8,
            augment_prob=0.0
        )
        print(f"✓ Successfully loaded AutoregressiveLapDataloader with multiple years")
        print(f"  - Total lap pairs: {len(ds)}")
        
        if len(ds) > 20:
            context, target, metadata = ds[20]
            print(f"  - Sample 20 (checking year={metadata['year']}):")
            print(f"    - Context shape: {context.shape}")
            print(f"    - Target: {target.item():.2f} ms")
            print(f"    - Race: {metadata['race_name']}, Context length: {metadata['context_length']}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_batching():
    """Test batching with PyTorch DataLoader."""
    print("\n" + "="*60)
    print("TEST 5: PyTorch DataLoader Batching")
    print("="*60)
    
    try:
        ds = AutoregressiveLapDataloader(year=2019, context_window=5)
        
        if len(ds) < 10:
            print(f"⚠ Skipping batching test (only {len(ds)} samples available)")
            return True
        
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        
        batch_count = 0
        for context_batch, target_batch, metadata_batch in loader:
            batch_count += 1
            print(f"  - Batch {batch_count}:")
            print(f"    - Context batch shape: {context_batch.shape}")
            print(f"    - Target batch shape: {target_batch.shape}")
            print(f"    - Batch has {len(metadata_batch['driver'])} samples")
            
            if batch_count >= 2:
                break
        
        print(f"✓ Successfully batched {batch_count} batches")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_normalization_statistics():
    """Test normalization and statistics."""
    print("\n" + "="*60)
    print("TEST 6: Normalization Statistics")
    print("="*60)
    
    try:
        ds = StintDataloader(year=2019, window_size=20)
        
        stats = ds.normalizer.get_statistics()
        print(f"✓ Successfully retrieved normalization statistics")
        print(f"  - Scaler type: StandardScaler")
        print(f"  - Normalized columns: {len(stats['columns'])}")
        print(f"  - First few column statistics:")
        for i, col in enumerate(stats['columns'][:3]):
            print(f"    - {col}: mean={stats['mean'][i]:.2f}, std={stats['std'][i]:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmentation():
    """Test data augmentation."""
    print("\n" + "="*60)
    print("TEST 7: Data Augmentation")
    print("="*60)
    
    try:
        ds_no_aug = StintDataloader(year=2019, window_size=20, augment_prob=0.0)
        ds_with_aug = StintDataloader(year=2019, window_size=20, augment_prob=1.0)
        
        print(f"✓ Successfully loaded datasets with and without augmentation")
        
        if len(ds_no_aug) > 0 and len(ds_with_aug) > 0:
            features_no_aug, target_no_aug, _ = ds_no_aug[0]
            
            # Get same sample with augmentation (multiple times to see variation)
            targets_aug = []
            for _ in range(3):
                _, target_aug, _ = ds_with_aug[0]
                targets_aug.append(target_aug.item())
            
            print(f"  - Sample 0 without augmentation: {target_no_aug.item():.2f} ms")
            print(f"  - Sample 0 with augmentation (3 variations):")
            for i, t in enumerate(targets_aug, 1):
                print(f"    - Variation {i}: {t:.2f} ms (Δ={t - target_no_aug.item():.2f} ms)")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("F1 DATA LOADERS - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Single Year Stint", test_stint_dataloader_single_year),
        ("Multi-Year Stint", test_stint_dataloader_multi_year),
        ("Single Year Autoregressive", test_autoregressive_dataloader_single_year),
        ("Multi-Year Autoregressive", test_autoregressive_dataloader_multi_year),
        ("DataLoader Batching", test_dataloader_batching),
        ("Normalization Stats", test_normalization_statistics),
        ("Data Augmentation", test_augmentation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
