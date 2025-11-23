"""
Quick test script to verify the pipeline works correctly
Run with: python -m graphssl.test_pipeline
"""
import sys
import torch
from pathlib import Path

from graphssl.utils.data_utils import load_ogb_mag, create_neighbor_loaders, get_dataset_info
from graphssl.utils.models import create_model, HeteroGraphSAGE
from graphssl.utils.training_utils import train_epoch, evaluate, train_model

print("Testing GraphSSL Pipeline Components")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
print("   ✓ All imports successful")

# Test PyTorch Geometric
print("\n2. Testing PyTorch Geometric...")
try:
    from torch_geometric.datasets import OGB_MAG
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import SAGEConv, to_hetero
    print("   ✓ PyTorch Geometric components available")
except Exception as e:
    print(f"   ✗ PyTorch Geometric import failed: {e}")
    sys.exit(1)

# Test CUDA availability
print("\n3. Checking CUDA availability...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
else:
    print("   ⚠ CUDA not available, will use CPU")

# Test dataset loading (optional, commented out to avoid downloading)
print("\n4. Dataset loading test...")
# print("   Note: Skipping actual dataset download in test mode")
# print("   To test dataset loading, uncomment the code in test_pipeline.py")
# Uncomment below to test actual dataset loading:
try:
    assert Path.cwd().name == 'GraphSSL'
    data_path = Path("data")
    data_path.mkdir(parents=True, exist_ok=True)
    data = load_ogb_mag(str(data_path), preprocess="metapath2vec")
    print("   ✓ Dataset loaded successfully")
except Exception as e:
    print(f"   ✗ Dataset loading failed: {e}")

print("\n" + "=" * 60)
print("Pipeline component test completed!")
print("=" * 60)
print("\nTo run the full pipeline:")
print("  python -m graphssl.main")
print("\nFor help:")
print("  python -m graphssl.main --help")
