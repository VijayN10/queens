# quick_fix_for_directory_name.py
"""
Quick fix to handle the directory name difference (queens_aaa_workflows vs queens_aaa_workflow)
"""
import sys
from pathlib import Path

def fix_import_paths():
    """Add both possible directory paths to handle naming variations."""
    
    # Get repo root - works from either queens_aaa_workflow or queens_aaa_workflows
    current_file = Path(__file__).absolute()
    
    # Try to find the queens repo root
    repo_root = None
    for parent in current_file.parents:
        if (parent / 'src' / 'queens').exists():
            repo_root = parent
            break
    
    if repo_root is None:
        raise RuntimeError("Could not find queens repository root!")
    
    print(f"✅ Found repo root: {repo_root}")
    
    # Add necessary paths
    queens_path = str(repo_root / 'src')
    aorta_path = str(repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4')
    
    if queens_path not in sys.path:
        sys.path.insert(0, queens_path)
    if aorta_path not in sys.path:
        sys.path.insert(0, aorta_path)
    
    print(f"✅ Added QUEENS path: {queens_path}")
    print(f"✅ Added AORTA path: {aorta_path}")
    
    return repo_root

if __name__ == "__main__":
    repo_root = fix_import_paths()
    
    # Test imports
    try:
        from queens.global_settings import GlobalSettings
        print("✅ QUEENS imports working")
        
        from config import ConfigParams
        print("✅ AORTA imports working")
        
        print("🎉 All imports successful!")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        sys.exit(1)