#!/usr/bin/env python3
"""
Data Cleanup Utility

Safely delete data directories with double confirmation.
This script provides a way to clean up all downloaded and generated data.

SAFETY FEATURES:
- Preview mode (dry-run)
- First confirmation prompt
- Second confirmation (must type exact phrase)
- Backup reminder
- Selective deletion options

Usage:
    python scripts/clear_all_data.py --dry-run  # Preview only
    python scripts/clear_all_data.py --all      # Delete everything (requires confirmation)
    python scripts/clear_all_data.py --fundamentals-only
    python scripts/clear_all_data.py --prices-only
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes"""
    if not path.exists():
        return 0
    
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def count_files(path: Path) -> int:
    """Count total files in directory"""
    if not path.exists():
        return 0
    
    try:
        return sum(1 for _ in path.rglob('*') if _.is_file())
    except Exception:
        return 0


class DataCleaner:
    """Manage data cleanup operations"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_root = project_root / "data"
        
        # Define data categories
        self.categories = {
            'fundamentals': {
                'path': self.data_root / 'fundamentals',
                'description': 'Fundamental Data',
            },
            'prices': {
                'path': self.data_root / 'parquet' / '1d',
                'description': 'Price Data (OHLCV)',
            },
            'simfin_cache': {
                'path': self.data_root / 'simfin_data',
                'description': 'SimFin Cache',
            },
            'features': {
                'path': self.data_root / 'features',
                'description': 'Generated Features',
            },
            'results': {
                'path': self.data_root / 'results',
                'description': 'Backtest Results',
            },
            'models': {
                'path': self.data_root / 'models',
                'description': 'Trained Models',
            },
            'checkpoints': {
                'path': self.data_root,
                'description': 'Progress Checkpoints',
                'files': ['.ohlcv_progress.json']
            }
        }
    
    def get_category_info(self, category: str) -> Tuple[int, int]:
        """Get file count and total size for a category"""
        info = self.categories[category]
        path = info['path']
        
        if 'files' in info:
            # Special case: specific files
            file_count = 0
            total_size = 0
            for filename in info['files']:
                file_path = path / filename
                if file_path.exists():
                    file_count += 1
                    total_size += file_path.stat().st_size
            return file_count, total_size
        else:
            # Regular directory
            return count_files(path), get_directory_size(path)
    
    def preview_deletion(self, categories: List[str]) -> dict:
        """Preview what would be deleted"""
        preview = {}
        total_files = 0
        total_size = 0
        
        for category in categories:
            if category not in self.categories:
                continue
                
            file_count, size = self.get_category_info(category)
            preview[category] = {
                'description': self.categories[category]['description'],
                'path': str(self.categories[category]['path']),
                'files': file_count,
                'size': size,
                'size_formatted': format_size(size)
            }
            total_files += file_count
            total_size += size
        
        preview['total'] = {
            'files': total_files,
            'size': total_size,
            'size_formatted': format_size(total_size)
        }
        
        return preview
    
    def delete_category(self, category: str) -> bool:
        """Delete a data category"""
        info = self.categories[category]
        path = info['path']
        
        try:
            if 'files' in info:
                # Delete specific files
                deleted = False
                for filename in info['files']:
                    file_path = path / filename
                    if file_path.exists():
                        file_path.unlink()
                        deleted = True
                return deleted
            else:
                # Delete entire directory
                if path.exists():
                    shutil.rmtree(path)
                    return True
            return False
        except Exception as e:
            print(f"  ✗ Error deleting {category}: {e}", file=sys.stderr)
            return False


def print_preview(preview: dict):
    """Print deletion preview"""
    print("\n" + "=" * 70)
    print("Data Cleanup Preview")
    print("=" * 70)
    print("\nWARNING: The following data will be deleted:\n")
    
    for category, info in preview.items():
        if category == 'total':
            continue
        
        if info['files'] > 0:
            print(f"📁 {info['description']} ({info['path']})")
            print(f"   - Files: {info['files']}")
            print(f"   - Size: {info['size_formatted']}\n")
    
    print("=" * 70)
    print(f"TOTAL: {preview['total']['files']} files, {preview['total']['size_formatted']} will be deleted")
    print("=" * 70)


def get_confirmation() -> bool:
    """Get user confirmation with two-step process"""
    # First confirmation
    print("\n⚠️  RECOMMENDATION: Backup your data first!")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"   Suggested command: tar -czf data_backup_{timestamp}.tar.gz data/\n")
    
    response = input("First Confirmation:\nAre you ABSOLUTELY sure you want to delete this data? [yes/no]: ").strip().lower()
    
    if response != 'yes':
        print("\n✓ Deletion cancelled")
        return False
    
    # Second confirmation
    print("\nSecond Confirmation:")
    response = input("Type 'DELETE ALL DATA' to proceed (case-sensitive): ").strip()
    
    if response != 'DELETE ALL DATA':
        print("\n✓ Deletion cancelled (incorrect confirmation phrase)")
        return False
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Data cleanup utility with safety confirmations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be deleted (safe, no actual deletion)
  python scripts/clear_all_data.py --dry-run
  
  # Delete everything (requires double confirmation)
  python scripts/clear_all_data.py --all
  
  # Delete only fundamentals
  python scripts/clear_all_data.py --fundamentals-only
  
  # Delete only prices
  python scripts/clear_all_data.py --prices-only
  
  # Delete SimFin cache only
  python scripts/clear_all_data.py --simfin-cache-only
  
  # Delete checkpoints only
  python scripts/clear_all_data.py --checkpoints-only
  
  # Multiple categories
  python scripts/clear_all_data.py --fundamentals-only --prices-only

SAFETY:
  - Preview mode with --dry-run (no deletion)
  - Two-step confirmation required
  - Backup reminder before deletion
        """
    )
    
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview only, do not delete')
    parser.add_argument('--all', action='store_true', 
                       help='Delete all data categories')
    parser.add_argument('--fundamentals-only', action='store_true', 
                       help='Delete only fundamental data')
    parser.add_argument('--prices-only', action='store_true', 
                       help='Delete only price data')
    parser.add_argument('--simfin-cache-only', action='store_true', 
                       help='Delete only SimFin cache')
    parser.add_argument('--features-only', action='store_true', 
                       help='Delete only generated features')
    parser.add_argument('--results-only', action='store_true', 
                       help='Delete only backtest results')
    parser.add_argument('--models-only', action='store_true', 
                       help='Delete only trained models')
    parser.add_argument('--checkpoints-only', action='store_true', 
                       help='Delete only progress checkpoints')
    
    args = parser.parse_args()
    
    # Determine which categories to delete
    categories_to_delete = []
    
    if args.all:
        categories_to_delete = list(DataCleaner(project_root).categories.keys())
    else:
        if args.fundamentals_only:
            categories_to_delete.append('fundamentals')
        if args.prices_only:
            categories_to_delete.append('prices')
        if args.simfin_cache_only:
            categories_to_delete.append('simfin_cache')
        if args.features_only:
            categories_to_delete.append('features')
        if args.results_only:
            categories_to_delete.append('results')
        if args.models_only:
            categories_to_delete.append('models')
        if args.checkpoints_only:
            categories_to_delete.append('checkpoints')
    
    if not categories_to_delete:
        print("Error: No deletion options specified")
        print("Use --help to see available options")
        sys.exit(1)
    
    # Initialize cleaner
    cleaner = DataCleaner(project_root)
    
    # Preview deletion
    preview = cleaner.preview_deletion(categories_to_delete)
    
    if preview['total']['files'] == 0:
        print("\nNo data found to delete (directories may already be empty)")
        sys.exit(0)
    
    print_preview(preview)
    
    # Dry run mode - stop here
    if args.dry_run:
        print("\n✓ Dry-run complete (no data was deleted)")
        sys.exit(0)
    
    # Get confirmation
    if not get_confirmation():
        sys.exit(0)
    
    # Perform deletion
    print("\n✓ Deleting data...\n")
    
    deleted_count = 0
    deleted_size = 0
    
    for category in categories_to_delete:
        info = preview.get(category)
        if info and info['files'] > 0:
            print(f"  - Removing {info['description']}...", end=' ')
            if cleaner.delete_category(category):
                print(f"✓ ({info['size_formatted']})")
                deleted_count += info['files']
                deleted_size += info['size']
            else:
                print("✗ (already empty or error)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✓ Cleanup Complete!")
    print("=" * 70)
    print(f"  Files deleted: {deleted_count}")
    print(f"  Space freed: {format_size(deleted_size)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
