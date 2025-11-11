#!/usr/bin/env python3
"""
Competition Information and Dataset Collector
Fetches competition descriptions, dataset info, and downloads datasets for all competitions.
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import argparse


class CompetitionInfoCollector:
    """Collects competition metadata and downloads datasets."""
    
    def __init__(self, notebooks_dir: str = "notebooks"):
        self.notebooks_dir = Path(notebooks_dir)
        self.competitions = self._get_competition_list()
        # Use kaggle from venv
        self.kaggle_path = Path(__file__).parent / 'venv' / 'bin' / 'kaggle'
        if not self.kaggle_path.exists():
            self.kaggle_path = 'kaggle'  # fallback to PATH
        
    def _get_competition_list(self) -> List[str]:
        """Get list of all competition folders."""
        competitions = []
        if self.notebooks_dir.exists():
            competitions = [d.name for d in self.notebooks_dir.iterdir() if d.is_dir()]
        return sorted(competitions)
    
    def get_competition_info(self, competition: str) -> Dict[str, Any]:
        """Fetch competition information using Kaggle API."""
        print(f"\n📊 Fetching info for: {competition}")
        
        try:
            # Get competition details
            result = subprocess.run(
                [str(self.kaggle_path), 'competitions', 'list', '-s', competition],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get detailed competition info (if available)
            files_result = subprocess.run(
                [str(self.kaggle_path), 'competitions', 'files', competition],
                capture_output=True,
                text=True
            )
            
            info = {
                'competition_name': competition,
                'kaggle_url': f'https://www.kaggle.com/competitions/{competition}',
                'api_output': result.stdout,
                'dataset_files': files_result.stdout if files_result.returncode == 0 else 'N/A',
                'collected_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return info
            
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Error fetching info: {e}")
            return {
                'competition_name': competition,
                'kaggle_url': f'https://www.kaggle.com/competitions/{competition}',
                'error': str(e),
                'collected_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def download_competition_dataset(self, competition: str, output_dir: Path) -> bool:
        """Download competition dataset."""
        print(f"  📥 Downloading dataset...")
        
        # Create datasets subdirectory
        dataset_dir = output_dir / 'dataset'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download competition files
            result = subprocess.run(
                [str(self.kaggle_path), 'competitions', 'download', '-c', competition, '-p', str(dataset_dir)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"  ✅ Dataset downloaded successfully")
                
                # Check if we got zip files and extract them
                zip_files = list(dataset_dir.glob('*.zip'))
                if zip_files:
                    print(f"  📦 Found {len(zip_files)} zip file(s), extracting...")
                    for zip_file in zip_files:
                        subprocess.run(['unzip', '-q', '-o', str(zip_file), '-d', str(dataset_dir)])
                        print(f"    ✓ Extracted {zip_file.name}")
                
                return True
            else:
                print(f"  ⚠️  Download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Download timeout (>5 min)")
            return False
        except Exception as e:
            print(f"  ⚠️  Download error: {e}")
            return False
    
    def create_competition_readme(self, competition: str, info: Dict[str, Any], output_dir: Path):
        """Create a README.md file for the competition."""
        readme_path = output_dir / 'COMPETITION_INFO.md'
        
        readme_content = f"""# {competition.replace('-', ' ').title()}

## Competition Information

**Competition Name:** {competition}  
**Kaggle URL:** {info['kaggle_url']}  
**Data Collected:** {info['collected_at']}

## Description

This competition focuses on machine learning and data science challenges. The notebooks in this folder contain various approaches and solutions submitted by Kaggle community members.

## Dataset Information

### Files Available

```
{info.get('dataset_files', 'Dataset information not available')}
```

### Dataset Location

The competition dataset is available in the `dataset/` subdirectory of this folder.

## Notebooks

This folder contains {len(list(output_dir.glob('*/*.ipynb')))} downloaded notebooks from this competition.

### How to Use

1. **View Notebooks**: Browse the subdirectories to find individual notebooks
2. **Access Dataset**: Check the `dataset/` folder for competition data files
3. **Run Notebooks**: Install required dependencies and run notebooks in your environment

## API Information

```
{info.get('api_output', 'N/A')}
```

## Attribution

All notebooks and datasets are from Kaggle's public repository.  
Please attribute original authors when using their work.

---
*Generated by Kaggle Notebook Extractor*  
*Repository: https://github.com/Surfing-Ninja/Kaggle_NB_EXTRACTOR*
"""
        
        readme_path.write_text(readme_content)
        print(f"  📝 Created COMPETITION_INFO.md")
    
    def save_competition_metadata(self, competition: str, info: Dict[str, Any], output_dir: Path):
        """Save competition metadata as JSON."""
        metadata_path = output_dir / 'competition_metadata.json'
        
        # Add notebook count
        notebooks = list(output_dir.glob('*/*.ipynb'))
        info['notebook_count'] = len(notebooks)
        info['notebook_authors'] = list(set([nb.parent.name.split('__')[0] for nb in notebooks]))
        
        with open(metadata_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"  💾 Saved metadata JSON")
    
    def process_competition(self, competition: str, download_dataset: bool = True):
        """Process a single competition: fetch info, create README, download dataset."""
        print(f"\n{'='*70}")
        print(f"Processing: {competition}")
        print(f"{'='*70}")
        
        competition_dir = self.notebooks_dir / competition
        
        if not competition_dir.exists():
            print(f"  ⚠️  Competition folder not found")
            return
        
        # Get competition info
        info = self.get_competition_info(competition)
        
        # Create README
        self.create_competition_readme(competition, info, competition_dir)
        
        # Save metadata
        self.save_competition_metadata(competition, info, competition_dir)
        
        # Download dataset if requested
        if download_dataset:
            dataset_downloaded = self.download_competition_dataset(competition, competition_dir)
            info['dataset_downloaded'] = dataset_downloaded
            # Update metadata with download status
            self.save_competition_metadata(competition, info, competition_dir)
        
        print(f"  ✅ Completed processing {competition}")
        
        # Rate limiting
        time.sleep(2)
    
    def process_all_competitions(self, download_datasets: bool = True):
        """Process all competitions."""
        print(f"\n🚀 Starting to process {len(self.competitions)} competitions")
        print(f"Download datasets: {'Yes' if download_datasets else 'No'}")
        
        summary = {
            'total_competitions': len(self.competitions),
            'processed': 0,
            'failed': 0,
            'datasets_downloaded': 0
        }
        
        for idx, competition in enumerate(self.competitions, 1):
            print(f"\n[{idx}/{len(self.competitions)}]")
            try:
                self.process_competition(competition, download_dataset=download_datasets)
                summary['processed'] += 1
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                summary['failed'] += 1
        
        # Generate overall summary
        self._generate_summary(summary)
        
        print(f"\n{'='*70}")
        print(f"✅ COMPLETED!")
        print(f"{'='*70}")
        print(f"Processed: {summary['processed']}/{summary['total_competitions']}")
        print(f"Failed: {summary['failed']}")
    
    def _generate_summary(self, summary: Dict):
        """Generate a summary file."""
        summary_path = Path('competition_collection_summary.json')
        summary['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        summary['competitions'] = self.competitions
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Collect competition info and datasets')
    parser.add_argument('--notebooks-dir', default='notebooks', help='Notebooks directory')
    parser.add_argument('--no-download', action='store_true', help='Skip dataset downloads')
    parser.add_argument('--competition', help='Process specific competition only')
    
    args = parser.parse_args()
    
    collector = CompetitionInfoCollector(notebooks_dir=args.notebooks_dir)
    
    if args.competition:
        # Process single competition
        collector.process_competition(args.competition, download_dataset=not args.no_download)
    else:
        # Process all competitions
        collector.process_all_competitions(download_datasets=not args.no_download)


if __name__ == '__main__':
    main()
