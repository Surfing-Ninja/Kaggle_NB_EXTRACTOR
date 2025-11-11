#!/usr/bin/env python3
"""
Competition Information and Dataset Collector
Fetches competition descriptions, dataset info, and downloads datasets for all competitions.
"""

import os
import json
import subprocess
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
try:
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    print("Warning: Install requests, beautifulsoup4, lxml for web scraping: pip install requests beautifulsoup4 lxml pandas")


class CompetitionInfoCollector:
    """Collects competition metadata and downloads datasets."""
    
    def __init__(self, notebooks_dir: str = "notebooks"):
        self.notebooks_dir = Path(notebooks_dir)
        self.competitions = self._get_competition_list()
        # Use kaggle from venv
        self.kaggle_path = Path(__file__).parent / 'venv' / 'bin' / 'kaggle'
        if not self.kaggle_path.exists():
            self.kaggle_path = 'kaggle'  # fallback to PATH
        self.session = requests.Session() if SCRAPING_AVAILABLE else None
        
    def _get_competition_list(self) -> List[str]:
        """Get list of all competition folders."""
        competitions = []
        if self.notebooks_dir.exists():
            competitions = [d.name for d in self.notebooks_dir.iterdir() if d.is_dir()]
        return sorted(competitions)
    
    def scrape_competition_data_page(self, competition: str) -> Dict[str, Any]:
        """Scrape competition data page for detailed dataset information."""
        if not SCRAPING_AVAILABLE:
            return {'error': 'Scraping libraries not available'}
        
        url = f'https://www.kaggle.com/competitions/{competition}/data'
        print(f"  🌐 Scraping data page: {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return {'error': f'HTTP {response.status_code}'}
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            data_info = {
                'files': [],
                'description': '',
                'columns_info': []
            }
            
            # Try to extract dataset description
            desc_sections = soup.find_all('div', class_='markdown-converter__text')
            if desc_sections:
                data_info['description'] = desc_sections[0].get_text(strip=True)[:500]  # First 500 chars
            
            # Look for column information in various formats
            # Method 1: Look for tables
            tables = soup.find_all('table')
            for table in tables:
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                if any(keyword in ' '.join(headers).lower() for keyword in ['column', 'field', 'feature', 'variable']):
                    rows = table.find_all('tr')[1:]  # Skip header
                    for row in rows[:20]:  # Limit to first 20 rows
                        cols = [td.get_text(strip=True) for td in row.find_all('td')]
                        if len(cols) >= 2:
                            data_info['columns_info'].append({
                                'name': cols[0],
                                'description': cols[1] if len(cols) > 1 else ''
                            })
            
            # Method 2: Look for file information
            file_sections = soup.find_all('div', class_='file-container')
            for file_section in file_sections:
                file_name = file_section.find('a')
                if file_name:
                    data_info['files'].append(file_name.get_text(strip=True))
            
            print(f"  ✓ Found {len(data_info['columns_info'])} column definitions")
            return data_info
            
        except Exception as e:
            print(f"  ⚠️  Scraping error: {e}")
            return {'error': str(e)}
    
    def get_csv_columns_from_dataset(self, dataset_dir: Path) -> List[Dict[str, Any]]:
        """Extract column information from downloaded CSV files."""
        if not SCRAPING_AVAILABLE:
            return []
        
        columns_info = []
        
        # Look for CSV files
        csv_files = list(dataset_dir.glob('*.csv'))
        if not csv_files:
            # Try compressed files
            csv_files = list(dataset_dir.glob('*.csv.gz'))
        
        for csv_file in csv_files[:1]:  # Just check the first CSV (usually train.csv)
            try:
                print(f"  📊 Analyzing {csv_file.name}...")
                df = pd.read_csv(csv_file, nrows=5)  # Read just first 5 rows for efficiency
                
                for col in df.columns:
                    col_info = {
                        'name': col,
                        'dtype': str(df[col].dtype),
                        'sample_values': df[col].dropna().head(3).tolist(),
                        'null_count': int(df[col].isnull().sum()),
                        'unique_count': int(df[col].nunique()) if df[col].nunique() < 100 else '>100'
                    }
                    columns_info.append(col_info)
                
                print(f"  ✓ Extracted {len(columns_info)} columns from {csv_file.name}")
                break
                
            except Exception as e:
                print(f"  ⚠️  Error reading {csv_file.name}: {e}")
        
        return columns_info
    
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
        """Download competition dataset with multiple retry strategies."""
        print(f"  📥 Downloading dataset...")
        
        # Create datasets subdirectory
        dataset_dir = output_dir / 'dataset'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Try method 1: Standard competition download
        try:
            result = subprocess.run(
                [str(self.kaggle_path), 'competitions', 'download', '-c', competition, '-p', str(dataset_dir)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print(f"  ✅ Dataset downloaded successfully")
                self._extract_zip_files(dataset_dir)
                return True
            else:
                error_msg = result.stderr.lower()
                print(f"  ⚠️  Method 1 failed: {result.stderr[:100]}")
                
                # Check if it's a permission/rules issue
                if 'must accept' in error_msg or 'rules' in error_msg:
                    print(f"  ℹ️  Competition requires accepting rules first")
                    return False
                    
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Download timeout (>10 min)")
        except Exception as e:
            print(f"  ⚠️  Method 1 error: {e}")
        
        # Try method 2: Download as dataset (some competitions expose data as datasets)
        try:
            print(f"  🔄 Trying alternative method...")
            result = subprocess.run(
                [str(self.kaggle_path), 'datasets', 'download', '-d', f'kaggle/{competition}', '-p', str(dataset_dir)],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                print(f"  ✅ Dataset downloaded via datasets API")
                self._extract_zip_files(dataset_dir)
                return True
                
        except Exception as e:
            print(f"  ⚠️  Method 2 error: {e}")
        
        # Try method 3: Download individual files
        try:
            print(f"  � Trying individual file download...")
            # Get file list first
            files_result = subprocess.run(
                [str(self.kaggle_path), 'competitions', 'files', competition],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if files_result.returncode == 0 and files_result.stdout:
                # Parse file names (skip header lines)
                lines = files_result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines[:5]:  # Try first 5 files
                    parts = line.split()
                    if parts:
                        filename = parts[0]
                        try:
                            subprocess.run(
                                [str(self.kaggle_path), 'competitions', 'download', '-c', competition, '-f', filename, '-p', str(dataset_dir)],
                                capture_output=True,
                                timeout=120,
                                check=True
                            )
                            print(f"    ✓ Downloaded {filename}")
                        except:
                            pass
                
                # Check if we got any files
                if list(dataset_dir.glob('*')):
                    self._extract_zip_files(dataset_dir)
                    return True
                    
        except Exception as e:
            print(f"  ⚠️  Method 3 error: {e}")
        
        return False
    
    def _extract_zip_files(self, dataset_dir: Path):
        """Extract any zip files in the dataset directory."""
        zip_files = list(dataset_dir.glob('*.zip'))
        if zip_files:
            print(f"  📦 Found {len(zip_files)} zip file(s), extracting...")
            for zip_file in zip_files:
                try:
                    subprocess.run(['unzip', '-q', '-o', str(zip_file), '-d', str(dataset_dir)], 
                                 capture_output=True, timeout=120)
                    print(f"    ✓ Extracted {zip_file.name}")
                except Exception as e:
                    print(f"    ⚠️  Failed to extract {zip_file.name}: {e}")
    
    def create_competition_readme(self, competition: str, info: Dict[str, Any], output_dir: Path, 
                                  scraped_data: Optional[Dict] = None, columns_info: Optional[List] = None):
        """Create a detailed README.md file for the competition."""
        readme_path = output_dir / 'COMPETITION_INFO.md'
        
        # Build column information section
        columns_section = ""
        if columns_info:
            columns_section = "\n### Dataset Columns\n\n"
            if isinstance(columns_info[0], dict) and 'dtype' in columns_info[0]:
                # From CSV analysis
                columns_section += "| Column Name | Data Type | Sample Values | Unique Values |\n"
                columns_section += "|-------------|-----------|---------------|---------------|\n"
                for col in columns_info:
                    sample_vals = ', '.join([str(v) for v in col.get('sample_values', [])[:2]])
                    columns_section += f"| `{col['name']}` | {col['dtype']} | {sample_vals} | {col.get('unique_count', 'N/A')} |\n"
            else:
                # From web scraping
                columns_section += "| Column Name | Description |\n"
                columns_section += "|-------------|-------------|\n"
                for col in columns_info:
                    columns_section += f"| `{col['name']}` | {col.get('description', 'N/A')} |\n"
        
        # Build dataset description
        dataset_description = ""
        if scraped_data and scraped_data.get('description'):
            dataset_description = f"\n### Dataset Description\n\n{scraped_data['description']}\n"
        
        readme_content = f"""# {competition.replace('-', ' ').title()}

## Competition Information

**Competition Name:** {competition}  
**Kaggle URL:** {info['kaggle_url']}  
**Data Collected:** {info['collected_at']}

## Description

This competition focuses on machine learning and data science challenges. The notebooks in this folder contain various approaches and solutions submitted by Kaggle community members.

{dataset_description}

## Dataset Information

### Files Available

```
{info.get('dataset_files', 'Dataset information not available')}
```

{columns_section}

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
        print(f"  📝 Created COMPETITION_INFO.md with detailed dataset info")
    
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
        """Process a single competition: fetch info, scrape data page, create README, download dataset."""
        print(f"\n{'='*70}")
        print(f"Processing: {competition}")
        print(f"{'='*70}")
        
        competition_dir = self.notebooks_dir / competition
        
        if not competition_dir.exists():
            print(f"  ⚠️  Competition folder not found")
            return
        
        # Get competition info from API
        info = self.get_competition_info(competition)
        
        # Scrape competition data page for detailed info
        scraped_data = self.scrape_competition_data_page(competition)
        
        # Download dataset if requested
        columns_info = None
        if download_dataset:
            dataset_downloaded = self.download_competition_dataset(competition, competition_dir)
            info['dataset_downloaded'] = dataset_downloaded
            
            # If dataset was downloaded, try to extract column info from CSV
            if dataset_downloaded:
                dataset_dir = competition_dir / 'dataset'
                columns_info = self.get_csv_columns_from_dataset(dataset_dir)
        
        # Use scraped column info if we didn't get it from CSV
        if not columns_info and scraped_data.get('columns_info'):
            columns_info = scraped_data['columns_info']
        
        # Store column info in metadata
        if columns_info:
            info['columns_info'] = columns_info
            info['column_count'] = len(columns_info)
        
        # Create README with all collected info
        self.create_competition_readme(competition, info, competition_dir, scraped_data, columns_info)
        
        # Save metadata
        self.save_competition_metadata(competition, info, competition_dir)
        
        print(f"  ✅ Completed processing {competition}")
        
        # Rate limiting
        time.sleep(3)  # Increased to be respectful to Kaggle servers
    
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
