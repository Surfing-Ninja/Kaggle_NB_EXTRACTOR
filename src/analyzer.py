#!/usr/bin/env python3
"""
Kaggle Notebook Analyzer - Production Grade
Analyzes downloaded notebooks, filters for quality and relevance,
removes duplicates, and produces a curated dataset focused on feature engineering.

Author: AI Assistant  
Date: 2024-01-15
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import yaml
try:
    import nbformat
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: Required packages not installed. Run: pip install nbformat tqdm")

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class NotebookAnalyzer:
    """
    Production-grade notebook analyzer for quality assessment and curation.
    
    Features:
    - Deep code analysis for ML/FE techniques
    - Quality scoring system
    - Duplicate detection (exact and near)
    - Parallel processing
    - Comprehensive reporting
    """
    
    # Pattern definitions for technique detection
    PATTERNS = {
        'feature_engineering': {
            'polynomial': r'(?i)(PolynomialFeatures|poly_features)',
            'interaction': r'(?i)(interaction|multiply.*features|product.*features)',
            'aggregation': r'(?i)(groupby|\.agg\(|aggregate|rolling|expanding)',
            'time_features': r'(?i)(dt\.hour|dt\.dayofweek|dt\.month|dt\.year|DatetimeIndex)',
            'lag_features': r'(?i)(shift\(|\.lag|previous.*value)',
            'label_encoding': r'(?i)(LabelEncoder|label_encode)',
            'onehot_encoding': r'(?i)(OneHotEncoder|get_dummies|pd\.get_dummies)',
            'target_encoding': r'(?i)(target_encode|mean_encode)',
            'scaling': r'(?i)(StandardScaler|MinMaxScaler|RobustScaler|Normalizer)',
            'binning': r'(?i)(pd\.cut\(|pd\.qcut\(|\.cut\(|\.qcut\(|binning)',
            'text_features': r'(?i)(TfidfVectorizer|CountVectorizer|word2vec|Word2Vec)',
            'embedding': r'(?i)(embedding|Embedding\()',
        },
        'eda': {
            'histogram': r'(?i)(\.hist\(|distplot|histogram|kdeplot)',
            'correlation': r'(?i)(\.corr\(|correlation|heatmap.*corr)',
            'scatter': r'(?i)(scatter|scatterplot|pairplot|jointplot)',
            'missing_analysis': r'(?i)(isnull|isna|missing.*values|fillna)',
            'outlier_detection': r'(?i)(outlier|boxplot|IQR|interquartile)',
            'statistics': r'(?i)(\.describe\(|value_counts|\.info\()',
            'visualization': r'(?i)(plt\.plot|sns\.|seaborn|matplotlib)',
        },
        'dimensionality': {
            'pca': r'(?i)(PCA\(|principal.*component|explained_variance)',
            'tsne': r'(?i)(TSNE\(|t-sne|tsne)',
            'umap': r'(?i)(UMAP\(|umap)',
            'svd': r'(?i)(TruncatedSVD|SVD\()',
            'nmf': r'(?i)(NMF\(|non-negative.*matrix)',
            'autoencoder': r'(?i)(autoencoder|encoder.*decoder)',
        },
        'hyperparameter': {
            'grid_search': r'(?i)(GridSearchCV|grid.*search)',
            'random_search': r'(?i)(RandomizedSearchCV|random.*search)',
            'bayesian_opt': r'(?i)(BayesianOptimization|bayesian.*optim)',
            'optuna': r'(?i)(optuna|create_study|study\.optimize)',
            'cross_validation': r'(?i)(cross_val_score|KFold|StratifiedKFold|cross.*validat)',
        },
        'advanced_ml': {
            'voting': r'(?i)(VotingClassifier|VotingRegressor)',
            'stacking': r'(?i)(StackingClassifier|StackingRegressor|stacking)',
            'xgboost': r'(?i)(xgboost|XGBClassifier|XGBRegressor|xgb\.)',
            'lightgbm': r'(?i)(lightgbm|LGBMClassifier|LGBMRegressor|lgb\.)',
            'catboost': r'(?i)(catboost|CatBoostClassifier|CatBoostRegressor)',
            'neural_network': r'(?i)(keras|tensorflow|torch|nn\.Module|Sequential|Dense)',
            'ensemble': r'(?i)(RandomForest|GradientBoosting|AdaBoost|ExtraTree)',
        }
    }
    
    LIBRARY_PATTERNS = {
        'pandas': r'import pandas|from pandas',
        'numpy': r'import numpy|from numpy',
        'sklearn': r'from sklearn|import sklearn',
        'xgboost': r'import xgboost|from xgboost',
        'lightgbm': r'import lightgbm|from lightgbm',
        'catboost': r'import catboost|from catboost',
        'matplotlib': r'import matplotlib|from matplotlib',
        'seaborn': r'import seaborn|from seaborn',
        'plotly': r'import plotly|from plotly',
        'tensorflow': r'import tensorflow|from tensorflow',
        'keras': r'import keras|from keras',
        'pytorch': r'import torch|from torch',
    }
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the analyzer."""
        # Load config
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Setup logging
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / 'analyzer.log'
        self.setup_logging()
        
        self.logger.info("NotebookAnalyzer initialized")
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'min_score': 50,
            'min_code_cells': 5,
            'min_code_length': 500,
            'target_count': 5000,
            'max_per_competition': 15,
            'workers': 4,
            'category_quotas': {
                'feature_engineering_min': 0.30,
                'hyperparameter_min': 0.20,
                'dimensionality_min': 0.15,
                'eda_min': 0.40
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('notebook_analyzer')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def parse_notebook(self, notebook_path: Path) -> Dict:
        """Extract and parse notebook content safely."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            metadata = {
                'kernel_version': nb.metadata.get('kernelspec', {}).get('display_name', 'unknown'),
                'language': nb.metadata.get('kernelspec', {}).get('language', 'unknown'),
                'kaggle_id': nb.metadata.get('kaggle', {}).get('id', None),
                'author': nb.metadata.get('kaggle', {}).get('author', None),
            }
            
            code_cells = []
            markdown_cells = []
            
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    code_cells.append({
                        'source': cell.source,
                        'execution_count': cell.execution_count,
                        'outputs': len(cell.get('outputs', []))
                    })
                elif cell.cell_type == 'markdown':
                    markdown_cells.append({
                        'source': cell.source,
                        'length': len(cell.source)
                    })
            
            return {
                'success': True,
                'metadata': metadata,
                'code_cells': code_cells,
                'markdown_cells': markdown_cells,
                'parse_error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'metadata': {},
                'code_cells': [],
                'markdown_cells': [],
                'parse_error': str(e)
            }
    
    def extract_features_from_code(self, code_cells: List[Dict]) -> Dict:
        """Analyze code content for ML/FE techniques."""
        all_code = '\n'.join([cell['source'] for cell in code_cells])
        
        techniques_detected = defaultdict(list)
        technique_count = defaultdict(int)
        
        for category, patterns in self.PATTERNS.items():
            for technique_name, pattern in patterns.items():
                if re.search(pattern, all_code):
                    techniques_detected[category].append(technique_name)
                    technique_count[category] += 1
        
        libraries_used = []
        for lib_name, pattern in self.LIBRARY_PATTERNS.items():
            if re.search(pattern, all_code):
                libraries_used.append(lib_name)
        
        has_custom_functions = bool(re.search(r'\bdef\s+\w+\s*\(', all_code))
        has_custom_classes = bool(re.search(r'\bclass\s+\w+', all_code))
        
        # Calculate code complexity score (0-100)
        complexity_score = 0
        code_length = len(all_code)
        complexity_score += min(code_length / 10000 * 20, 20)
        
        function_count = len(re.findall(r'\bdef\s+\w+\s*\(', all_code))
        complexity_score += min(function_count * 3, 15)
        
        class_count = len(re.findall(r'\bclass\s+\w+', all_code))
        complexity_score += min(class_count * 5, 10)
        
        control_flow = len(re.findall(r'\b(if|for|while|try)\b', all_code))
        complexity_score += min(control_flow / 20 * 15, 15)
        
        complexity_score += min(len(libraries_used) * 2, 20)
        
        advanced_constructs = len(re.findall(
            r'(lambda|comprehension|@decorator|yield|async|await)', all_code
        ))
        complexity_score += min(advanced_constructs * 4, 20)
        
        return {
            'techniques_detected': dict(techniques_detected),
            'technique_count': dict(technique_count),
            'libraries_used': libraries_used,
            'has_custom_functions': has_custom_functions,
            'has_custom_classes': has_custom_classes,
            'code_complexity_score': min(complexity_score, 100),
            'code_length': code_length,
            'function_count': function_count,
            'class_count': class_count
        }
    
    def analyze_markdown_content(self, markdown_cells: List[Dict]) -> Dict:
        """Extract insights from markdown documentation."""
        if not markdown_cells:
            return {
                'has_documentation': False,
                'documentation_quality': 0,
                'sections_detected': [],
                'has_math_formulas': False,
                'avg_markdown_length': 0,
                'total_markdown_length': 0
            }
        
        all_markdown = '\n'.join([cell['source'] for cell in markdown_cells])
        total_length = len(all_markdown)
        avg_length = total_length / len(markdown_cells)
        
        section_patterns = {
            'Feature Engineering': r'(?i)#.*feature.*engineering',
            'EDA': r'(?i)#.*(eda|exploratory.*data.*analysis)',
            'Data Preprocessing': r'(?i)#.*data.*preprocessing',
            'Model Training': r'(?i)#.*(model|training)',
            'Hyperparameter': r'(?i)#.*hyperparameter',
            'Evaluation': r'(?i)#.*(evaluation|results)',
        }
        
        sections_detected = []
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, all_markdown):
                sections_detected.append(section_name)
        
        has_math_formulas = bool(re.search(r'\$\$|\$[^$]+\$|\\begin\{equation\}', all_markdown))
        
        doc_quality = 0
        doc_quality += min(total_length / 2000 * 30, 30)
        doc_quality += len(sections_detected) * 5
        
        substantial_cells = sum(1 for cell in markdown_cells if cell['length'] > 100)
        doc_quality += min(substantial_cells * 4, 20)
        
        if has_math_formulas:
            doc_quality += 10
        
        has_lists = bool(re.search(r'^\s*[-*+]\s', all_markdown, re.MULTILINE))
        has_numbered = bool(re.search(r'^\s*\d+\.\s', all_markdown, re.MULTILINE))
        if has_lists or has_numbered:
            doc_quality += 10
        
        return {
            'has_documentation': True,
            'documentation_quality': min(doc_quality, 100),
            'sections_detected': sections_detected,
            'has_math_formulas': has_math_formulas,
            'avg_markdown_length': avg_length,
            'total_markdown_length': total_length,
            'markdown_cell_count': len(markdown_cells)
        }
    
    def calculate_quality_score(self, notebook_analysis: Dict, features: Dict, documentation: Dict) -> Dict:
        """Assign comprehensive quality score."""
        scores = {
            'code_quality': 0,
            'techniques': 0,
            'documentation': 0,
            'libraries': 0
        }
        
        # CODE QUALITY (40 points max)
        code_cells_count = len(notebook_analysis.get('code_cells', []))
        scores['code_quality'] += min(code_cells_count / 20 * 10, 10)
        
        code_length = features.get('code_length', 0)
        scores['code_quality'] += min(code_length / 5000 * 10, 10)
        
        complexity = features.get('code_complexity_score', 0)
        scores['code_quality'] += complexity * 0.1
        
        if features.get('has_custom_functions'):
            scores['code_quality'] += 5
        if features.get('has_custom_classes'):
            scores['code_quality'] += 5
        
        # TECHNIQUE COVERAGE (30 points max)
        technique_counts = features.get('technique_count', {})
        
        fe_count = technique_counts.get('feature_engineering', 0)
        scores['techniques'] += min(fe_count * 2, 10)
        
        hp_count = technique_counts.get('hyperparameter', 0)
        scores['techniques'] += min(hp_count * 3, 10)
        
        dim_count = technique_counts.get('dimensionality', 0)
        scores['techniques'] += min(dim_count * 2, 5)
        
        ml_count = technique_counts.get('advanced_ml', 0)
        scores['techniques'] += min(ml_count * 1, 5)
        
        # DOCUMENTATION (15 points max)
        doc_quality = documentation.get('documentation_quality', 0)
        scores['documentation'] = doc_quality * 0.15
        
        # LIBRARIES & TOOLS (15 points max)
        libraries = features.get('libraries_used', [])
        
        if 'pandas' in libraries:
            scores['libraries'] += 3
        if 'sklearn' in libraries:
            scores['libraries'] += 3
        if any(lib in libraries for lib in ['xgboost', 'lightgbm', 'catboost']):
            scores['libraries'] += 4
        if any(lib in libraries for lib in ['matplotlib', 'seaborn', 'plotly']):
            scores['libraries'] += 2
        if any(lib in libraries for lib in ['tensorflow', 'keras', 'pytorch']):
            scores['libraries'] += 3
        
        # BONUS POINTS
        bonus_points = 0
        
        if (fe_count > 0 and technique_counts.get('eda', 0) > 0 and hp_count > 0):
            bonus_points += 10
        
        if dim_count > 0:
            bonus_points += 5
        
        if doc_quality > 70:
            bonus_points += 5
        
        total_score = sum(scores.values()) + bonus_points
        total_score = min(total_score, 100)
        
        if total_score >= 80:
            grade = 'A'
        elif total_score >= 60:
            grade = 'B'
        elif total_score >= 40:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'total_score': total_score,
            'component_scores': scores,
            'bonus_points': bonus_points,
            'grade': grade
        }
    
    def compute_file_hash(self, notebook_path: Path) -> str:
        """Compute SHA256 for exact duplicate detection."""
        hasher = hashlib.sha256()
        
        try:
            with open(notebook_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to hash {notebook_path}: {e}")
            return "error"
    
    def compute_content_hash(self, code_cells: List[Dict]) -> str:
        """Compute similarity hash for near-duplicate detection."""
        all_code = '\n'.join([cell['source'] for cell in code_cells])
        
        # Normalize
        normalized = re.sub(r'#.*$', '', all_code, flags=re.MULTILINE)
        normalized = re.sub(r'""".*?"""', '', normalized, flags=re.DOTALL)
        normalized = re.sub(r"'''.*?'''", '', normalized, flags=re.DOTALL)
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.lower()
        
        common_imports = [
            'import pandas', 'import numpy', 'import matplotlib',
            'from sklearn', 'import seaborn'
        ]
        for imp in common_imports:
            normalized = normalized.replace(imp, '')
        
        hasher = hashlib.sha256()
        hasher.update(normalized.encode('utf-8'))
        return hasher.hexdigest()
    
    def analyze_single_notebook(self, notebook_path: Path) -> Dict:
        """Complete analysis pipeline for one notebook."""
        parse_result = self.parse_notebook(notebook_path)
        
        if not parse_result['success']:
            return {
                'notebook_path': str(notebook_path),
                'file_hash': self.compute_file_hash(notebook_path),
                'content_hash': 'error',
                'parsing_success': False,
                'parse_error': parse_result['parse_error'],
                'metrics': {},
                'features': {},
                'documentation': {},
                'quality': {'total_score': 0, 'grade': 'D'},
                'timestamp': datetime.now().isoformat()
            }
        
        features = self.extract_features_from_code(parse_result['code_cells'])
        documentation = self.analyze_markdown_content(parse_result['markdown_cells'])
        quality = self.calculate_quality_score(parse_result, features, documentation)
        
        file_hash = self.compute_file_hash(notebook_path)
        content_hash = self.compute_content_hash(parse_result['code_cells'])
        
        metrics = {
            'code_cells': len(parse_result['code_cells']),
            'markdown_cells': len(parse_result['markdown_cells']),
            'total_code_length': sum(len(cell['source']) for cell in parse_result['code_cells']),
            'total_markdown_length': sum(cell['length'] for cell in parse_result['markdown_cells']),
        }
        
        return {
            'notebook_path': str(notebook_path),
            'file_hash': file_hash,
            'content_hash': content_hash,
            'parsing_success': True,
            'metadata': parse_result['metadata'],
            'metrics': metrics,
            'features': features,
            'documentation': documentation,
            'quality': quality,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_all_notebooks(self, notebooks_dir: Path, parallel: bool = True, workers: int = 4) -> List[Dict]:
        """Analyze all downloaded notebooks with parallel processing."""
        notebook_files = list(notebooks_dir.rglob('*.ipynb'))
        total_notebooks = len(notebook_files)
        
        print(f"\n{Colors.CYAN}Found {total_notebooks} notebooks to analyze{Colors.RESET}\n")
        self.logger.info(f"Found {total_notebooks} notebooks")
        
        if total_notebooks == 0:
            return []
        
        all_analyses = []
        checkpoint_path = Path('metadata/analysis_checkpoint.json')
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        if parallel and workers > 1:
            print(f"{Colors.CYAN}Using {workers} parallel workers{Colors.RESET}\n")
            
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_notebook = {
                    executor.submit(self.analyze_single_notebook, nb_path): nb_path
                    for nb_path in notebook_files
                }
                
                iterator = tqdm(
                    as_completed(future_to_notebook),
                    total=total_notebooks,
                    desc="Analyzing notebooks",
                    unit="notebook"
                ) if TQDM_AVAILABLE else as_completed(future_to_notebook)
                
                for idx, future in enumerate(iterator, 1):
                    try:
                        analysis = future.result()
                        all_analyses.append(analysis)
                        
                        if idx % 100 == 0:
                            with open(checkpoint_path, 'w') as f:
                                json.dump(all_analyses, f)
                            self.logger.info(f"Checkpoint saved at {idx} notebooks")
                    
                    except Exception as e:
                        notebook_path = future_to_notebook[future]
                        self.logger.error(f"Failed to analyze {notebook_path}: {e}")
        
        else:
            print(f"{Colors.CYAN}Analyzing sequentially...{Colors.RESET}\n")
            
            iterator = tqdm(notebook_files, desc="Analyzing", unit="notebook") if TQDM_AVAILABLE else notebook_files
            
            for idx, nb_path in enumerate(iterator, 1):
                try:
                    analysis = self.analyze_single_notebook(nb_path)
                    all_analyses.append(analysis)
                    
                    if idx % 100 == 0:
                        with open(checkpoint_path, 'w') as f:
                            json.dump(all_analyses, f)
                        self.logger.info(f"Checkpoint saved at {idx} notebooks")
                
                except Exception as e:
                    self.logger.error(f"Failed to analyze {nb_path}: {e}")
        
        output_path = Path('metadata/notebook_analyses.json')
        with open(output_path, 'w') as f:
            json.dump(all_analyses, f, indent=2)
        
        self.logger.info(f"Analysis complete. Results saved to {output_path}")
        self._print_analysis_statistics(all_analyses)
        
        return all_analyses
    
    def _print_analysis_statistics(self, analyses: List[Dict]):
        """Print analysis statistics."""
        total = len(analyses)
        successful = sum(1 for a in analyses if a['parsing_success'])
        
        scores = [a['quality']['total_score'] for a in analyses if a['parsing_success']]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        grade_counts = Counter(a['quality']['grade'] for a in analyses if a['parsing_success'])
        
        all_techniques = []
        for analysis in analyses:
            if analysis['parsing_success']:
                for category, techniques in analysis['features'].get('techniques_detected', {}).items():
                    all_techniques.extend(techniques)
        
        technique_freq = Counter(all_techniques)
        top_techniques = technique_freq.most_common(5)
        
        print(f"\n{Colors.BOLD}Analysis Statistics:{Colors.RESET}")
        print(f"  Total analyzed: {total}")
        print(f"  Successfully parsed: {successful} ({successful/total*100:.1f}%)")
        print(f"  Average quality score: {avg_score:.1f}")
        print(f"\n{Colors.BOLD}Grade Distribution:{Colors.RESET}")
        for grade in ['A', 'B', 'C', 'D']:
            count = grade_counts.get(grade, 0)
            print(f"  Grade {grade}: {count} ({count/successful*100:.1f}%)")
        
        print(f"\n{Colors.BOLD}Top Techniques:{Colors.RESET}")
        for technique, count in top_techniques:
            print(f"  {technique}: {count} notebooks")
        print()
    
    def detect_duplicates(self, all_analyses: List[Dict]) -> Dict:
        """Find and group duplicate/near-duplicate notebooks."""
        print(f"\n{Colors.CYAN}Detecting duplicates...{Colors.RESET}")
        
        file_hash_groups = defaultdict(list)
        for analysis in all_analyses:
            if analysis['parsing_success']:
                file_hash_groups[analysis['file_hash']].append(analysis)
        
        exact_duplicates = {
            h: [a['notebook_path'] for a in analyses]
            for h, analyses in file_hash_groups.items()
            if len(analyses) > 1
        }
        
        content_hash_groups = defaultdict(list)
        for analysis in all_analyses:
            if analysis['parsing_success']:
                content_hash_groups[analysis['content_hash']].append(analysis)
        
        near_duplicates = {
            h: [a['notebook_path'] for a in analyses]
            for h, analyses in content_hash_groups.items()
            if len(analyses) > 1 and h not in exact_duplicates
        }
        
        notebooks_to_remove = []
        notebooks_to_keep = []
        
        for hash_val, analyses in file_hash_groups.items():
            if len(analyses) > 1:
                sorted_analyses = sorted(
                    analyses,
                    key=lambda a: a['quality']['total_score'],
                    reverse=True
                )
                notebooks_to_keep.append(sorted_analyses[0]['notebook_path'])
                notebooks_to_remove.extend([a['notebook_path'] for a in sorted_analyses[1:]])
        
        for hash_val, analyses in content_hash_groups.items():
            if len(analyses) > 1 and hash_val not in exact_duplicates:
                sorted_analyses = sorted(
                    analyses,
                    key=lambda a: a['quality']['total_score'],
                    reverse=True
                )
                notebooks_to_keep.append(sorted_analyses[0]['notebook_path'])
                notebooks_to_remove.extend([a['notebook_path'] for a in sorted_analyses[1:]])
        
        print(f"  Found {len(exact_duplicates)} exact duplicate groups")
        print(f"  Found {len(near_duplicates)} near-duplicate groups")
        print(f"  {len(notebooks_to_remove)} notebooks marked for removal")
        
        return {
            'exact_duplicates': exact_duplicates,
            'near_duplicates': near_duplicates,
            'notebooks_to_remove': notebooks_to_remove,
            'notebooks_to_keep': notebooks_to_keep
        }
    
    def filter_by_quality_threshold(self, analyses: List[Dict], duplicates_info: Dict,
                                     min_score: int = 50, target_count: int = 5000) -> List[Dict]:
        """Filter notebooks that meet quality criteria."""
        print(f"\n{Colors.CYAN}Filtering by quality threshold...{Colors.RESET}")
        
        notebooks_to_remove = set(duplicates_info['notebooks_to_remove'])
        filtered = [a for a in analyses if a['notebook_path'] not in notebooks_to_remove]
        
        filtered = [
            a for a in filtered
            if (
                a['parsing_success'] and
                a['quality']['total_score'] >= min_score and
                a['metrics'].get('code_cells', 0) >= self.config.get('min_code_cells', 5) and
                a['metrics'].get('total_code_length', 0) >= self.config.get('min_code_length', 500) and
                (
                    a['features'].get('technique_count', {}).get('feature_engineering', 0) > 0 or
                    a['features'].get('technique_count', {}).get('hyperparameter', 0) > 0 or
                    a['features'].get('technique_count', {}).get('dimensionality', 0) > 0
                )
            )
        ]
        
        print(f"  {len(filtered)} notebooks passed basic filters")
        
        if len(filtered) > target_count:
            filtered = sorted(filtered, key=lambda a: a['quality']['total_score'], reverse=True)
            
            competition_counts = Counter()
            final_filtered = []
            max_per_comp = self.config.get('max_per_competition', 15)
            
            for analysis in filtered:
                path_parts = Path(analysis['notebook_path']).parts
                competition = path_parts[-3] if len(path_parts) >= 3 else 'unknown'
                
                if competition_counts[competition] < max_per_comp:
                    final_filtered.append(analysis)
                    competition_counts[competition] += 1
                
                if len(final_filtered) >= target_count:
                    break
            
            filtered = final_filtered
        
        print(f"  Selected {len(filtered)} notebooks for final dataset")
        
        return filtered
    
    def curate_final_dataset(self, filtered_analyses: List[Dict], output_dir: Path) -> Dict:
        """Create final curated dataset."""
        print(f"\n{Colors.CYAN}Curating final dataset...{Colors.RESET}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        categories = {
            'pure_fe': [],
            'eda_fe': [],
            'hyperparam_fe': [],
            'complete_pipeline': [],
            'advanced': []
        }
        
        for analysis in filtered_analyses:
            tc = analysis['features'].get('technique_count', {})
            fe_count = tc.get('feature_engineering', 0)
            eda_count = tc.get('eda', 0)
            hp_count = tc.get('hyperparameter', 0)
            dim_count = tc.get('dimensionality', 0)
            ml_count = tc.get('advanced_ml', 0)
            
            if fe_count > 3 and eda_count == 0 and hp_count == 0:
                categories['pure_fe'].append(analysis)
            elif fe_count > 0 and eda_count > 0 and hp_count == 0:
                categories['eda_fe'].append(analysis)
            elif fe_count > 0 and hp_count > 0 and eda_count == 0:
                categories['hyperparam_fe'].append(analysis)
            elif fe_count > 0 and eda_count > 0 and hp_count > 0:
                categories['complete_pipeline'].append(analysis)
            elif fe_count > 0 and dim_count > 0 and ml_count > 0:
                categories['advanced'].append(analysis)
            else:
                categories['eda_fe'].append(analysis)
        
        total_size = 0
        competitions = set()
        authors = set()
        
        for category, analyses in categories.items():
            category_dir = output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            print(f"  Processing {len(analyses)} notebooks in '{category}'...")
            
            for analysis in analyses:
                try:
                    src_path = Path(analysis['notebook_path'])
                    path_parts = src_path.parts
                    competition = path_parts[-3] if len(path_parts) >= 3 else 'unknown'
                    notebook_name = path_parts[-2] if len(path_parts) >= 2 else src_path.stem
                    
                    competitions.add(competition)
                    
                    dest_dir = category_dir / competition / notebook_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    dest_file = dest_dir / src_path.name
                    shutil.copy2(src_path, dest_file)
                    
                    metadata_file = src_path.parent / 'kernel-metadata.json'
                    if metadata_file.exists():
                        shutil.copy2(metadata_file, dest_dir / 'kernel-metadata.json')
                        
                        try:
                            with open(metadata_file, 'r') as f:
                                meta = json.load(f)
                                if 'author' in meta:
                                    authors.add(meta['author'])
                        except:
                            pass
                    
                    meta_file = dest_dir / f"{src_path.stem}.meta.json"
                    with open(meta_file, 'w') as f:
                        json.dump({
                            'original_path': str(src_path),
                            'quality_score': analysis['quality']['total_score'],
                            'grade': analysis['quality']['grade'],
                            'techniques': analysis['features'].get('techniques_detected', {}),
                            'libraries': analysis['features'].get('libraries_used', []),
                            'category': category,
                            'competition': competition
                        }, f, indent=2)
                    
                    readme_file = dest_dir / 'README.md'
                    kaggle_id = analysis.get('metadata', {}).get('kaggle_id', 'unknown')
                    kaggle_url = f"https://www.kaggle.com/code/{kaggle_id}" if kaggle_id != 'unknown' else 'N/A'
                    
                    readme_content = f"""# {notebook_name}

## Original Source
- **Kaggle URL**: {kaggle_url}
- **Competition**: {competition}
- **Category**: {category}

## Quality Metrics
- **Quality Score**: {analysis['quality']['total_score']:.1f}/100
- **Grade**: {analysis['quality']['grade']}

## Techniques Detected
"""
                    for cat, techniques in analysis['features'].get('techniques_detected', {}).items():
                        if techniques:
                            readme_content += f"\n### {cat.replace('_', ' ').title()}\n"
                            for tech in techniques:
                                readme_content += f"- {tech}\n"
                    
                    readme_content += f"""
## Libraries Used
{', '.join(analysis['features'].get('libraries_used', []))}

## License
This notebook is from Kaggle and follows Kaggle's terms of use.
"""
                    
                    with open(readme_file, 'w') as f:
                        f.write(readme_content)
                    
                    total_size += src_path.stat().st_size
                
                except Exception as e:
                    self.logger.error(f"Failed to curate {analysis['notebook_path']}: {e}")
        
        catalog = {
            'generated_at': datetime.now().isoformat(),
            'total_notebooks': sum(len(analyses) for analyses in categories.values()),
            'categories': {cat: len(analyses) for cat, analyses in categories.items()},
            'statistics': {
                'avg_quality_score': sum(a['quality']['total_score'] for analyses in categories.values() for a in analyses) / len(filtered_analyses) if filtered_analyses else 0,
                'competitions_represented': len(competitions),
                'unique_authors': len(authors),
                'total_size_gb': total_size / (1024**3)
            },
            'notebooks': [
                {
                    'path': str(Path(analysis['notebook_path']).relative_to(Path('notebooks'))),
                    'quality_score': analysis['quality']['total_score'],
                    'category': category,
                    'competition': Path(analysis['notebook_path']).parts[-3] if len(Path(analysis['notebook_path']).parts) >= 3 else 'unknown'
                }
                for category, analyses in categories.items()
                for analysis in analyses
            ]
        }
        
        catalog_path = Path('metadata/curated_catalog.json')
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        print(f"\n{Colors.GREEN}✓ Catalog saved to {catalog_path}{Colors.RESET}")
        
        return catalog
    
    def generate_analysis_report(self, analyses: List[Dict], catalog: Dict):
        """Generate comprehensive analysis report."""
        print(f"\n{Colors.CYAN}Generating analysis report...{Colors.RESET}")
        
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / 'analysis_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("NOTEBOOK ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Notebooks Analyzed: {len(analyses)}\n")
            f.write(f"Final Curated Count: {catalog['total_notebooks']}\n")
            f.write(f"Average Quality Score: {catalog['statistics']['avg_quality_score']:.1f}\n")
            f.write(f"Competitions Represented: {catalog['statistics']['competitions_represented']}\n")
            f.write(f"Total Size: {catalog['statistics']['total_size_gb']:.2f} GB\n\n")
            
            f.write("CATEGORY DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            for category, count in catalog['categories'].items():
                pct = count / catalog['total_notebooks'] * 100 if catalog['total_notebooks'] > 0 else 0
                f.write(f"{category:30s}: {count:5d} ({pct:5.1f}%)\n")
            f.write("\n")
            
            all_techniques = []
            for analysis in analyses:
                if analysis['parsing_success']:
                    for cat, techniques in analysis['features'].get('techniques_detected', {}).items():
                        all_techniques.extend(techniques)
            
            technique_freq = Counter(all_techniques).most_common(10)
            
            f.write("TOP TECHNIQUES DETECTED\n")
            f.write("-" * 70 + "\n")
            for idx, (tech, count) in enumerate(technique_freq, 1):
                f.write(f"{idx:2d}. {tech:40s}: {count:5d} notebooks\n")
        
        print(f"{Colors.GREEN}✓ Report saved to {report_path}{Colors.RESET}")
    
    def run(self, input_dir: Path, output_dir: Path, min_score: int = 50,
            target_count: int = 5000, workers: int = 4):
        """Main orchestration."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}KAGGLE NOTEBOOK ANALYZER{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        print(f"{Colors.BOLD}Configuration:{Colors.RESET}")
        print(f"  Input directory: {input_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Minimum score: {min_score}")
        print(f"  Target count: {target_count}")
        print(f"  Workers: {workers}")
        
        analyses = self.analyze_all_notebooks(input_dir, parallel=(workers > 1), workers=workers)
        
        if not analyses:
            print(f"{Colors.RED}No notebooks found to analyze!{Colors.RESET}")
            return
        
        duplicates_info = self.detect_duplicates(analyses)
        filtered = self.filter_by_quality_threshold(analyses, duplicates_info, min_score=min_score, target_count=target_count)
        catalog = self.curate_final_dataset(filtered, output_dir)
        self.generate_analysis_report(analyses, catalog)
        self._print_final_summary(analyses, duplicates_info, catalog)
    
    def _print_final_summary(self, analyses: List[Dict], duplicates_info: Dict, catalog: Dict):
        """Print final summary."""
        total = len(analyses)
        successful = sum(1 for a in analyses if a['parsing_success'])
        
        grade_counts = Counter(a['quality']['grade'] for a in analyses if a['parsing_success'])
        
        all_techniques = []
        for analysis in analyses:
            if analysis['parsing_success']:
                for cat, techniques in analysis['features'].get('techniques_detected', {}).items():
                    all_techniques.extend(techniques)
        
        technique_freq = Counter(all_techniques).most_common(5)
        
        print(f"\n{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}║           NOTEBOOK ANALYSIS COMPLETE                         ║{Colors.RESET}")
        print(f"{Colors.BOLD}╠══════════════════════════════════════════════════════════════╣{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} Total Analyzed:       {total:<40} {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} {Colors.GREEN}✅ High Quality:{Colors.RESET}      {grade_counts.get('A', 0) + grade_counts.get('B', 0):<6} (Grade A/B)               {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} {Colors.YELLOW}⚠️  Medium Quality:{Colors.RESET}   {grade_counts.get('C', 0):<6} (Grade C)                 {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} {Colors.RED}❌ Low Quality:{Colors.RESET}       {grade_counts.get('D', 0):<6} (Grade D)                 {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET}                                                              {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} Duplicates Removed:   {len(duplicates_info['notebooks_to_remove']):<40} {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET}                                                              {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} CATEGORY DISTRIBUTION:                                       {Colors.BOLD}║{Colors.RESET}")
        
        for category, count in catalog['categories'].items():
            pct = count / catalog['total_notebooks'] * 100 if catalog['total_notebooks'] > 0 else 0
            label = category.replace('_', ' ').title()
            print(f"{Colors.BOLD}║{Colors.RESET} {label:20s} {count:<6} ({pct:>5.1f}%)                   {Colors.BOLD}║{Colors.RESET}")
        
        print(f"{Colors.BOLD}║{Colors.RESET}                                                              {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} TOP TECHNIQUES FOUND:                                        {Colors.BOLD}║{Colors.RESET}")
        
        for idx, (tech, count) in enumerate(technique_freq, 1):
            print(f"{Colors.BOLD}║{Colors.RESET} {idx}. {tech:30s} ({count:>4} notebooks)        {Colors.BOLD}║{Colors.RESET}")
        
        print(f"{Colors.BOLD}║{Colors.RESET}                                                              {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} 📁 Curated notebooks: notebooks_curated/                     {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} 📊 Full report:       reports/analysis_summary.txt           {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze and curate Kaggle notebooks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', default='notebooks', help='Input directory with downloaded notebooks')
    parser.add_argument('--output', default='notebooks_curated', help='Output directory for curated notebooks')
    parser.add_argument('--min-score', type=int, default=50, help='Minimum quality score')
    parser.add_argument('--target-count', type=int, default=5000, help='Target number of notebooks')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"{Colors.RED}Error: Input directory not found: {input_dir}{Colors.RESET}")
        print(f"{Colors.YELLOW}Download notebooks first using:{Colors.RESET}")
        print(f"  python src/downloader.py")
        sys.exit(1)
    
    analyzer = NotebookAnalyzer(config_path=args.config)
    analyzer.run(
        input_dir=input_dir,
        output_dir=Path(args.output),
        min_score=args.min_score,
        target_count=args.target_count,
        workers=args.workers
    )
    
    print(f"\n{Colors.GREEN}✨ Analysis complete!{Colors.RESET}\n")


if __name__ == '__main__':
    main()
