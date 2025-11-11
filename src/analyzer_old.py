"""
Notebook Analyzer
Quality filtering, content analysis, deduplication, and keyword tagging.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import nbformat
import pandas as pd

from .utils import (
    setup_logger,
    load_config,
    load_filters,
    load_json,
    save_json,
    compute_sha256,
    extract_code_cells,
    validate_notebook,
    Timer,
    ensure_dir,
    append_jsonl
)


class NotebookAnalyzer:
    """Analyze notebooks for quality, content, and relevance."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.filters = load_filters("config/filters.json")
        self.logger = setup_logger(
            "analyzer",
            log_dir=self.config['storage']['logs_dir'],
            level=self.config['logging']['level'],
            format_type=self.config['logging']['format']
        )
        
        # Paths
        self.notebooks_dir = Path(self.config['storage']['notebooks_dir'])
        self.curated_dir = Path(self.config['storage']['curated_dir'])
        self.metadata_dir = Path(self.config['storage']['metadata_dir'])
        self.analysis_dir = self.metadata_dir / "analysis"
        
        ensure_dir(self.curated_dir)
        ensure_dir(self.analysis_dir)
        
        # Deduplication tracking
        self.seen_hashes: Set[str] = set()
        self.hash_file = self.analysis_dir / "notebook_hashes.json"
        self._load_hashes()
        
        self.logger.info("NotebookAnalyzer initialized")
    
    def _load_hashes(self):
        """Load previously computed hashes."""
        if self.hash_file.exists():
            hashes_data = load_json(self.hash_file)
            self.seen_hashes = set(hashes_data.get('hashes', []))
            self.logger.info(f"Loaded {len(self.seen_hashes)} existing hashes")
    
    def _save_hashes(self):
        """Save computed hashes."""
        save_json({'hashes': list(self.seen_hashes)}, self.hash_file)
    
    def parse_notebook(self, notebook_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse notebook and extract metadata.
        
        Args:
            notebook_path: Path to .ipynb file
        
        Returns:
            Dictionary with notebook analysis or None if invalid
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Count cells
            code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
            markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
            
            # Extract code content
            code_content = '\n'.join(cell.source for cell in code_cells)
            markdown_content = '\n'.join(cell.source for cell in markdown_cells)
            
            # Extract imports
            imports = self._extract_imports(code_content)
            
            # Check for outputs
            has_outputs = any(
                hasattr(cell, 'outputs') and cell.outputs
                for cell in code_cells
            )
            
            # Compute hash
            code_hash = compute_sha256(code_content) if code_content else ""
            
            analysis = {
                'path': str(notebook_path),
                'filename': notebook_path.name,
                'total_cells': len(nb.cells),
                'code_cells': len(code_cells),
                'markdown_cells': len(markdown_cells),
                'code_length': len(code_content),
                'markdown_length': len(markdown_content),
                'imports': imports,
                'unique_imports': len(imports),
                'has_outputs': has_outputs,
                'code_hash': code_hash,
                'file_size_bytes': notebook_path.stat().st_size,
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to parse {notebook_path}: {e}")
            return None
    
    def _extract_imports(self, code: str) -> List[str]:
        """
        Extract import statements from code.
        
        Args:
            code: Python code string
        
        Returns:
            List of imported module names
        """
        imports = []
        
        # Match import statements
        import_patterns = [
            r'^import\s+([\w\.]+)',
            r'^from\s+([\w\.]+)\s+import',
        ]
        
        for line in code.split('\n'):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1).split('.')[0]
                    imports.append(module)
        
        return list(set(imports))
    
    def check_quality_filters(self, analysis: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if notebook meets quality criteria.
        
        Args:
            analysis: Notebook analysis dictionary
        
        Returns:
            Tuple of (passes, reasons_for_failure)
        """
        quality_filters = self.config.get('quality_filters', {})
        reasons = []
        
        # Check minimum code cells
        min_code = quality_filters.get('min_code_cells', 5)
        if analysis['code_cells'] < min_code:
            reasons.append(f"Too few code cells: {analysis['code_cells']} < {min_code}")
        
        # Check minimum markdown cells
        min_markdown = quality_filters.get('min_markdown_cells', 2)
        if analysis['markdown_cells'] < min_markdown:
            reasons.append(f"Too few markdown cells: {analysis['markdown_cells']} < {min_markdown}")
        
        # Check minimum code length
        min_code_length = quality_filters.get('min_code_length', 500)
        if analysis['code_length'] < min_code_length:
            reasons.append(f"Code too short: {analysis['code_length']} < {min_code_length}")
        
        # Check maximum code length
        max_code_length = quality_filters.get('max_code_length', 50000)
        if analysis['code_length'] > max_code_length:
            reasons.append(f"Code too long: {analysis['code_length']} > {max_code_length}")
        
        # Check minimum unique imports
        min_imports = quality_filters.get('min_unique_imports', 3)
        if analysis['unique_imports'] < min_imports:
            reasons.append(f"Too few imports: {analysis['unique_imports']} < {min_imports}")
        
        # Check required outputs
        require_outputs = quality_filters.get('require_outputs', False)
        if require_outputs and not analysis['has_outputs']:
            reasons.append("No execution outputs found")
        
        passes = len(reasons) == 0
        return passes, reasons
    
    def check_required_libraries(self, analysis: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if notebook uses required libraries.
        
        Args:
            analysis: Notebook analysis dictionary
        
        Returns:
            Tuple of (has_required, coverage_score)
        """
        required = self.filters.get('required_libraries', {})
        imports = set(analysis['imports'])
        
        # Check core libraries (pandas, numpy)
        core_libs = set(required.get('core', []))
        has_core = bool(imports & core_libs)
        
        # Check ML libraries
        ml_libs = set(required.get('ml', []))
        has_ml = bool(imports & ml_libs)
        
        # Compute coverage score
        all_required = core_libs | ml_libs
        coverage = len(imports & all_required) / len(all_required) if all_required else 0
        
        return (has_core or has_ml), coverage
    
    def compute_keyword_scores(self, notebook_path: Path) -> Dict[str, float]:
        """
        Compute keyword match scores for notebook content.
        
        Args:
            notebook_path: Path to notebook
        
        Returns:
            Dictionary of category -> score
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            keywords = self.filters.get('keywords', {})
            keyword_weights = self.config.get('keywords_weight', {})
            
            scores = {}
            
            for category, data in keywords.items():
                terms = data.get('terms', [])
                weight = keyword_weights.get(category, 1.0)
                
                # Count keyword matches
                matches = sum(1 for term in terms if term.lower() in content)
                
                # Compute normalized score
                score = (matches / len(terms)) * weight if terms else 0
                scores[category] = round(score, 3)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Failed to compute keyword scores for {notebook_path}: {e}")
            return {}
    
    def is_duplicate(self, analysis: Dict[str, Any]) -> bool:
        """
        Check if notebook is a duplicate based on hash.
        
        Args:
            analysis: Notebook analysis dictionary
        
        Returns:
            True if duplicate, False otherwise
        """
        dedup_config = self.config.get('deduplication', {})
        
        if not dedup_config.get('enabled', True):
            return False
        
        code_hash = analysis.get('code_hash', '')
        
        if not code_hash:
            return False
        
        if code_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(code_hash)
        self._save_hashes()
        
        return False
    
    def analyze_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """
        Perform complete analysis of a notebook.
        
        Args:
            notebook_path: Path to notebook
        
        Returns:
            Complete analysis dictionary
        """
        self.logger.info(f"Analyzing: {notebook_path.name}")
        
        # Parse notebook
        analysis = self.parse_notebook(notebook_path)
        
        if not analysis:
            return {
                'path': str(notebook_path),
                'status': 'parse_failed',
                'quality_pass': False
            }
        
        # Check quality
        quality_pass, quality_reasons = self.check_quality_filters(analysis)
        analysis['quality_pass'] = quality_pass
        analysis['quality_reasons'] = quality_reasons
        
        # Check required libraries
        has_required, lib_coverage = self.check_required_libraries(analysis)
        analysis['has_required_libraries'] = has_required
        analysis['library_coverage'] = round(lib_coverage, 3)
        
        # Compute keyword scores
        keyword_scores = self.compute_keyword_scores(notebook_path)
        analysis['keyword_scores'] = keyword_scores
        analysis['total_keyword_score'] = round(sum(keyword_scores.values()), 3)
        
        # Check for duplicates
        is_dup = self.is_duplicate(analysis)
        analysis['is_duplicate'] = is_dup
        
        # Compute overall quality score
        quality_score = self._compute_quality_score(analysis)
        analysis['quality_score'] = round(quality_score, 3)
        
        # Final decision
        analysis['selected'] = (
            quality_pass and
            has_required and
            not is_dup and
            quality_score > 0
        )
        
        analysis['timestamp'] = datetime.utcnow().isoformat()
        
        return analysis
    
    def _compute_quality_score(self, analysis: Dict[str, Any]) -> float:
        """
        Compute overall quality score.
        
        Args:
            analysis: Notebook analysis dictionary
        
        Returns:
            Quality score (0-100)
        """
        score = 0.0
        
        # Base score from quality pass
        if analysis.get('quality_pass', False):
            score += 30
        
        # Library coverage
        score += analysis.get('library_coverage', 0) * 20
        
        # Keyword relevance
        score += min(analysis.get('total_keyword_score', 0) * 5, 30)
        
        # Documentation (markdown cells)
        markdown_ratio = analysis['markdown_cells'] / max(analysis['total_cells'], 1)
        score += markdown_ratio * 10
        
        # Execution outputs
        if analysis.get('has_outputs', False):
            score += 10
        
        return min(score, 100)
    
    def analyze_all_notebooks(self) -> Dict[str, Any]:
        """
        Analyze all downloaded notebooks.
        
        Returns:
            Summary with analysis results
        """
        self.logger.info("Starting batch analysis of all notebooks...")
        
        # Find all notebooks
        notebook_files = list(self.notebooks_dir.rglob('*.ipynb'))
        self.logger.info(f"Found {len(notebook_files)} notebooks to analyze")
        
        results = []
        
        with Timer("Batch Analysis", self.logger):
            for i, nb_path in enumerate(notebook_files, 1):
                if i % 100 == 0:
                    self.logger.info(f"Progress: {i}/{len(notebook_files)}")
                
                analysis = self.analyze_notebook(nb_path)
                results.append(analysis)
                
                # Save individual analysis
                analysis_file = self.analysis_dir / f"{nb_path.stem}_analysis.json"
                save_json(analysis, analysis_file)
        
        # Create summary
        summary = self._create_summary(results)
        
        # Save all results
        all_results_file = self.analysis_dir / "all_notebooks_analysis.json"
        save_json(results, all_results_file)
        
        # Save as CSV
        df = pd.DataFrame(results)
        csv_file = self.analysis_dir / "all_notebooks_analysis.csv"
        df.to_csv(csv_file, index=False)
        
        # Save summary
        summary_file = self.analysis_dir / "analysis_summary.json"
        save_json(summary, summary_file)
        
        self.logger.info(f"Analysis complete: {summary['selected_count']} notebooks selected")
        
        return summary
    
    def _create_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Create summary statistics from analysis results."""
        total = len(results)
        selected = [r for r in results if r.get('selected', False)]
        
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_analyzed': total,
            'selected_count': len(selected),
            'selection_rate': len(selected) / total if total else 0,
            'quality_pass_count': sum(1 for r in results if r.get('quality_pass', False)),
            'duplicate_count': sum(1 for r in results if r.get('is_duplicate', False)),
            'has_required_libs_count': sum(1 for r in results if r.get('has_required_libraries', False)),
        }
        
        # Average scores
        if results:
            summary['avg_quality_score'] = sum(r.get('quality_score', 0) for r in results) / total
            summary['avg_keyword_score'] = sum(r.get('total_keyword_score', 0) for r in results) / total
            summary['avg_code_cells'] = sum(r.get('code_cells', 0) for r in results) / total
            summary['avg_markdown_cells'] = sum(r.get('markdown_cells', 0) for r in results) / total
        
        # Top categories
        category_scores = defaultdict(list)
        for r in selected:
            for cat, score in r.get('keyword_scores', {}).items():
                if score > 0:
                    category_scores[cat].append(score)
        
        summary['category_distribution'] = {
            cat: len(scores)
            for cat, scores in category_scores.items()
        }
        
        return summary
    
    def select_top_notebooks(self, target_count: int = 5000) -> List[Dict[str, Any]]:
        """
        Select top N notebooks based on quality scores.
        
        Args:
            target_count: Number of notebooks to select
        
        Returns:
            List of selected notebook analyses
        """
        # Load all analyses
        all_results_file = self.analysis_dir / "all_notebooks_analysis.json"
        
        if not all_results_file.exists():
            self.logger.error("No analysis results found. Run analyze_all_notebooks first.")
            return []
        
        results = load_json(all_results_file)
        
        # Filter selected notebooks
        selected = [r for r in results if r.get('selected', False)]
        
        self.logger.info(f"Found {len(selected)} selected notebooks")
        
        # Sort by quality score
        selected_sorted = sorted(
            selected,
            key=lambda x: x.get('quality_score', 0),
            reverse=True
        )
        
        # Take top N
        top_notebooks = selected_sorted[:target_count]
        
        self.logger.info(f"Selected top {len(top_notebooks)} notebooks")
        
        # Save selection
        selection_file = self.analysis_dir / f"top_{target_count}_selection.json"
        save_json(top_notebooks, selection_file)
        
        # Save as CSV
        df = pd.DataFrame(top_notebooks)
        csv_file = self.analysis_dir / f"top_{target_count}_selection.csv"
        df.to_csv(csv_file, index=False)
        
        return top_notebooks
    
    def copy_curated_notebooks(self, selection: List[Dict[str, Any]]):
        """
        Copy selected notebooks to curated directory.
        
        Args:
            selection: List of selected notebook analyses
        """
        self.logger.info(f"Copying {len(selection)} notebooks to curated directory...")
        
        import shutil
        
        copied = 0
        for analysis in selection:
            src_path = Path(analysis['path'])
            
            if not src_path.exists():
                self.logger.warning(f"Source not found: {src_path}")
                continue
            
            # Create destination path
            dst_path = self.curated_dir / src_path.name
            
            try:
                shutil.copy2(src_path, dst_path)
                copied += 1
            except Exception as e:
                self.logger.error(f"Failed to copy {src_path}: {e}")
        
        self.logger.info(f"Copied {copied} notebooks to {self.curated_dir}")


def main():
    """Command-line interface for notebook analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Kaggle notebooks")
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--analyze', action='store_true', help='Analyze all notebooks')
    parser.add_argument('--select', type=int, help='Select top N notebooks')
    parser.add_argument('--copy', action='store_true', help='Copy selected to curated directory')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = NotebookAnalyzer(config_path=args.config)
    
    # Analyze all notebooks
    if args.analyze:
        summary = analyzer.analyze_all_notebooks()
        
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Analyzed: {summary['total_analyzed']}")
        print(f"Selected: {summary['selected_count']}")
        print(f"Selection Rate: {summary['selection_rate']:.2%}")
        print(f"Quality Pass: {summary['quality_pass_count']}")
        print(f"Duplicates: {summary['duplicate_count']}")
        print(f"Has Required Libs: {summary['has_required_libs_count']}")
        print(f"\nAverage Scores:")
        print(f"  Quality: {summary.get('avg_quality_score', 0):.2f}")
        print(f"  Keyword: {summary.get('avg_keyword_score', 0):.2f}")
        print(f"{'='*60}\n")
    
    # Select top notebooks
    if args.select:
        selection = analyzer.select_top_notebooks(target_count=args.select)
        print(f"Selected top {len(selection)} notebooks")
        
        # Copy if requested
        if args.copy:
            analyzer.copy_curated_notebooks(selection)


if __name__ == "__main__":
    main()
