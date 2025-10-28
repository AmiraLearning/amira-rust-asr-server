#!/usr/bin/env python3
"""
Build User Language Models from Historical Transcripts

This script processes historical ASR transcripts to build personalized
language models (LMs) for each user, then compiles them to k2 FST format.

Workflow:
1. Aggregate transcripts per user
2. Clean and normalize text
3. Train ARPA n-gram LM using KenLM
4. Convert ARPA to k2 FST format
5. Save to user FST directory

Usage:
    # Single user from transcripts
    python build_lm_from_transcripts.py \
        --user-id user123 \
        --transcripts-dir /data/transcripts/user123 \
        --output-dir /models/user_fsts

    # Batch build for all users
    python build_lm_from_transcripts.py \
        --batch \
        --transcripts-root /data/transcripts \
        --output-dir /models/user_fsts \
        --min-utterances 50

    # From database query results
    python build_lm_from_transcripts.py \
        --user-id user123 \
        --from-json transcripts.json \
        --output-dir /models/user_fsts

Requirements:
    pip install k2 torch kaldi_native_io
    brew install kenlm  # or build from source
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import k2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """Process and normalize ASR transcripts for LM training"""

    def __init__(self):
        # Common ASR artifacts to remove or normalize
        self.noise_tokens = {
            '[noise]', '[laughter]', '[music]', '[inaudible]',
            '<unk>', '<eps>', '(inaudible)', '(crosstalk)'
        }

        # Punctuation to preserve (helps n-gram boundaries)
        self.keep_punct = {'.', '?', '!', ','}

    def clean_transcript(self, text: str) -> str:
        """
        Clean and normalize a single transcript line

        Args:
            text: Raw transcript text

        Returns:
            Cleaned, normalized text suitable for LM training
        """
        # Convert to lowercase
        text = text.lower()

        # Remove noise tokens
        for token in self.noise_tokens:
            text = text.replace(token.lower(), '')

        # Normalize whitespace
        text = ' '.join(text.split())

        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Normalize numbers (optional - comment out to keep verbatim)
        # text = re.sub(r'\b\d+\b', '<number>', text)

        # Remove punctuation except those we want to keep
        punct_pattern = f"[^a-z0-9\\s{''.join(self.keep_punct)}']"
        text = re.sub(punct_pattern, ' ', text)

        # Remove extra spaces
        text = ' '.join(text.split())

        return text.strip()

    def aggregate_transcripts(
        self,
        transcript_files: List[Path],
        min_length: int = 5
    ) -> List[str]:
        """
        Read and aggregate multiple transcript files

        Args:
            transcript_files: List of paths to transcript files
            min_length: Minimum word count to include utterance

        Returns:
            List of cleaned transcript lines
        """
        utterances = []
        stats = {'total': 0, 'cleaned': 0, 'too_short': 0, 'empty': 0}

        for file_path in transcript_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        stats['total'] += 1
                        cleaned = self.clean_transcript(line)

                        if not cleaned:
                            stats['empty'] += 1
                            continue

                        if len(cleaned.split()) < min_length:
                            stats['too_short'] += 1
                            continue

                        utterances.append(cleaned)
                        stats['cleaned'] += 1

            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")

        logger.info(f"Transcript aggregation stats: {stats}")
        return utterances

    def load_from_json(self, json_path: Path) -> List[str]:
        """
        Load transcripts from JSON export (e.g., from database query)

        Expected format:
        {
            "user_id": "user123",
            "transcripts": [
                {"text": "...", "timestamp": "2025-01-01T00:00:00Z"},
                ...
            ]
        }

        Args:
            json_path: Path to JSON file

        Returns:
            List of cleaned transcript texts
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        utterances = []
        for transcript in data.get('transcripts', []):
            text = transcript.get('text', '')
            cleaned = self.clean_transcript(text)
            if cleaned and len(cleaned.split()) >= 5:
                utterances.append(cleaned)

        logger.info(f"Loaded {len(utterances)} utterances from JSON")
        return utterances


class LanguageModelBuilder:
    """Build ARPA language models using KenLM"""

    def __init__(
        self,
        kenlm_bin: str = 'lmplz',
        order: int = 3,
        prune: str = '0 0 1'
    ):
        """
        Initialize LM builder

        Args:
            kenlm_bin: Path to KenLM lmplz binary
            order: N-gram order (3 = trigram)
            prune: Pruning thresholds for each order (e.g., "0 0 1" = prune singletons for trigrams)
        """
        self.kenlm_bin = kenlm_bin
        self.order = order
        self.prune = prune

        # Verify KenLM is installed
        try:
            subprocess.run(
                [self.kenlm_bin, '--help'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                f"KenLM not found at {kenlm_bin}. "
                "Install with: brew install kenlm (macOS) or build from source"
            )

    def train_arpa(
        self,
        utterances: List[str],
        output_arpa: Path,
        vocab_path: Optional[Path] = None
    ) -> bool:
        """
        Train ARPA language model from utterances

        Args:
            utterances: List of cleaned text utterances
            output_arpa: Output path for ARPA LM file
            vocab_path: Optional path to vocabulary file (for closed vocab)

        Returns:
            True if successful
        """
        logger.info(f"Training {self.order}-gram ARPA LM from {len(utterances)} utterances")

        # Write utterances to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            corpus_file = Path(f.name)
            for utterance in utterances:
                f.write(utterance + '\n')

        try:
            # Build command
            cmd = [
                self.kenlm_bin,
                '-o', str(self.order),
                '--prune', self.prune,
                '--discount_fallback',  # Handle unseen n-grams gracefully
                '--text', str(corpus_file),
            ]

            if vocab_path:
                cmd.extend(['--limit_vocab_file', str(vocab_path)])

            logger.info(f"Running: {' '.join(cmd)}")

            # Run KenLM
            with open(output_arpa, 'w') as f_out:
                result = subprocess.run(
                    cmd,
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )

            logger.info(f"ARPA LM saved to {output_arpa}")

            # Log statistics from stderr
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if any(x in line for x in ['=====', 'unigrams', 'trigrams']):
                        logger.info(f"KenLM: {line}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"KenLM training failed: {e.stderr}")
            return False

        finally:
            # Clean up temporary corpus file
            corpus_file.unlink(missing_ok=True)


class FSTCompiler:
    """Convert ARPA language models to k2 FST format"""

    def __init__(self, vocab_size: int = 1030):
        """
        Initialize FST compiler

        Args:
            vocab_size: Expected vocabulary size (must match acoustic model)
        """
        self.vocab_size = vocab_size

    def arpa_to_fst(
        self,
        arpa_path: Path,
        output_fst: Path,
        token_table: Optional[Dict[str, int]] = None
    ) -> bool:
        """
        Convert ARPA LM to k2 FST format

        Args:
            arpa_path: Path to ARPA LM file
            output_fst: Output path for k2 FST
            token_table: Optional mapping of tokens to IDs

        Returns:
            True if successful
        """
        logger.info(f"Converting ARPA to k2 FST: {arpa_path} -> {output_fst}")

        try:
            # Parse ARPA file
            unigrams, bigrams, trigrams = self._parse_arpa(arpa_path)

            # Build FST
            fsa = self._build_fsa(unigrams, bigrams, trigrams, token_table)

            # Save to disk
            output_fst.parent.mkdir(parents=True, exist_ok=True)
            fsa.save(str(output_fst))

            logger.info(f"FST saved: {output_fst}")
            logger.info(f"  States: {fsa.num_states()}")
            logger.info(f"  Arcs: {fsa.num_arcs()}")

            return True

        except Exception as e:
            logger.error(f"FST compilation failed: {e}")
            return False

    def _parse_arpa(
        self,
        arpa_path: Path
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Parse ARPA file to extract n-grams

        Returns:
            Tuple of (unigrams, bigrams, trigrams) dicts
        """
        unigrams = {}
        bigrams = {}
        trigrams = {}

        current_section = None

        with open(arpa_path, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('\\1-grams:'):
                    current_section = 'unigram'
                    continue
                elif line.startswith('\\2-grams:'):
                    current_section = 'bigram'
                    continue
                elif line.startswith('\\3-grams:'):
                    current_section = 'trigram'
                    continue
                elif line.startswith('\\end\\'):
                    break

                if not line or line.startswith('\\'):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                prob = float(parts[0])
                ngram = tuple(parts[1].split('_') if '_' in parts[1] else [parts[1]])

                if current_section == 'unigram':
                    unigrams[ngram] = prob
                elif current_section == 'bigram' and len(ngram) == 2:
                    bigrams[ngram] = prob
                elif current_section == 'trigram' and len(ngram) == 3:
                    trigrams[ngram] = prob

        logger.info(f"Parsed ARPA: {len(unigrams)} unigrams, {len(bigrams)} bigrams, {len(trigrams)} trigrams")
        return unigrams, bigrams, trigrams

    def _build_fsa(
        self,
        unigrams: Dict,
        bigrams: Dict,
        trigrams: Dict,
        token_table: Optional[Dict[str, int]]
    ) -> k2.Fsa:
        """
        Build k2 FSA from n-gram probabilities

        This creates a simple backoff LM structure.
        For production, consider using k2's built-in LM classes.
        """
        # Create a simple linear FST from unigrams for now
        # TODO: Implement proper backoff structure with trigrams/bigrams

        # Get tokens
        tokens = sorted(unigrams.keys(), key=lambda x: unigrams[x], reverse=True)

        # Build simple FST (all tokens from start state)
        arcs = []
        state = 0  # Start state

        for token_tuple in tokens:
            token = token_tuple[0] if isinstance(token_tuple, tuple) else token_tuple

            # Map token to ID (if table provided, otherwise use hash)
            if token_table:
                label = token_table.get(token, 0)  # 0 = <unk>
            else:
                label = hash(token) % self.vocab_size

            # Score (convert log10 prob to ln)
            score = unigrams[token_tuple] * 2.302585  # ln(10)

            # Arc: (src, dest, label, score)
            arcs.append([state, state, label, score])

        # Add final state arc
        arcs.append([state, state + 1, -1, 0.0])  # -1 = final

        # Convert to k2 FSA
        arcs_tensor = torch.tensor(arcs, dtype=torch.float32)
        fsa = k2.Fsa.from_tensor(arcs_tensor)

        # Arc-sort for efficiency
        fsa = k2.arc_sort(fsa)

        return fsa


class UserLMBuilder:
    """High-level interface for building user language models"""

    def __init__(
        self,
        output_dir: Path,
        order: int = 3,
        prune: str = '0 0 1',
        vocab_size: int = 1030,
        min_utterances: int = 50
    ):
        """
        Initialize user LM builder

        Args:
            output_dir: Root directory for user FSTs
            order: N-gram order
            prune: KenLM pruning thresholds
            vocab_size: Acoustic model vocab size
            min_utterances: Minimum utterances required to build LM
        """
        self.output_dir = Path(output_dir)
        self.min_utterances = min_utterances

        self.processor = TranscriptProcessor()
        self.lm_builder = LanguageModelBuilder(order=order, prune=prune)
        self.fst_compiler = FSTCompiler(vocab_size=vocab_size)

    def build_for_user(
        self,
        user_id: str,
        utterances: List[str],
        force: bool = False
    ) -> bool:
        """
        Build complete pipeline for a single user

        Args:
            user_id: User identifier
            utterances: List of cleaned transcript utterances
            force: Overwrite existing FST if present

        Returns:
            True if successful
        """
        logger.info(f"Building LM for user_id={user_id} ({len(utterances)} utterances)")

        # Check minimum utterances
        if len(utterances) < self.min_utterances:
            logger.warning(
                f"User {user_id} has only {len(utterances)} utterances "
                f"(min: {self.min_utterances}), skipping"
            )
            return False

        # Setup output paths
        user_dir = self.output_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        arpa_path = user_dir / 'lm.arpa'
        fst_path = user_dir / 'G.fst'

        # Check if already exists
        if fst_path.exists() and not force:
            logger.info(f"FST already exists for {user_id}, skipping (use --force to overwrite)")
            return True

        # Step 1: Train ARPA LM
        success = self.lm_builder.train_arpa(utterances, arpa_path)
        if not success:
            return False

        # Step 2: Convert to FST
        success = self.fst_compiler.arpa_to_fst(arpa_path, fst_path)
        if not success:
            return False

        # Step 3: Save metadata
        metadata = {
            'user_id': user_id,
            'utterances_count': len(utterances),
            'order': self.lm_builder.order,
            'prune': self.lm_builder.prune,
            'vocab_size': self.fst_compiler.vocab_size,
        }

        with open(user_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Successfully built LM for {user_id}")
        return True

    def build_batch(
        self,
        transcripts_root: Path,
        force: bool = False,
        max_users: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Build LMs for all users in transcript directory

        Expected structure:
        transcripts_root/
            user001/
                transcript_001.txt
                transcript_002.txt
            user002/
                ...

        Args:
            transcripts_root: Root directory containing user subdirectories
            force: Overwrite existing FSTs
            max_users: Maximum number of users to process (for testing)

        Returns:
            Dict mapping user_id to success status
        """
        results = {}
        user_dirs = sorted([d for d in transcripts_root.iterdir() if d.is_dir()])

        if max_users:
            user_dirs = user_dirs[:max_users]

        logger.info(f"Processing {len(user_dirs)} users from {transcripts_root}")

        for user_dir in user_dirs:
            user_id = user_dir.name

            # Find all transcript files
            transcript_files = list(user_dir.glob('*.txt')) + list(user_dir.glob('*.json'))

            if not transcript_files:
                logger.warning(f"No transcript files found for {user_id}")
                results[user_id] = False
                continue

            # Aggregate transcripts
            utterances = self.processor.aggregate_transcripts(transcript_files)

            # Build LM
            success = self.build_for_user(user_id, utterances, force=force)
            results[user_id] = success

        # Summary
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Batch build complete: {successful}/{len(results)} users successful")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Build user language models from historical transcripts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--user-id', help='Single user ID to process')
    input_group.add_argument('--batch', action='store_true',
                             help='Process all users in transcripts root')

    parser.add_argument('--transcripts-dir', type=Path,
                        help='Directory containing transcript files for single user')
    parser.add_argument('--transcripts-root', type=Path,
                        help='Root directory containing user subdirectories (for batch)')
    parser.add_argument('--from-json', type=Path,
                        help='Load transcripts from JSON export')

    # Output options
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for user FSTs')

    # LM training options
    parser.add_argument('--order', type=int, default=3,
                        help='N-gram order (default: 3)')
    parser.add_argument('--prune', default='0 0 1',
                        help='KenLM pruning thresholds (default: "0 0 1")')
    parser.add_argument('--vocab-size', type=int, default=1030,
                        help='Acoustic model vocabulary size (default: 1030)')
    parser.add_argument('--min-utterances', type=int, default=50,
                        help='Minimum utterances required (default: 50)')

    # Misc options
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing FSTs')
    parser.add_argument('--max-users', type=int,
                        help='Maximum users to process in batch mode (for testing)')
    parser.add_argument('--kenlm-bin', default='lmplz',
                        help='Path to KenLM lmplz binary (default: lmplz)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if args.user_id and not (args.transcripts_dir or args.from_json):
        parser.error("--user-id requires --transcripts-dir or --from-json")

    if args.batch and not args.transcripts_root:
        parser.error("--batch requires --transcripts-root")

    # Initialize builder
    builder = UserLMBuilder(
        output_dir=args.output_dir,
        order=args.order,
        prune=args.prune,
        vocab_size=args.vocab_size,
        min_utterances=args.min_utterances
    )

    # Override KenLM path if specified
    builder.lm_builder.kenlm_bin = args.kenlm_bin

    try:
        if args.batch:
            # Batch processing
            results = builder.build_batch(
                transcripts_root=args.transcripts_root,
                force=args.force,
                max_users=args.max_users
            )

            # Exit with error if any failed
            if not all(results.values()):
                sys.exit(1)

        else:
            # Single user processing
            if args.from_json:
                utterances = builder.processor.load_from_json(args.from_json)
            else:
                transcript_files = list(args.transcripts_dir.glob('*.txt'))
                utterances = builder.processor.aggregate_transcripts(transcript_files)

            success = builder.build_for_user(args.user_id, utterances, force=args.force)

            if not success:
                sys.exit(1)

        logger.info("✅ All done!")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
