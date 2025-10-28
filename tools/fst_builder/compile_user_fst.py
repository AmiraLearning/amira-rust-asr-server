#!/usr/bin/env python3
"""
Compile per-user language models into k2 FST format.

This script takes user-specific text data and creates an ARPA n-gram
language model, then converts it to k2 FST format for GPU decoding.

Dependencies:
    - KenLM (for ARPA LM estimation)
    - k2 Python bindings (for FST creation)
    - Pynini (alternative FST toolkit)

Usage:
    # Single user
    python compile_user_fst.py --user-id user123 --text-file user123_data.txt --output-dir /models/user_fsts

    # Batch compilation
    python compile_user_fst.py --batch --users-dir /data/users --output-dir /models/user_fsts

    # With custom LM parameters
    python compile_user_fst.py --user-id user123 --text-file data.txt --order 3 --prune 1e-8
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, List

try:
    import k2
    import torch
except ImportError:
    print("ERROR: k2 and torch are required. Install with: pip install k2 torch", file=sys.stderr)
    sys.exit(1)


class FSTCompiler:
    """Compiles text data into k2 FST format for personalized ASR."""

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        lm_order: int = 3,
        prune_threshold: float = 1e-7,
        kenlm_binary_path: str = "lmplz",
    ):
        """
        Initialize FST compiler.

        Args:
            vocab_file: Path to vocabulary file (optional - will use text if not provided)
            lm_order: N-gram order for language model (default: 3)
            prune_threshold: Pruning threshold for ARPA LM (default: 1e-7)
            kenlm_binary_path: Path to KenLM lmplz binary
        """
        self.vocab_file = vocab_file
        self.lm_order = lm_order
        self.prune_threshold = prune_threshold
        self.kenlm_binary = kenlm_binary_path

        # Check if KenLM is available
        self._check_kenlm()

    def _check_kenlm(self):
        """Check if KenLM is installed and accessible."""
        try:
            result = subprocess.run(
                [self.kenlm_binary, "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 and "lmplz" not in result.stderr:
                raise FileNotFoundError
        except FileNotFoundError:
            print(
                f"ERROR: KenLM '{self.kenlm_binary}' not found. "
                "Install from https://github.com/kpu/kenlm",
                file=sys.stderr,
            )
            sys.exit(1)

    def train_arpa_lm(self, text_file: str, output_arpa: str) -> None:
        """
        Train ARPA language model from text file using KenLM.

        Args:
            text_file: Path to input text file (one sentence per line)
            output_arpa: Path to output ARPA file
        """
        print(f"Training {self.lm_order}-gram ARPA LM from {text_file}...")

        cmd = [
            self.kenlm_binary,
            "-o", str(self.lm_order),
            "--prune", str(self.prune_threshold),
            "--discount_fallback",
        ]

        # Optional: limit memory usage
        # cmd.extend(["-S", "2G"])  # 2GB memory limit

        with open(text_file, "r", encoding="utf-8") as infile:
            with open(output_arpa, "w", encoding="utf-8") as outfile:
                result = subprocess.run(
                    cmd,
                    stdin=infile,
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if result.returncode != 0:
                    print(f"ERROR: KenLM failed: {result.stderr}", file=sys.stderr)
                    sys.exit(1)

        print(f"ARPA LM saved to {output_arpa}")

    def arpa_to_fst(self, arpa_file: str, output_fst: str, vocab_size: int = 1030) -> None:
        """
        Convert ARPA language model to k2 FST format.

        Args:
            arpa_file: Path to ARPA file
            output_fst: Path to output FST file
            vocab_size: Vocabulary size (default: 1030 for standard model)
        """
        print(f"Converting ARPA to k2 FST...")

        try:
            # Read ARPA file and create FST
            # This is a simplified version - in production, use k2's ARPA parser
            with open(arpa_file, "r", encoding="utf-8") as f:
                arpa_content = f.read()

            # Parse ARPA format (simplified - production should use proper parser)
            # For now, create a simple FST with unigram probabilities
            # TODO: Implement full n-gram FST construction

            # Create a simple linear FST as placeholder
            # In production, this should be a proper language model FST
            # with backoff states and n-gram arcs

            # Dummy implementation - replace with actual k2 ARPA parsing
            print("WARNING: Using simplified FST creation. Implement full ARPA parser for production.")

            # Create FSA with k2
            # Format: "src_state dest_state label score"
            fsa_str_parts = []
            fsa_str_parts.append("0 1 0 0.0")  # Start state with epsilon
            for i in range(1, min(vocab_size, 100)):  # Simplified
                score = -1.0  # Dummy scores
                fsa_str_parts.append(f"1 1 {i} {score}")
            fsa_str_parts.append("1")  # Final state

            fsa_str = "\n".join(fsa_str_parts)

            # Create FSA from string
            fsa = k2.Fsa.from_str(fsa_str)
            fsa = k2.arc_sort(fsa)

            # Save to file
            fsa.save(output_fst)
            print(f"FST saved to {output_fst}")

        except Exception as e:
            print(f"ERROR: Failed to convert ARPA to FST: {e}", file=sys.stderr)
            sys.exit(1)

    def compile_user_fst(
        self,
        user_id: str,
        text_file: str,
        output_dir: str,
        keep_arpa: bool = False,
    ) -> str:
        """
        Compile a user's text data into FST format.

        Args:
            user_id: User identifier
            text_file: Path to user's text data
            output_dir: Output directory for FST files
            keep_arpa: Whether to keep intermediate ARPA file

        Returns:
            Path to compiled FST file
        """
        print(f"\n{'='*60}")
        print(f"Compiling FST for user: {user_id}")
        print(f"{'='*60}")

        # Create user output directory
        user_dir = Path(output_dir) / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        arpa_file = user_dir / "G.arpa"
        fst_file = user_dir / "G.fst"

        # Step 1: Train ARPA LM
        self.train_arpa_lm(text_file, str(arpa_file))

        # Step 2: Convert to FST
        self.arpa_to_fst(str(arpa_file), str(fst_file))

        # Cleanup
        if not keep_arpa:
            arpa_file.unlink()

        print(f"\nFST compilation complete for {user_id}")
        print(f"Output: {fst_file}")
        return str(fst_file)

    def compile_batch(
        self,
        users_dir: str,
        output_dir: str,
        text_file_pattern: str = "*.txt",
    ) -> List[str]:
        """
        Compile FSTs for multiple users in batch.

        Args:
            users_dir: Directory containing user text files
            output_dir: Output directory for FST files
            text_file_pattern: Glob pattern for text files

        Returns:
            List of compiled FST paths
        """
        users_path = Path(users_dir)
        compiled_fsts = []

        # Find all user text files
        text_files = sorted(users_path.glob(text_file_pattern))

        if not text_files:
            print(f"WARNING: No text files found in {users_dir}", file=sys.stderr)
            return compiled_fsts

        print(f"Found {len(text_files)} users to compile")

        for text_file in text_files:
            # Extract user ID from filename (e.g., user123.txt -> user123)
            user_id = text_file.stem

            try:
                fst_path = self.compile_user_fst(
                    user_id=user_id,
                    text_file=str(text_file),
                    output_dir=output_dir,
                    keep_arpa=False,
                )
                compiled_fsts.append(fst_path)
            except Exception as e:
                print(f"ERROR compiling FST for {user_id}: {e}", file=sys.stderr)
                continue

        print(f"\nBatch compilation complete: {len(compiled_fsts)}/{len(text_files)} successful")
        return compiled_fsts


def validate_text_file(text_file: str) -> bool:
    """Validate that text file exists and is not empty."""
    path = Path(text_file)
    if not path.exists():
        print(f"ERROR: Text file not found: {text_file}", file=sys.stderr)
        return False
    if path.stat().st_size == 0:
        print(f"ERROR: Text file is empty: {text_file}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compile per-user language models into k2 FST format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single user FST
    python compile_user_fst.py --user-id user123 --text-file user_data.txt --output-dir /models/fsts

    # Batch compilation
    python compile_user_fst.py --batch --users-dir /data/users --output-dir /models/fsts

    # With custom parameters
    python compile_user_fst.py --user-id user123 --text-file data.txt --order 4 --prune 1e-9
        """,
    )

    parser.add_argument(
        "--user-id",
        type=str,
        help="User identifier (required for single user mode)",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        help="Path to user text data (one sentence per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for FST files",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode for multiple users",
    )
    parser.add_argument(
        "--users-dir",
        type=str,
        help="Directory containing user text files (for batch mode)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="N-gram order for language model (default: 3)",
    )
    parser.add_argument(
        "--prune",
        type=float,
        default=1e-7,
        help="Pruning threshold for ARPA LM (default: 1e-7)",
    )
    parser.add_argument(
        "--keep-arpa",
        action="store_true",
        help="Keep intermediate ARPA files",
    )
    parser.add_argument(
        "--kenlm-path",
        type=str,
        default="lmplz",
        help="Path to KenLM lmplz binary (default: lmplz)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.batch:
        if not args.users_dir:
            parser.error("--users-dir is required for batch mode")
    else:
        if not args.user_id or not args.text_file:
            parser.error("--user-id and --text-file are required for single user mode")
        if not validate_text_file(args.text_file):
            sys.exit(1)

    # Initialize compiler
    compiler = FSTCompiler(
        lm_order=args.order,
        prune_threshold=args.prune,
        kenlm_binary_path=args.kenlm_path,
    )

    # Compile FSTs
    if args.batch:
        compiler.compile_batch(
            users_dir=args.users_dir,
            output_dir=args.output_dir,
        )
    else:
        compiler.compile_user_fst(
            user_id=args.user_id,
            text_file=args.text_file,
            output_dir=args.output_dir,
            keep_arpa=args.keep_arpa,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
