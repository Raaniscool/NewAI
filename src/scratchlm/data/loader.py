"""
Data loading utilities for ScratchLM.

Provides functions for loading text data from files and directories.
"""

import json
from typing import List, Optional, Union, Iterator
from pathlib import Path
import os

from scratchlm.utils.paths import paths


def load_text_file(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    max_lines: Optional[int] = None,
) -> List[str]:
    """
    Load text from a single file.
    
    Args:
        file_path: Path to the text file
        encoding: File encoding
        max_lines: Maximum number of lines to read (None for all)
        
    Returns:
        List of lines (without newlines)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    texts = []
    with open(file_path, 'r', encoding=encoding) as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            # Remove trailing newline
            line = line.rstrip('\n').rstrip('\r')
            if line.strip():  # Skip empty lines
                texts.append(line)
    
    return texts


def load_text_directory(
    directory: Union[str, Path],
    encoding: str = "utf-8",
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
    max_files: Optional[int] = None,
) -> List[str]:
    """
    Load text from all files in a directory.
    
    Args:
        directory: Path to the directory
        encoding: File encoding
        extensions: List of file extensions to include (None for all)
        recursive: Whether to search subdirectories
        max_files: Maximum number of files to read (None for all)
        
    Returns:
        List of all lines from all files
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    all_texts = []
    file_count = 0
    
    # Find all files
    if recursive:
        files = list(directory.rglob("*"))
    else:
        files = list(directory.glob("*"))
    
    # Filter by extension
    if extensions is not None:
        files = [f for f in files if f.is_file() and f.suffix.lower() in extensions]
    else:
        files = [f for f in files if f.is_file()]
    
    # Limit files
    if max_files is not None:
        files = files[:max_files]
    
    # Load each file
    for file_path in files:
        try:
            file_texts = load_text_file(file_path, encoding=encoding)
            all_texts.extend(file_texts)
            file_count += 1
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return all_texts


def load_json_lines(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    max_lines: Optional[int] = None,
) -> List[dict]:
    """
    Load JSON lines format file.
    
    Each line is a JSON object.
    
    Args:
        file_path: Path to the JSON lines file
        encoding: File encoding
        max_lines: Maximum number of lines to read
        
    Returns:
        List of dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding=encoding) as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip invalid JSON
                    continue
    
    return data


def load_json_file(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
) -> dict:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        encoding: File encoding
        
    Returns:
        Dictionary with the JSON data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding=encoding) as f:
        return json.load(f)


def save_processed_text(
    texts: List[str],
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    one_per_line: bool = True,
) -> Path:
    """
    Save processed texts to a file.
    
    Args:
        texts: List of text strings to save
        file_path: Path to save to
        encoding: File encoding
        one_per_line: Whether to save one text per line
        
    Returns:
        Path to the saved file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        if one_per_line:
            for text in texts:
                f.write(text + '\n')
        else:
            # Save as JSON
            json.dump(texts, f, indent=2)
    
    return file_path


def iterate_text_files(
    directory: Union[str, Path],
    encoding: str = "utf-8",
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
) -> Iterator[str]:
    """
    Iterate through all text files in a directory.
    
    This is a generator that yields one line at a time from all files.
    Useful for processing large datasets without loading everything into memory.
    
    Args:
        directory: Path to the directory
        encoding: File encoding
        extensions: List of file extensions to include
        recursive: Whether to search subdirectories
        
    Yields:
        One line at a time from all files
    """
    directory = Path(directory)
    
    # Find all files
    if recursive:
        files = list(directory.rglob("*"))
    else:
        files = list(directory.glob("*"))
    
    # Filter by extension
    if extensions is not None:
        files = [f for f in files if f.is_file() and f.suffix.lower() in extensions]
    else:
        files = [f for f in files if f.is_file()]
    
    # Iterate through files
    for file_path in files:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.rstrip('\n').rstrip('\r')
                    if line.strip():
                        yield line
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")


def get_file_stats(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
) -> dict:
    """
    Get statistics about a text file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        Dictionary with file statistics
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()
    
    lines = content.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    
    return {
        'path': str(file_path),
        'size_bytes': os.path.getsize(file_path),
        'num_lines': len(lines),
        'num_non_empty_lines': len(non_empty_lines),
        'total_chars': len(content),
        'avg_line_length': sum(len(l) for l in lines) / len(lines) if lines else 0,
    }


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    
    # Create a temporary test file
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test file
        test_file = tmpdir / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n")
        
        # Test load_text_file
        print("\n=== load_text_file ===")
        texts = load_text_file(test_file)
        print(f"Loaded {len(texts)} lines: {texts}")
        
        # Test load_text_directory
        print("\n=== load_text_directory ===")
        texts = load_text_directory(tmpdir)
        print(f"Loaded {len(texts)} lines from directory: {texts}")
        
        # Test save_processed_text
        print("\n=== save_processed_text ===")
        output_file = tmpdir / "output.txt"
        save_processed_text(texts, output_file)
        print(f"Saved to {output_file}")
        
        # Test iterate_text_files
        print("\n=== iterate_text_files ===")
        for i, line in enumerate(iterate_text_files(tmpdir)):
            print(f"Line {i}: {line}")
            if i >= 2:
                break
