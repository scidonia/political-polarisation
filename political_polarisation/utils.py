import sys


def eprintln(*args, **kwargs):
    """Print to stderr with a newline, similar to Rust's eprintln! macro."""
    print(*args, file=sys.stderr, **kwargs)
