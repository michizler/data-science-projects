"""Quick CLI demo. Classify a single review.

Usage:
    python demo.py "Great service, will keep subscribing."
"""
import sys
from classifier import analyse


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python demo.py "<review text>"', file=sys.stderr)
        return 1

    review = " ".join(sys.argv[1:])
    result = analyse(review)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
