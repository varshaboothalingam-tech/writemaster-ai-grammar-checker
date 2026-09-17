import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("AI_PROVIDER", "none")
# keep mocked-AI runs from polluting datasets/missed_errors.jsonl (tests opt in
# explicitly and point the path at a temp file when they need to assert logging).
os.environ.setdefault("COLLECT_MISSED_ERRORS", "0")