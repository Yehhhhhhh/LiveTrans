import sys
from pathlib import Path

# The backend modules import each other by bare name (server.py runs from backend/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
