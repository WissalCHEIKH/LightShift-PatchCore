import os
import sys
from pathlib import Path

os.environ.setdefault("WARM_UP", "0")   # pas de préchauffage en arrière-plan pendant les tests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
