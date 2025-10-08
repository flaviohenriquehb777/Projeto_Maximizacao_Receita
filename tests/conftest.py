import sys
from pathlib import Path

# Garante que 'src' possa ser importado nos testes
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
