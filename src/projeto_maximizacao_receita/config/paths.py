from pathlib import Path


def get_project_root() -> Path:
    """Retorna a raiz do projeto baseado em marcadores."""
    current = Path(__file__).resolve()
    while not any((current / marker).exists() for marker in [".git", "src", "dados"]):
        if current.parent == current:
            raise RuntimeError("Raiz do projeto não encontrada.")
        current = current.parent
    return current


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "dados"

# Nome claro e neutro para o caminho do dataset de vendas
DADOS_VENDA_PATH = DATA_DIR / "dadosVenda.xlsx"

# Backward-compat para código/teses legadas
DADOS_AMOR_A_CAKES = DADOS_VENDA_PATH

# Não falhar no import; o código de carga trata ausência do arquivo.

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)  # Cria a pasta se não existir

# REPORTS_DIR = PROJECT_ROOT / "reports"
# REPORTS_DIR.mkdir(exist_ok=True)
