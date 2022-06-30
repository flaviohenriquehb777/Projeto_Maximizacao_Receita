import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.util import ensure_async


def run_notebook(input_path: Path, output_path: Path, kernel_name: str = "python3", timeout: int = 600):
    # Garante PYTHONPATH com raiz do projeto para import de src/
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PYTHONPATH", str(project_root))

    nb = nbformat.read(str(input_path), as_version=4)
    client = NotebookClient(nb, kernel_name=kernel_name, timeout=timeout)
    try:
        client.execute()
    except Exception as e:
        # Imprime um resumo amigável do erro da última célula
        print("[ERROR] Falha ao executar notebook:", input_path)
        # Tenta localizar a última célula executada para contexto
        for i, cell in enumerate(nb.cells):
            exec_count = cell.get('execution_count')
            if exec_count is not None:
                last_idx = i
        if 'last_idx' in locals():
            print(f"[ERROR] Última célula executada índice={last_idx} tipo={nb.cells[last_idx].cell_type}")
            print("[ERROR] Conteúdo da célula:")
            print(nb.cells[last_idx].source)
        # Mensagem e traceback
        import traceback
        print(f"[ERROR] Tipo: {type(e).__name__}")
        print(f"[ERROR] Mensagem: {e}")
        traceback.print_exc()
        raise SystemExit(1) from e
    else:
        nbformat.write(nb, str(output_path))
        print(f"[OK] Notebook executado e salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Executa um notebook Jupyter e salva a saída.")
    parser.add_argument("input", type=str, help="Caminho do notebook de entrada (.ipynb)")
    parser.add_argument("--output", type=str, default=None, help="Caminho de saída (.ipynb). Se omitido, usa out_<nome>.ipynb")
    parser.add_argument("--kernel", type=str, default="python3", help="Nome do kernel Jupyter")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout por célula (segundos)")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"[ERROR] Arquivo não encontrado: {in_path}")
        sys.exit(1)
    out_path = Path(args.output) if args.output else in_path.with_name(f"out_{in_path.stem}.ipynb")

    run_notebook(in_path, out_path, kernel_name=args.kernel, timeout=args.timeout)


if __name__ == "__main__":
    main()