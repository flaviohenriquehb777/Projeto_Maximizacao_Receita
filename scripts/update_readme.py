import json
import os
import re
from os import path as p

START = "<!-- MODEL_RESULTS_START -->"
END = "<!-- MODEL_RESULTS_END -->"


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_block():
    train = load_json("artifacts/metrics.json")
    best = load_json("artifacts/metrics_best.json")

    lines = [START, "", "### Resumo automático do último run", ""]
    # Treino
    lines.append("- Treino")
    lines.append(f"  - RMSE: {train.get('rmse', 'N/A')}")
    lines.append(f"  - R2: {train.get('r2', 'N/A')}")
    # Melhor Modelo
    lines.append("- Melhor Modelo")
    lines.append(f"  - Modelo: {best.get('model', 'N/A')}")
    lines.append(f"  - RMSE: {best.get('rmse', 'N/A')}")
    lines.append(f"  - R2: {best.get('r2', 'N/A')}")

    # Link para o run
    run_id = os.environ.get("GITHUB_RUN_ID")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    if run_id and repo:
        lines.append("")
        lines.append(f"Fonte: [GitHub Actions]({server}/{repo}/actions/runs/{run_id})")

    lines.append("")
    lines.append(END)
    return "\n".join(lines)


def update_readme(path="README.md"):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    block = build_block()

    if START in content and END in content:
        pattern = re.compile(re.escape(START) + ".*?" + re.escape(END), re.DOTALL)
        new_content = pattern.sub(block, content)
    else:
        # Inserir logo após o título da seção de resultados, se existir
        header = "## Resultados Chave e Recomendações"
        idx = content.find(header)
        if idx != -1:
            insert_pos = content.find("\n", idx) + 1
            new_content = content[:insert_pos] + block + "\n" + content[insert_pos:]
        else:
            # Como fallback, adicionar ao final
            new_content = content + "\n\n" + block + "\n"

    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


if __name__ == "__main__":
    changed = update_readme()
    print("README updated" if changed else "README unchanged")