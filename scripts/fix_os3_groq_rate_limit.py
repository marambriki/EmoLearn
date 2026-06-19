"""Ajoute retry Groq 429 et pauses dans OS3."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "Livrable" / "OS3.ipynb"

RETRY_HELPER = '''
import time

def groq_create_with_retry(**kwargs):
    """Relance automatiquement si limite Groq (429 TPM)."""
    for attempt in range(6):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "rate_limit" in msg:
                wait = 10 + attempt * 5
                print(f"⏳ Limite Groq atteinte — pause {wait}s puis nouvel essai...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Limite Groq toujours atteinte après plusieurs essais.")
'''

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# Cellule connexion Groq
src0 = "".join(nb["cells"][1]["source"])
if "groq_create_with_retry" not in src0:
    src0 = src0.replace(
        "from groq import Groq\n",
        "from groq import Groq\nimport time\n",
    )
    src0 = src0.rstrip() + "\n" + RETRY_HELPER.strip() + "\n"
    nb["cells"][1]["source"] = [line + "\n" for line in src0.splitlines()]
    nb["cells"][1]["source"][-1] = nb["cells"][1]["source"][-1].rstrip("\n")

# Cellules templates + universel + corpus
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell["source"])
    src = src.replace(
        "client.chat.completions.create(",
        "groq_create_with_retry(",
    )
    src = src.replace("time.sleep(0.5)", "time.sleep(3)")
    cell["source"] = [line + "\n" for line in src.splitlines()]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

# Note markdown
intro = "".join(nb["cells"][0]["source"])
if "6000 tokens/min" not in intro:
    intro = intro.rstrip() + (
        "\n\n**Limite Groq (gratuit) :** ~6000 tokens/minute. "
        "En cas d'erreur 429, attendez 10 s ou relancez la cellule. "
        "La génération du corpus 200+ est longue — exécutez-la seule en dernier."
    )
    nb["cells"][0]["source"] = [line + "\n" for line in intro.splitlines()]
    nb["cells"][0]["source"][-1] = nb["cells"][0]["source"][-1].rstrip("\n")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print("OS3.ipynb mis à jour")
