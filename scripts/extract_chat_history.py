"""Extract user messages from Cursor agent transcripts into a markdown file."""
import json
import re
from pathlib import Path

BASE = Path(r"C:\Users\Lenovo\.cursor\projects\c-Users-Lenovo-EmoLearn\agent-transcripts")
OUT = Path(__file__).resolve().parents[1] / "docs" / "HISTORIQUE_CONVERSATIONS.md"


def parse_user_line(line: str) -> tuple[str, str] | None:
    try:
        o = json.loads(line)
    except json.JSONDecodeError:
        return None
    if o.get("role") != "user":
        return None
    content = o.get("message", {}).get("content", [])
    if not isinstance(content, list):
        return None
    text = next((x.get("text", "") for x in content if x.get("type") == "text"), "")
    ts_m = re.search(r"<timestamp>(.*?)</timestamp>", text)
    q_m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", text, re.S)
    ts = ts_m.group(1).strip() if ts_m else ""
    q = (q_m.group(1) if q_m else text).strip()
    q = re.sub(r"\[Image\].*", "", q, flags=re.S).strip()
    if "Briefly inform the user about the task result" in q:
        return None
    if not q or q.startswith("<"):
        return None
    return ts, q


def load_transcript(path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_user_line(line)
            if parsed:
                items.append(parsed)
    return items


def main() -> None:
    transcripts: list[tuple[str, list[tuple[str, str]]]] = []
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        f = d / f"{d.name}.jsonl"
        if not f.exists():
            continue
        users = load_transcript(f)
        if users:
            transcripts.append((d.name, users))

    transcripts.sort(key=lambda x: x[1][0][0] if x[1][0][0] else "")

    lines = [
        "# EmoLearn — Historique complet des conversations Cursor",
        "",
        "> Document reconstruit automatiquement à partir des fichiers `agent-transcripts` sur disque.",
        "> Date de génération : août 2026. **Toutes vos conversations sont sauvegardées ici.**",
        "",
        "---",
        "",
        "## Index rapide (conversations principales)",
        "",
        "| ID | Période | Sujet | Messages |",
        "|---|---|---|:---:|",
    ]

    key_ids = {
        "25180616": "Fiche encadrante, WavLM, AU, développement (3–19 août)",
        "a7f72e19": "Fiche, métriques, séquences, oral encadrante (18–20 août)",
        "4137c188": "Explication projet A→Z (4 août)",
        "723e8bcb": "Chapitre 2 état de l'art (13 août)",
        "53e52839": "Chapitre 3 mémoire (14 août)",
        "7f1736fb": "Simulation questions encadrante 60/40 (16 août)",
        "8ff6d5d4": "Scénarios boredom vs engagement (6 août)",
        "8c3dbde8": "Schéma architecture (6 août)",
        "5ea9e29b": "Reprise historique (21 août)",
    }

    for tid, users in transcripts:
        short = tid[:8]
        if short not in key_ids and len(users) < 5:
            continue
        start = users[0][0][:20] if users[0][0] else "?"
        end = users[-1][0][:20] if users[-1][0] else "?"
        topic = key_ids.get(short, users[0][1][:60].replace("|", "/"))
        lines.append(f"| `{short}` | {start} → {end} | {topic} | {len(users)} |")

    lines.extend(["", "---", ""])

    for tid, users in transcripts:
        short = tid[:8]
        lines.append(f"## Conversation `{short}`")
        if users[0][0]:
            lines.append(f"**Début :** {users[0][0]}")
        if users[-1][0]:
            lines.append(f"**Fin :** {users[-1][0]}")
        lines.append(f"**Messages utilisateur :** {len(users)}")
        lines.append("")
        for i, (ts, q) in enumerate(users, 1):
            q_one = q.replace("\n", " ").strip()
            if len(q_one) > 300:
                q_one = q_one[:297] + "..."
            ts_part = f" ({ts})" if ts else ""
            lines.append(f"{i}. **{q_one}**{ts_part}")
        lines.extend(["", "---", ""])

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {OUT} ({len(transcripts)} conversations, {sum(len(u) for _, u in transcripts)} messages)")


if __name__ == "__main__":
    main()
