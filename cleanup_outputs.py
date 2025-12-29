from __future__ import annotations

from pathlib import Path


def clean_dir_recursive(dir_path: Path) -> int:
    """
    Löscht rekursiv alle *nicht versteckten* Dateien unterhalb von dir_path.
    Ordner bleiben bestehen.
    Gibt die Anzahl gelöschter Dateien zurück.
    """
    if not dir_path.exists():
        print(f"ℹ️ Ordner existiert nicht (übersprungen): {dir_path}")
        return 0

    deleted = 0
    for item in dir_path.rglob("*"):
        if item.is_file() and not item.name.startswith("."):
            item.unlink()
            deleted += 1

    print(f"🧹 {deleted} Dateien gelöscht unter: {dir_path}")
    return deleted


def main() -> None:
    # Projekt-Root = Ordner, in dem dieses Skript liegt
    base = Path(__file__).resolve().parent

    # Nur der Output-Baum wird geleert
    outputs_root = base / "outputs"

    print("🚀 Starte kompletten Cleanup aller Output-Dateien...\n")

    total = clean_dir_recursive(outputs_root)

    print(f"\n✅ Cleanup fertig. Insgesamt gelöschte Dateien in 'outputs': {total}")
    print("ℹ️ 'cot data', 'time frame data' und alle Rohdaten wurden NICHT angerührt.")


if __name__ == "__main__":
    main()
