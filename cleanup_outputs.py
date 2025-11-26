from __future__ import annotations

from pathlib import Path


def clean_dir(dir_path: Path) -> int:
    """
    Löscht alle Dateien direkt in dir_path (keine Unterordner).
    Gibt die Anzahl gelöschter Dateien zurück.
    """
    if not dir_path.exists():
        print(f"ℹ️ Ordner existiert nicht (übersprungen): {dir_path}")
        return 0

    deleted = 0
    for item in dir_path.iterdir():
        if item.is_file():
            item.unlink()
            deleted += 1
    print(f"🧹 {deleted} Dateien gelöscht in: {dir_path}")
    return deleted


def main() -> None:
    # Projekt-Root = Ordner, in dem dieses Skript liegt
    base = Path(__file__).resolve().parent

    # Zielordner, die geleert werden sollen
    target_dirs = [
        base / "outputs" / "pivots" / "3D",
        base / "outputs" / "pivots" / "W",
        base / "outputs" / "wickdiffs" / "3D→H1",
        base / "outputs" / "wickdiffs" / "W→H4",
    ]

    print("🚀 Starte Cleanup der Output-Ordner...\n")
    total = 0
    for d in target_dirs:
        total += clean_dir(d)

    print(f"\n✅ Cleanup fertig. Insgesamt gelöschte Dateien: {total}")
    print("ℹ️ 'time frame data' und alle Rohdaten wurden NICHT angerührt.")


if __name__ == "__main__":
    main()
