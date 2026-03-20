"""Helper script to inject JupyterLite launch buttons into notebooks."""

import json
from pathlib import Path


def inject_buttons() -> None:
    """Inject JupyterLite launch buttons into all notebooks in docs/notebooks/."""
    notebooks_dir = Path("docs/notebooks")
    for p in notebooks_dir.glob("*.ipynb"):
        if "hfss" in p.name:
            continue
        try:
            with p.open("r") as f:
                d = json.load(f)

            # Check if already injected
            if any(
                "jupyterlite-launch" in cell.get("id", "")
                for cell in d.get("cells", [])
            ):
                continue

            rel = f"../../notebooks/{p.name}"
            btn = {
                "cell_type": "markdown",
                "id": "jupyterlite-launch",
                "metadata": {},
                "source": [
                    f"```{{notebooklite}} {rel}\n",
                    ":new_tab: True\n",
                    "```",
                ],
            }
            # Insert after the title cell (index 0)
            d["cells"].insert(1, btn)

            with p.open("w") as f:
                json.dump(d, f, indent=1)
            print(f"Injected button into {p.name}")
        except Exception as e:
            print(f"Failed to inject into {p.name}: {e}")


if __name__ == "__main__":
    inject_buttons()
