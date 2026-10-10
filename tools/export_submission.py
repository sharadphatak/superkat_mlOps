"""Export an actually executed Colab notebook to portable HTML; reject errors."""
import argparse
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter


def export(notebook, output):
    nb = nbformat.read(notebook, as_version=4)
    unexecuted = [i + 1 for i, c in enumerate(nb.cells)
                  if c.cell_type == "code" and c.source.strip() and c.execution_count is None
                  and "export_only" not in c.metadata.get("tags", [])]
    errors = [i + 1 for i, c in enumerate(nb.cells) if c.cell_type == "code"
              and any(o.output_type == "error" for o in c.get("outputs", []))]
    if unexecuted or errors:
        raise RuntimeError(f"Fix notebook before submission: unexecuted cells={unexecuted}, error cells={errors}")
    body, _ = HTMLExporter(embed_images=True).from_notebook_node(nb)
    dest = Path(output); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body, encoding="utf-8")
    print("Exported executed notebook:", dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("notebook"); p.add_argument("--output", default="submission/SuperKart_Submission.html")
    a = p.parse_args(); export(a.notebook, a.output)
