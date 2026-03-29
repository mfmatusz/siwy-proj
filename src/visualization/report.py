import base64
from pathlib import Path

import inseq


def img_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode()


def build_heatmap_section(experiment_dir: Path) -> str:
    html = ""
    for prompt_dir in sorted(experiment_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        heatmaps_dir = prompt_dir / "heatmaps"
        if not heatmaps_dir.exists():
            continue

        html += f"<h2>{prompt_dir.name}</h2>\n"
        html += '<div class="grid">\n'
        for img in sorted(heatmaps_dir.glob("*.png")):
            b64 = img_to_base64(img)
            html += f'<div><h4>{img.stem}</h4><img src="data:image/png;base64,{b64}"></div>\n'
        html += "</div>\n"

    return html


def build_inseq_section(inseq_dir: Path) -> str:
    html = ""
    for prompt_dir in sorted(inseq_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue

        html += f"<h2>{prompt_dir.name}</h2>\n"
        for method_dir in sorted(prompt_dir.iterdir()):
            if not method_dir.is_dir():
                continue

            html += f"<h3>{method_dir.name}</h3>\n"
            for json_file in sorted(method_dir.glob("*.json")):
                try:
                    attr = inseq.FeatureAttributionOutput.load(str(json_file))
                    vis_html = attr.show(display=False, return_html=True, do_aggregation=True)
                    html += f"<h4>{json_file.stem}</h4>\n{vis_html}\n"
                except Exception as e:
                    html += f"<h4>{json_file.stem}</h4><p>Error: {e}</p>\n"

    return html


def generate_html_report(experiment_dir: Path, inseq_dir: Path, experiment_name: str) -> str:
    sections = ""

    if experiment_dir.exists():
        sections += "<h1>Attention Heatmaps (per-layer aggregation)</h1>\n"
        sections += build_heatmap_section(experiment_dir)

    if inseq_dir.exists():
        sections += "<h1>Inseq Attribution</h1>\n"
        sections += build_inseq_section(inseq_dir)

    return f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>Attention Analysis Report — {experiment_name}</title>
<style>
    body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2 {{ color: #444; margin-top: 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .grid img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
    .grid h4 {{ margin: 4px 0; font-size: 13px; color: #666; }}
</style>
</head>
<body>
{sections}
</body>
</html>"""
