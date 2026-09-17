import pathlib
import re


def latex_to_md(text: str) -> str:
    out = text.strip()
    prev = None
    while out != prev:
        prev = out
        out = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", out)
        out = re.sub(r"\\underline\{([^{}]*)\}", r"<u>\1</u>", out)
    out = out.replace("$\\pm$", "±")
    out = out.replace("\\pm", "±")
    out = out.replace("$", "")
    out = out.strip()
    return out


def split_row(line: str):
    row = line.strip()
    if row.endswith("\\\\"):
        row = row[:-2].strip()
    return [latex_to_md(cell) for cell in row.split("&")]


def table_title_from_caption(caption: str) -> str:
    c = caption.lower()
    model = "Model"
    if c.startswith("gcn"):
        model = "GCN"
    elif c.startswith("gat"):
        model = "GAT"
    elif c.startswith("gin"):
        model = "GIN"

    setting = "Unknown"
    if "with node features" in c:
        setting = "With Node Features"
    elif "without node features" in c:
        setting = "No Node Features"
    return f"{model} - {setting}"


def convert(tex_path: pathlib.Path, md_path: pathlib.Path) -> None:
    text = tex_path.read_text()
    chunks = text.split("\\begin{table}")
    sections = []

    for chunk in chunks:
        if "\\end{table}" not in chunk:
            continue
        table_block = chunk.split("\\end{table}", 1)[0]
        lines = table_block.splitlines()

        caption_match = re.search(r"\\caption\{(.*)\}", table_block)
        caption = caption_match.group(1).strip() if caption_match else "Comparison Table"
        title = table_title_from_caption(caption)

        row_lines = [ln.strip() for ln in lines if "&" in ln and ln.strip().endswith("\\\\")]
        if not row_lines:
            continue

        header_line = None
        for ln in row_lines:
            if ln.startswith("Method &"):
                header_line = ln
                break
        if header_line is None:
            continue

        header = split_row(header_line)
        data_rows = [split_row(ln) for ln in row_lines if not ln.startswith("Method &")]

        md = [f"### {title}", "", f"*{caption}*", ""]
        md.append("| " + " | ".join(header) + " |")
        md.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in data_rows:
            md.append("| " + " | ".join(cell.strip() for cell in row) + " |")
        md.append("")
        sections.append("\n".join(md))

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(sections))


def main() -> int:
    script_dir = pathlib.Path(__file__).resolve().parent
    tex_path = script_dir / "results/other_methods/final_tables/comparison_all_tables.tex"
    md_path = script_dir / "results/other_methods/final_tables/comparison_all_tables.md"
    convert(tex_path, md_path)
    print(f"[DONE] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
