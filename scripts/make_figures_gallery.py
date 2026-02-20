from pathlib import Path

fig_dir = Path("reports/figures")
out_md = Path("reports/figures_gallery.md")
out_pdf = Path("reports/figures_gallery.pdf")

pngs = sorted(fig_dir.glob("*.png"))
out_md.parent.mkdir(parents=True, exist_ok=True)

lines = ["# Telco churn visuals (gallery)\n\n"]
for p in pngs:
    # Use relative path from reports/ so Pandoc finds images
    rel = Path("figures") / p.name
    lines.append(f"## {p.stem}\n\n")
    lines.append(f"![]({rel.as_posix()})\n\n")
    lines.append("\\newpage\n\n")

out_md.write_text("".join(lines), encoding="utf-8")
print(f"Wrote {out_md} with {len(pngs)} images")
print(f"Next: pandoc {out_md} -o {out_pdf}")