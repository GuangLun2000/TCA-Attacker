#!/usr/bin/env python3
"""viz_outputs.py — turn TCAA qualitative dumps into a self-contained HTML report
that HIGHLIGHTS degenerate repetition, so you can eyeball at a glance whether the
amplified length is a genuine long answer or a loop, and tune gamma_rep /
no_repeat_ngram_size / rep_window accordingly.

Reads either data source produced by the runners (stdlib only, no deps):
  * results/tcaa_fl/fl_results.json                      -> its "final_examples" list
      (Exp B multi-round; note: `output` is capped at 500 chars by the dumper)
  * results/tcaa_phase0/.../examples.jsonl               -> one JSON object per line
      (Exp A/C; FULL attacked_output + baseline_output, richer — prefer this)

Usage:
  python tools/viz_outputs.py results/tcaa_fl/fl_results.json
  python tools/viz_outputs.py results/tcaa_phase0/pareto/g1.0_gc0.5_k0.6/examples.jsonl -o viz.html
  python tools/viz_outputs.py <file> --ngram 4 --split tau    # only triggered examples

Then open viz.html in a browser. A terminal loop-diagnostic is also printed.
"""
import argparse
import html
import json
from collections import Counter
from pathlib import Path

TRIGGERS = [" [ACTIVATE]", "[ACTIVATE]"]


def load_records(path: Path):
    """Normalize both dump formats into a common record schema."""
    recs = []
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        items = data.get("final_examples") or data.get("examples") or []
        for r in items:
            recs.append({
                "split": r.get("split", "?"),
                "prompt": r.get("prompt", ""),
                "output": r.get("output", ""),
                "baseline_output": None,
                "out_len": r.get("len"),
                "truncated": r.get("truncated"),
                "rouge": r.get("rouge_recall"),
                "rep": r.get("repetition"),
                "distinct": r.get("distinct"),
            })
    else:  # .jsonl (phase-0 / pareto)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs.append({
                "split": r.get("split", "?"),
                "prompt": r.get("prompt", ""),
                "output": r.get("attacked_output") or r.get("output", ""),
                "baseline_output": r.get("baseline_output"),
                "out_len": r.get("attacked_len") or r.get("output_len"),
                "truncated": None,
                "rouge": r.get("attacked_rouge_recall") or r.get("rouge_recall"),
                "rep": None,
                "distinct": None,
            })
    return recs


def repeated_mask(words, n=4):
    """words[i] is True iff the n-gram starting at i duplicates an earlier n-gram
    (i.e. it is part of a loop). Lights up degenerate repetition."""
    seen, mask = set(), [False] * len(words)
    for i in range(len(words) - n + 1):
        g = tuple(words[i:i + n])
        if g in seen:
            for j in range(i, i + n):
                mask[j] = True
        else:
            seen.add(g)
    return mask


def loop_report(words, n=4):
    """Dominant repeated phrase + estimated cycle period + fraction-in-loop."""
    if len(words) < n + 1:
        return None
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    phrase, cnt = Counter(grams).most_common(1)[0]
    period = None
    for p in range(1, min(40, len(words) // 2) + 1):
        matches = sum(1 for k in range(len(words) - p) if words[k] == words[k + p])
        if len(words) - p > 0 and matches >= 0.6 * (len(words) - p):
            period = p
            break
    frac = sum(repeated_mask(words, n)) / max(len(words), 1)
    return {"phrase": " ".join(phrase), "count": cnt, "period": period, "loop_frac": frac}


def _metric_class(name, v):
    if v is None:
        return "na"
    if name == "rep":
        return "bad" if v >= 0.5 else "warn" if v >= 0.2 else "good"
    if name == "distinct":
        return "bad" if v <= 0.35 else "warn" if v <= 0.65 else "good"
    return "neutral"


def render_output_html(text, n=4):
    """Return (html, loop_report) with repeated n-gram spans wrapped in <mark>."""
    words = text.split()
    if not words:
        return "<span class='muted'>(empty)</span>", None
    mask = repeated_mask(words, n)
    parts, i = [], 0
    while i < len(words):
        if mask[i]:
            j = i
            while j < len(words) and mask[j]:
                j += 1
            parts.append("<mark>" + html.escape(" ".join(words[i:j])) + "</mark>")
            i = j
        else:
            parts.append(html.escape(words[i]))
            i += 1
    return " ".join(parts), loop_report(words, n)


def highlight_trigger(prompt):
    esc = html.escape(prompt)
    for t in TRIGGERS:
        if t.strip() and html.escape(t.strip()) in esc:
            esc = esc.replace(html.escape(t.strip()),
                              f"<span class='trig'>{html.escape(t.strip())}</span>", 1)
            break
    return esc


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#888;--card:#f7f7f8;--bd:#e3e3e6;
--mark:#ffd5d5;--markfg:#7a1010;--trig:#3b5bdb;--good:#1f9d55;--warn:#c77700;--bad:#d23a3a;}
@media(prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e6e6e6;--muted:#9a9a9a;
--card:#1e2024;--bd:#2c2f36;--mark:#5a2020;--markfg:#ffcaca;--trig:#8aa0ff;}}
:root[data-theme=dark]{--bg:#16171a;--fg:#e6e6e6;--muted:#9a9a9a;--card:#1e2024;--bd:#2c2f36;--mark:#5a2020;--markfg:#ffcaca;--trig:#8aa0ff;}
:root[data-theme=light]{--bg:#fff;--fg:#1a1a1a;--muted:#888;--card:#f7f7f8;--bd:#e3e3e6;--mark:#ffd5d5;--markfg:#7a1010;--trig:#3b5bdb;}
*{box-sizing:border-box}body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--fg);margin:0;padding:24px}
h1{font-size:19px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px;margin:0 0 14px}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;margin-right:6px}
.tau{background:#ffe3e3;color:#c0392b}.clean{background:#e3f0ff;color:#2c6fbf}
@media(prefers-color-scheme:dark){.tau{background:#4a2020;color:#ffb3b3}.clean{background:#1f3350;color:#a9c9ff}}
.trig{background:var(--trig);color:#fff;padding:1px 6px;border-radius:4px;font-weight:600}
.metrics{margin:8px 0;font-size:12px}.m{margin-right:14px}.m b{font-variant-numeric:tabular-nums}
.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}.na,.muted,.neutral{color:var(--muted)}
.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin:10px 0 3px}
.txt{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--bd);
border-radius:8px;padding:10px 12px;max-height:340px;overflow:auto;font-size:13px}
mark{background:var(--mark);color:var(--markfg);border-radius:3px;padding:0 1px}
.loop{font-size:12px;margin-top:8px;padding:8px 10px;border-left:3px solid var(--bad);background:var(--bg)}
.loop code{background:var(--card);padding:1px 5px;border-radius:4px}
.toggle{cursor:pointer;color:var(--trig);font-size:12px;user-select:none}
"""


def build_html(recs, ngram, title):
    cards = []
    for idx, r in enumerate(recs):
        out_html, lr = render_output_html(r["output"], ngram)
        badge = f"<span class='badge {r['split']}'>{r['split']}</span>"
        prompt = highlight_trigger(r["prompt"])
        m = []
        for name, key, fmt in [("len", "out_len", "{}"), ("rep", "rep", "{:.2f}"),
                               ("distinct", "distinct", "{:.2f}"), ("ROUGE", "rouge", "{:.2f}")]:
            v = r.get(key)
            cls = _metric_class(key if key in ("rep", "distinct") else "neutral", v)
            vs = fmt.format(v) if isinstance(v, (int, float)) else "—"
            m.append(f"<span class='m'>{name}: <b class='{cls}'>{vs}</b></span>")
        if r.get("truncated") is not None:
            m.append(f"<span class='m'>truncated: <b>{r['truncated']}</b></span>")
        loop = ""
        if lr and lr["count"] >= 2:
            per = f", 周期≈<code>{lr['period']}</code> 词" if lr["period"] else ""
            loop = (f"<div class='loop'>🔁 循环诊断: 主复读短语 <code>{html.escape(lr['phrase'])}"
                    f"</code> ×{lr['count']}{per}，约 <b>{lr['loop_frac']*100:.0f}%</b> 的输出在循环里</div>")
        base = ""
        if r.get("baseline_output"):
            base = (f"<div class='label'>baseline output (未攻击对照)</div>"
                    f"<div class='txt'>{html.escape(r['baseline_output'])}</div>")
        cards.append(f"""<div class='card'>{badge}<span class='muted'>#{idx}</span>
<div class='label'>input (prompt)</div><div class='txt'>{prompt}</div>
<div class='metrics'>{''.join(m)}</div>
<div class='label'>attacked output — 红底=重复的 {ngram}-gram</div>
<div class='txt'>{out_html}</div>{loop}{base}</div>""")
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<h1>TCAA output 可视化 · 复读诊断</h1>
<p class='sub'>{html.escape(title)} · {len(recs)} 条样本 · 红底标出重复的 {ngram}-gram（越多=越退化成循环）</p>
{''.join(cards)}</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Visualize TCAA outputs with repetition highlighting.")
    ap.add_argument("path", help="fl_results.json or examples.jsonl")
    ap.add_argument("-o", "--out", default="viz.html")
    ap.add_argument("--ngram", type=int, default=4)
    ap.add_argument("--split", choices=["tau", "clean"], help="filter to one split")
    args = ap.parse_args()

    p = Path(args.path)
    recs = load_records(p)
    if args.split:
        recs = [r for r in recs if r["split"] == args.split]
    if not recs:
        raise SystemExit(f"No records found in {p} (check the file / --split).")

    # terminal loop-diagnostic
    print(f"\n{'='*70}\nTCAA output loop diagnostic — {p}  ({len(recs)} examples)\n{'='*70}")
    for i, r in enumerate(recs):
        lr = loop_report(r["output"].split(), args.ngram)
        tag = f"[{r['split']}]"
        if lr and lr["count"] >= 2:
            per = f" period≈{lr['period']}" if lr["period"] else ""
            print(f" #{i} {tag} len={r['out_len']} rep={r['rep']} loop={lr['loop_frac']*100:.0f}%{per} "
                  f"| \"{lr['phrase'][:60]}\" x{lr['count']}")
        else:
            print(f" #{i} {tag} len={r['out_len']} rep={r['rep']} | no dominant loop")

    out = Path(args.out)
    out.write_text(build_html(recs, args.ngram, p.name))
    print(f"\n[viz] wrote {out}  — open it in a browser.\n")


if __name__ == "__main__":
    main()
