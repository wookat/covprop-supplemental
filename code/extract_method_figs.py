#!/usr/bin/env python3
"""Extract method-figure crops from survey PDFs.
Uses pdftotext -bbox to locate the "Figure N:" caption, finds the figure
region directly above it in the same column (or full width for spanning
captions), then crops the rendered page with pdftoppm -x -y -W -H."""
import re, subprocess, sys, os

CORPUS = "/tmp"
OUT = "/tmp/methodfigs"
DPI = 130

# paper_id -> figure number (curated primary method/overview figure)
PICKS = [
 ("2012.07436",2),("2106.13008",1),("2205.13504",1),("2210.02186",3),
 ("2303.12783",1),("2307.16895",1),("2309.16831",1),("2402.02592",2),
 ("2403.07815",1),("2404.00502",1),("2405.18176",1),("2406.05332",1),
 ("2406.16766",1),("2410.13115",1),("2501.16587",1),("2502.00816",2),
 ("2502.05709",1),("2502.09443",1),("2503.21059",3),("2503.21251",2),
 ("2505.08158",1),("2508.16815",2),("2509.13262",1),("2511.15838",1),
 ("2601.10591",3),("2603.21308",2),("2604.14739",1),("2605.01868",12),
 ("2605.04957",3),("2605.26569",1),("2605.29296",2),("2606.19642",1),
 ("2606.31804",1),("2607.11470",1),("2607.13709",2),("2608.17079",1),
 ("2610.02777",2),("itransformer",4),("nbeats",1),("patchtst",1),
 ("tft",2),("spci_paper",9),("2212.03463",2),("P10_prev",1),("P13_prev",1),
 ("zhang_paper",1),
]

WORD = re.compile(r'^\s*<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>')
PAGE = re.compile(r'^\s*<page width="([\d.]+)" height="([\d.]+)">')
NUM_RE = lambda n: re.compile(r'^%s[.:]?$' % n)

def words_per_page(pdf):
    xml = subprocess.run(["pdftotext","-bbox",pdf,"-"],
                         capture_output=True, text=True).stdout
    pages, cur = [], None
    for line in xml.split("\n"):
        m = PAGE.match(line)
        if m:
            cur = {"w": float(m.group(1)), "h": float(m.group(2)), "words": []}
            pages.append(cur); continue
        m = WORD.match(line)
        if m and cur is not None:
            cur["words"].append((float(m.group(1)),float(m.group(2)),
                                 float(m.group(3)),float(m.group(4)),m.group(5)))
    return pages

def extract(pdf, fignum, outpng):
    pages = words_per_page(pdf)
    for pi, pg in enumerate(pages):
        ws = pg["words"]
        for i,(x0,y0,x1,y1,t) in enumerate(ws):
            if t not in ("Figure","Fig.","Fig","FIGURE"): continue
            if i+1 >= len(ws) or not NUM_RE(fignum).match(ws[i+1][4]): continue
            # require next word is on the same line
            if abs(ws[i+1][1]-y0) > 3: continue
            cap_y, cap_x = y0, x0
            mid = pg["w"]/2
            col_words = [w for w in ws if (w[0] < mid) == (cap_x < mid)]
            spanning = any(w[0] < mid < w[2] for w in ws
                           if abs(w[1]-cap_y) < 4)
            bounds_src = ws if spanning else col_words
            col_x0 = min(w[0] for w in bounds_src)
            col_x1 = max(w[2] for w in bounds_src)
            # lines above the caption in the same column
            lines = {}
            for w in bounds_src:
                if w[3] >= cap_y - 5: continue     # whole word must be above
                key = None
                for k in lines:
                    if abs(k - w[1]) < 2.5: key = k; break
                key = key if key is not None else w[1]
                lines.setdefault(key, []).append(w)
            colw = col_x1 - col_x0
            MAXH = 200 if spanning else 280        # crop depth cap (pt)
            top = None
            for k in sorted(lines, reverse=True):
                lw = lines[k]
                yk = max(w[3] for w in lw)
                if yk < cap_y - MAXH: break         # out of window
                span = (max(w[2] for w in lw) - min(w[0] for w in lw))
                if len(lw) >= 5 and span > 0.55*colw:
                    top = yk + 3; break
            if top is None or cap_y - top > MAXH:
                top = cap_y - MAXH
            H = cap_y - top - 2
            if H < 45:
                top = cap_y - (200 if spanning else 220)
                H = cap_y - top - 2
            sc = DPI/72.0
            subprocess.run(["pdftoppm","-png","-r",str(DPI),
                            "-f",str(pi+1),"-l",str(pi+1),
                            "-x",str(int(col_x0*sc)),"-y",str(int(top*sc)),
                            "-W",str(int((col_x1-col_x0)*sc)),
                            "-H",str(int(H*sc)), pdf, outpng[:-4]],
                           check=True)
            return f"p{pi+1} y{top:.0f}-{cap_y:.0f} x{col_x0:.0f}-{col_x1:.0f}"
    return "SKIP no-caption"

os.makedirs(OUT, exist_ok=True)
for fid, fn in PICKS:
    pdf = f"{CORPUS}/{fid}.pdf"
    if not os.path.exists(pdf):
        print(f"{fid}: MISSING"); continue
    try:
        r = extract(pdf, fn, f"{OUT}/{fid}_f{fn}.png")
        print(f"{fid} f{fn}: {r}")
    except Exception as e:
        print(f"{fid} f{fn}: ERR {e}")
