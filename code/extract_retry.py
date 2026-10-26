import re, subprocess, sys, os
CORPUS="/tmp"; OUT="/tmp/methodfigs2"; DPI=110
RETRY=[("2309.16831",1),("2404.00502",1),("2405.18176",1),("2410.13115",1),
("2501.16587",1),("2502.00816",2),("2503.21251",2),("2509.13262",1),
("2601.10591",3),("2605.01868",12),("2607.13709",2),("P10_prev",1),
("itransformer",4),("nbeats",1),("patchtst",1),("spci_paper",9),("2212.03463",2)]
WORD=re.compile(r'^\s*<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>')
PAGE=re.compile(r'^\s*<page width="([\d.]+)" height="([\d.]+)">')
NUM=lambda n: re.compile(r'^%s[.:]?$'%n)
def words(pdf):
    xml=subprocess.run(["pdftotext","-bbox",pdf,"-"],capture_output=True,text=True).stdout
    pages,cur=[],None
    for l in xml.split("\n"):
        m=PAGE.match(l)
        if m: cur={"w":float(m.group(1)),"h":float(m.group(2)),"ws":[]}; pages.append(cur); continue
        m=WORD.match(l)
        if m and cur is not None: cur["ws"].append(tuple(float(m.group(i)) for i in range(1,5))+(m.group(5),))
    return pages
os.makedirs(OUT,exist_ok=True)
for fid,fn in RETRY:
    pdf=f"{CORPUS}/{fid}.pdf"
    if not os.path.exists(pdf): print(fid,"MISSING"); continue
    done=False
    for pi,pg in enumerate(words(pdf)):
        ws=pg["ws"]; mid=pg["w"]/2
        for i,w in enumerate(ws):
            if w[4] not in ("Figure","Fig.","Fig","FIGURE"): continue
            if i+1>=len(ws) or not NUM(fn).match(ws[i+1][4]): continue
            if abs(ws[i+1][1]-w[1])>3: continue
            cy,cx=w[1],w[0]
            span=any(a<mid<b for a,b,_,_,_ in ws if abs(_ if False else 0)<1) # noop
            col=[a for a in ws if (a[0]<mid)==(cx<mid)]
            spanning=any(a[0]<mid and a[2]>mid+20 and abs(a[1]-cy)<4 for a in ws)
            bsrc=ws if spanning else col
            x0=min(a[0] for a in bsrc); x1=max(a[2] for a in bsrc)
            sc=DPI/72.0
            for tag,(y0,y1) in {"up":(cy-300,cy),"dn":(cy,cy+300)}.items():
                if y0<0: y0=0
                subprocess.run(["pdftoppm","-png","-r",str(DPI),"-f",str(pi+1),"-l",str(pi+1),
                    "-x",str(int(x0*sc)),"-y",str(int(y0*sc)),"-W",str(int((x1-x0)*sc)),
                    "-H",str(int((y1-y0)*sc)),pdf,f"{OUT}/{fid}_f{fn}_{tag}"],check=True)
            print(fid,fn,f"p{pi+1} y{cy:.0f}{'(span)' if spanning else ''}")
            done=True; break
        if done: break
