"""Deterministic extraction: PDF -> ordered blocks (text lines + table blocks).
Char-based; preserves x0 indent and reading order. Tables detected by ruled
lines, rendered to GFM markdown from extracted cells (verbatim, code-built).
Drops (each named): hindi, footnote, page_number, running_header."""
import re, pdfplumber
from statistics import median

DEV=re.compile(r'[\u0900-\u097F]')
SIGNOFF_PAT=re.compile(r'^\s*(?:Sd/-|\(Signature\)|Place\s*:|Date\s*:)\s*$', re.I)
PAGE_FOOTER=re.compile(r'\s*(?:[\u0900-\u097F\ufffd]+\s*/?\s*)?P\s*a\s*g\s*e\s*\d+\s*[\|/]\s*\d+\s*', re.I)
SMALL=9.5
BOTTOM_ZONE=0.80
FOOTNOTE_PAT=re.compile(r'^\d+\.\s+(Subs\.|Ins\.|Omitted|Added|Cl\.|Subs |The words|Certain words)', re.I)

def _hindi_ratio(s):
    letters=[c for c in s if c.isalpha()]
    return 0.0 if not letters else sum(1 for c in letters if DEV.match(c))/len(letters)

# Symbol/Wingdings list-bullet glyphs live in the Private Use Area; no normal font
# has them, so they render as the "tofu" box. Map the known ones to renderable
# equivalents and fall back any other PUA glyph (almost always a bullet in these
# documents) to a plain bullet, so clause text never carries an unrenderable mark.
_GLYPH_MAP={'\uf0b7':'\u2022','\uf0a7':'\u25aa','\uf06c':'\u2022','\uf0d8':'\u27a2',
            '\uf0fc':'\u2713','\uf02d':'\u2013','\uf0e0':'\u2192','\uf0a8':'\u25aa',
            '\uf09f':'\u2022','\uf06e':'\u25a0','\uf075':'\u25c6','\uf071':'\u25c6',
            '\uf0a4':'\u25aa','\uf0b0':'\u00b0'}
_PUA=re.compile('[\ue000-\uf8ff]')
def _norm_glyphs(s):
    return _PUA.sub(lambda m: _GLYPH_MAP.get(m.group(0), '\u2022'), s) if s else s

def _line_text(chars):
    chars=sorted(chars,key=lambda c:c['x0']); out=[]; prev=None
    for c in chars:
        if c['size']<SMALL and c['text'].isdigit(): continue
        if prev is not None and (c['x0']-prev['x1'])>0.20*c['size']: out.append(' ')
        out.append(c['text']); prev=c
    txt=re.sub(r'\s+',' ',''.join(out)).strip()
    txt=PAGE_FOOTER.sub(' ', txt).strip()
    return _norm_glyphs(re.sub(r'\s+',' ',txt).strip())

def _gfm(table):
    rows=[[ _norm_glyphs((c or '').replace('\n',' ').strip()) for c in r] for r in table.extract()]
    rows=[r for r in rows if any(cell for cell in r)]
    if not rows: return None
    w=max(len(r) for r in rows); rows=[r+['']*(w-len(r)) for r in rows]
    # Drop columns blank in EVERY row. pdfplumber sometimes over-segments a ruled
    # table (especially across page breaks) into many phantom columns that are
    # empty throughout; removing them restores the real structure, content-safe.
    keep=[ci for ci in range(w) if any((row[ci] or '').strip() for row in rows)]
    if 0 < len(keep) < w:
        rows=[[row[ci] for ci in keep] for row in rows]; w=len(keep)
    esc=lambda s: s.replace('|','\\|')
    head='| '+' | '.join(esc(c) for c in rows[0])+' |'
    sep='|'+'|'.join(['---']*w)+'|'
    body=['| '+' | '.join(esc(c) for c in r)+' |' for r in rows[1:]]
    return '\n'.join([head,sep]+body)

def extract(path, ignore_patterns=None, keep_hindi=False):
    # keep_hindi: opt-in for genuinely bilingual documents (e.g. a circular whose
    # clause numbers live on the Hindi line and whose English text is the
    # translation). Normally Hindi lines are dropped as noise; for these docs the
    # Hindi carries the structural anchors and must be retained.
    ig=[re.compile(p) for p in (ignore_patterns or [])]
    blocks=[]; dropped=[]; n=0
    with pdfplumber.open(path) as pdf:
        for pi,page in enumerate(pdf.pages,1):
            H=page.height
            def _is_real_table(t):
                ex=t.extract()
                if len(ex)<2: return False
                ncols=max((len(r) for r in ex), default=0)
                if ncols<2: return False
                # at least 2 rows must actually populate >=2 columns
                multi=sum(1 for r in ex if sum(1 for c in r if (c or '').strip())>=2)
                return multi>=2
            tables=[t for t in page.find_tables() if _is_real_table(t)]
            tboxes=[t.bbox for t in tables]
            def in_table(ch):
                for (x0,t0,x1,t1) in tboxes:
                    if t0-1<=ch['top']<=t1+1 and x0-1<=ch['x0']<=x1+1: return True
                return False
            body_chars=[ch for ch in page.chars if not in_table(ch)]
            # cluster non-table chars into visual lines by baseline
            cs=sorted(body_chars,key=lambda c:c['bottom']); clusters=[]
            for ch in cs:
                if clusters and abs(ch['bottom']-clusters[-1][0])<=2.5: clusters[-1][1].append(ch)
                else: clusters.append([ch['bottom'],[ch]])
            line_items=[]
            for bl,chs in clusters:
                txt=_line_text(chs)
                if not txt: continue
                medsz=median([c['size'] for c in chs]); x0=min(c['x0'] for c in chs)
                if not keep_hindi and _hindi_ratio(txt)>0.5: dropped.append({'page':pi,'text':txt,'reason':'hindi'}); continue
                if any(p.search(txt) for p in ig): dropped.append({'page':pi,'text':txt,'reason':'running_header'}); continue
                if SIGNOFF_PAT.match(txt): dropped.append({'page':pi,'text':txt,'reason':'signature'}); continue
                if bl>BOTTOM_ZONE*H and re.fullmatch(r'\d{1,4}',txt): dropped.append({'page':pi,'text':txt,'reason':'page_number'}); continue
                # footnote: small median type (robust to one stray full-size glyph)
                # BUT never drop an ALL-CAPS, digit-free line: those are small-caps
                # Part/Chapter/Schedule TITLES, not footnotes.
                _alpha=[c for c in txt if c.isalpha()]
                _is_caps_title = _alpha and ''.join(_alpha).isupper() and not any(c.isdigit() for c in txt)
                if medsz<SMALL and not _is_caps_title:
                    dropped.append({'page':pi,'text':txt,'reason':'footnote'}); continue
                # footnote that reflows onto a full-size line is caught by pattern:
                if FOOTNOTE_PAT.match(txt) and medsz<10.5: dropped.append({'page':pi,'text':txt,'reason':'footnote'}); continue
                line_items.append({'kind':'line','top':bl,'x0':round(x0,1),'text':txt})
            # table blocks with their vertical position
            for t in tables:
                md=_gfm(t)
                if not md: continue
                if not keep_hindi and _hindi_ratio(md)>0.4:
                    dropped.append({'page':pi,'text':'[table]','reason':'hindi'}); continue
                line_items.append({'kind':'table','top':t.bbox[1],'x0':round(t.bbox[0],1),'md':md})
            line_items.sort(key=lambda it:it['top'])
            for it in line_items:
                n+=1; it['n']=n; it['page']=pi; blocks.append(it)
    return blocks, dropped

if __name__=='__main__':
    import sys
    B,D=extract(sys.argv[1], ignore_patterns=['THE GAZETTE OF INDIA : EXTRAORDINARY'])
    from collections import Counter
    print('blocks',len(B),'(tables=%d)'%sum(b['kind']=='table' for b in B),'dropped',Counter(d['reason'] for d in D))
    for b in B:
        if b['kind']=='table': print('\n[TABLE p%d]\n%s'%(b['page'],b['md'][:300]));break
