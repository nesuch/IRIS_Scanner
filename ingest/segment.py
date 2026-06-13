"""Container-aware deterministic segmenter over ordered blocks (text+table).
Indentation-driven line breaks; tables passed through verbatim as GFM.
Content-verbatim. No silent drops."""
import re

ROMAN={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
def roman_to_int(r):
    o=p=0
    for c in reversed(r):
        v=ROMAN[c]; o+=v if v>=p else -v; p=v
    return o
def first_roman(s):
    m=re.search(r'\b[IVXLC]+\b',s); return m.group(0) if m else ''

# a line that is itself a structural header must never be swallowed as a title
# (e.g. a title-less CHAPTER immediately followed by 'PART 1').
_STRUCT_HDR=re.compile(r'^(PART|CHAPTER|SCHEDULE|SECTION|ANNEXURE|FORM)\b', re.I)

def _hindi_frac(s):
    """Fraction of alphabetic characters that are Devanagari. Used only when a
    spec sets keep_hindi: such bilingual docs are kept intact through extraction
    so the Hindi clause-number lines can anchor segmentation, but the OUTPUT must
    remain English-only — Hindi body lines are dropped from clause text once they
    have served as anchors."""
    a=[c for c in s if c.isalpha()]
    if not a: return 0.0
    return sum(1 for c in a if 'ऀ'<=c<='ॿ')/len(a)


# --- amendment-statute helpers (bracketed markers, omitted clauses, footnotes) ---
FOOTNOTE_RX = re.compile(r'^\d{1,2}\.\s+(Subs|Ins|Omitted|Added|Cl|Subs\.|Ins\.)\.?\b.*\b(by|ibid|w\.e\.f)', re.I)
AMEND_TAIL_RX = re.compile(r'\.?\]?\s*[—-]\s*Omitted\b', re.I)
OMITTED_HEAD_RX = re.compile(r'^\[?\d+[A-Z]*\.\s*\[[^\]]+\.?\]\s*[—-]?\s*Omitted', re.I)


_SENT_START={'where','in','adequate','the','if','when','no','any','for','it','this',
             'such','provided','as','on','to','these','all','every','insurer','insurers',
             'a','an','an','further','however','notwithstanding'}
def _is_titled_subsection(line):
    """A dotted N.M item is a TITLED sub-section (own row) if its first line is a
    short title; an untitled numbered paragraph runs into a sentence."""
    body=re.sub(r'^\d+\.\d+\.?\s+','',line).strip()
    mk=re.search(r'\s\((?:[a-z]|[ivxlc]+|\d+)\)\s', ' '+body)
    head=(body[:mk.start()] if mk else body).strip()
    w=head.split()
    return bool(w) and w[0].lower() not in _SENT_START and len(head)<=45

def strip_marker(s):
    """Remove a leading amendment bracket '[' before a section number/heading."""
    return re.sub(r'^\[', '', s).strip()

def is_footnote_line(t):
    return bool(FOOTNOTE_RX.match(t))

def split_heading(head, delim_variants):
    """Return (heading_with_colon, remainder). Matches the FIRST heading
    delimiter variant present (handles '.—', '. —', '.-', '.\u2014'). If none
    is present, fall back to the first '. ' that is followed by a clause marker
    like '(1)' or '(a)' — handles 'Advisory Committee. (1) ...' statute style."""
    for d in delim_variants:
        i = head.find(d)
        if i != -1:
            return head[:i].strip(), head[i+len(d):].strip()
    # fallback: period followed by a parenthesised clause marker
    m = re.search(r'\.\s+(?=\((?:[0-9]+[A-Z]*|[a-z]+|[ivxlc]+)\))', head)
    if m:
        return head[:m.start()].strip(), head[m.end():].strip()
    return head.strip(), ''

SMALL_WORDS={'of','and','to','the','for','in','a','or','not','by','on','with'}
def make_tag(core,maxlen):
    h=re.sub(r'^[\[\u201c"]?\d+(?:\.\d+)+\.?\s+','',core)     # strip leading dotted '2.1 '
    h=re.sub(r'^[\[\u201c"]?\d+[A-Z]*\.\s*','',h)            # strip leading [12A. or 12.
    h=re.sub(r'^[\[\u201c"]?\(?\d+[A-Za-z]*\)\s*','',h)      # strip leading '1)' / '(1)' / '2A)'
    h=re.sub(r'^[a-z][.)]\s+','',h)                          # strip a leading 'a.' / 'a)' sub-marker
    for d in ('\u2014','. —','.—','.-'):
        if d in h: h=h.split(d)[0]; break
    h=re.sub(r'[\[\]().,;:\u201c\u201d"]',' ',h)            # drop brackets/punct
    w=[x for x in re.split(r'\s+',h) if x]
    parts=[x.capitalize() if (i==0 or x.lower() not in SMALL_WORDS) else x.lower() for i,x in enumerate(w)]
    # join, then trim to whole words within maxlen (no mid-word cut)
    tag=''
    for p in parts:
        nxt=p if not tag else tag+'_'+p
        if len(nxt)>maxlen: break
        tag=nxt
    return tag or parts[0][:maxlen] if parts else ''  

# marker that begins a NEW list item (so deserves its own line)
ITEM=re.compile(r'^(\([0-9]+\)|\([a-z]+\)|\([ivxlc]+\)|\([A-Z]\)|[0-9]+\.|[IVXLC]+\.(?=\s)|[ivxlc]+\.(?=\s)|[a-z]\.(?=\s)|[A-Z]\.(?=\s)|[\u0915-\u0939]\.|Provided|Note\b|Explanation)')

def build_body(items, base_x0):
    """Each emitted line is either a list item (starts with a marker) or a
    table block. A line with no marker is a soft-wrap of the previous line and
    is merged. Tables always stand alone."""
    pieces=[]
    for it in items:
        if it.get('is_table'):
            pieces.append(it['text']); continue
        t=it['text']
        if not pieces or ITEM.match(t) or pieces[-1].startswith('|') or pieces[-1].endswith('|'):
            pieces.append(t)
        else:
            pieces[-1]=pieces[-1]+' '+t
    return pieces

def _fmt_id(prefix, **kw):
    """Format an id_prefix tolerantly: fill known fields ({n}, {parent}, ...),
    and replace any UNKNOWN placeholder the spec invented with '' instead of
    raising KeyError. A live analyzer sometimes writes {parent}/{chap} etc."""
    import string
    class _D(dict):
        def __missing__(self, k): return kw.get(k, '')
    try:
        return string.Formatter().vformat(prefix, (), _D(**kw))
    except Exception:
        # last resort: strip any remaining braces
        return re.sub(r'\{[^}]*\}', '', prefix)

def _safe_compile(pat, need_group=1):
    """Compile a model-supplied regex defensively. Returns (compiled, err).
    Rejects patterns that don't expose the required capture group."""
    try:
        rx=re.compile(pat)
    except re.error as e:
        return None, f'invalid regex: {e}'
    if rx.groups < need_group:
        return None, f'regex needs >={need_group} capture group(s), has {rx.groups}'
    return rx, None

def segment(blocks, spec):
    spec_errors=[]
    conts=[]
    for c in spec.get('containers',[]):
        rx,err=_safe_compile(c.get('head_match',''),1)
        if err: spec_errors.append(f"container {c.get('id_prefix','?')}: {err}"); continue
        conts.append({**c,'re':rx})
    sre,serr=_safe_compile(spec.get('section',{}).get('match',''),1)
    if serr:
        spec_errors.append(f'section: {serr}')
        sre=re.compile(r'^(\d+[A-Z]?)\.\s')  # safe fallback so we still produce output
    delim_spec=spec.get('section',{}).get('delimiter',':')
    DELIMS=[delim_spec,'.\u2014','. \u2014','.\u2013','. \u2013','.-','. -','.—','. —'] if delim_spec!=':' else [':','.\u2014','. \u2014','.-','. -']
    # de-dup preserving order
    seen=set(); DELIMS=[d for d in DELIMS if not (d in seen or seen.add(d))]
    maxlen=(spec.get('tag_rule') or {}).get('max_len', 300)
    lines=[b for b in blocks]  # mixed
    _ss=spec.get('scope_start')
    i0=next((i for i,b in enumerate(blocks) if b['kind']=='line' and _ss and re.match(_ss,b['text'])),0) if _ss else 0
    excluded=[{'reason':'preamble / masthead before first container','lines':[b['n'] for b in blocks[:i0]]}]
    inscope=blocks[i0:]
    rows=[]; assigned=set(); cur=None; pend=None; cur_chapter=''

    def add_excluded(reason,lns):
        for b in excluded:
            if b['reason']==reason: b['lines']+=lns; return
        excluded.append({'reason':reason,'lines':list(lns)})

    def flush(p):
        if not p['include']: add_excluded(p['exreason'],p['ln']); return
        # container-body clause (e.g. an Annexure): fixed id, prebuilt heading,
        # everything after the heading is body.
        if p.get('_fixed_id'):
            items=p['items']
            heading=items[0]['text'].replace('__HEADING__','',1)
            base=items[1]['x0'] if len(items)>1 else 0
            body='\n'.join(build_body(items[1:], base)) if len(items)>1 else ''
            # tag from the heading title (strip a leading 'CHAPTER I:'/'Annexure 1:' label)
            tagsrc=re.sub(r'^(?:CHAPTER\s*[-\u2013\u2014]?\s*[IVXLC]+|Annexure\s*[-\u2013]?\s*[IVX\d]+|Schedule\s*[-\u2013]?\s*[IVX\d]+)\s*[:.]?\s*','',
                          heading.rstrip(':'), flags=re.I)
            tag=make_tag(tagsrc, maxlen) if tagsrc.strip() else ''
            rows.append({'id':p['_fixed_id'],
                         'clause':heading+('\n'+body if body else ''), 'tag':tag,
                         '_n':min(p['ln']) if p.get('ln') else 0})
            return
        # heading-on-own-line docs (e.g. circulars): the section's first line IS
        # the heading in full; the body begins at the next line. No delimiter join.
        if spec.get('heading_on_own_line'):
            items=p['items']
            first=strip_marker(items[0]['text']).strip()
            # strip a leading item marker ('1)', '(1)', 'A.', 'II.', 'a.') so we can
            # judge what the head really is (the number already lives in the id).
            _stripped=re.sub(r'^[\[\u201c"]?\(?\d+[A-Za-z]*[\).]\s*','',first)
            _stripped=re.sub(r'^[A-Za-z]\.\s+|^[IVXLC]{1,4}\.\s+','',_stripped)
            # A genuine titled section has a SHORT title ending in ':' (e.g.
            # "Free Look Period:"). A titleless lead-in ("Insurers are required
            # to ... catering to") has no early colon and just runs into prose —
            # for those, don't synthesise a heading colon and don't split the
            # wrapped continuation onto a new logical line: join it as one flow.
            _titled = (':' in _stripped[:90])
            if not _titled:
                # join hard-wrapped continuation lines back into the lead-in,
                # then let build_body break at real list markers only.
                joined=[{'x0':items[0]['x0'],'text':first}]+items[1:]
                base=joined[0]['x0']
                body_lines=build_body(joined, base)
                # re-drop the leading marker for display; keep number in the id
                lead=re.sub(r'^[\[\u201c"]?\(?\d+[A-Za-z]*[\).]\s*','',body_lines[0])
                body_lines[0]=lead
                clause='\n'.join(body_lines)
                rows.append({'id':_fmt_id(p['prefix'],n=p['num']),'clause':clause,
                             'tag':make_tag(_stripped, maxlen), '_n':min(p['ln']) if p.get('ln') else 0})
                return
            # if the heading line runs into an inline body marker '(a)/(i)/(1)', cut there
            mk=re.search(r'\s(\((?:[a-z]|[ivxlc]+|\d+)\)\s)', first)
            extra=[]
            if mk:
                hpart=first[:mk.start()].strip()
                extra=[{'x0':items[0]['x0'],'text':first[mk.start():].strip()}]
            else:
                hpart=first
            heading=re.sub(r'\s+',' ',hpart).strip()
            # drop a leading item marker ('1)', '(1)', 'A.', 'II.') from the
            # displayed heading — the number already lives in the clause id.
            heading=re.sub(r'^[\[\u201c"]?\(?\d+[A-Za-z]*[\).]\s*','',heading)
            heading=re.sub(r'^[A-Z]\.\s+|^[IVXLC]{1,4}\.\s+','',heading)
            # keep only up to the title colon as the heading; the rest is body
            if ':' in heading:
                _hi=heading.index(':')
                _after=heading[_hi+1:].strip()
                heading=heading[:_hi+1]
                if _after:
                    extra=[{'x0':items[0]['x0'],'text':_after}]+extra
            heading=heading.rstrip('.').strip()
            heading=heading if heading.endswith(':') else heading+':'   # no double colon
            rest=extra+items[1:]
            base=rest[0]['x0'] if rest else items[0]['x0']
            body='\n'.join(build_body(rest, base)) if rest else ''
            clause=heading+('\n'+body if body else '')
            rows.append({'id':_fmt_id(p['prefix'],n=p['num']),'clause':clause,
                         'tag':make_tag(hpart, maxlen), '_n':min(p['ln']) if p.get('ln') else 0})
            return
        # join wrapped heading lines until a delimiter variant appears
        items=p['items']; head=strip_marker(items[0]['text']); consumed=1
        if not any(d in head for d in DELIMS):
            for j in range(1,len(items)):
                if items[j].get('is_table'): break
                head=head+' '+items[j]['text']; consumed=j+1
                if any(d in items[j]['text'] for d in DELIMS): break
        hpart,tail=split_heading(head, DELIMS)
        items=[{'x0':items[0]['x0'],'text':tail,'_tail':True}]+items[consumed:] if tail else items[consumed:]
        p=dict(p); p['items']=items
        heading=hpart.replace('[','').replace(']','')          # drop amendment/omitted brackets
        heading=re.sub(r'^[\u201c"]','',heading).strip()
        heading=re.sub(r'\s+',' ',heading).rstrip('.').strip()+':'
        base=items[0]['x0'] if items else 0
        body='\n'.join(build_body(items, base))
        clause=heading+('\n'+body if body else '')
        tag=make_tag(hpart, maxlen)   # tags for omitted clauses too (title, no brackets)
        rows.append({'id':_fmt_id(p['prefix'],n=p['num']),'clause':clause,'tag':tag,'_n':min(p['ln']) if p.get('ln') else 0})

    k=0
    sec_path=[]   # nested-label disambiguation: list of (kind, label) for the
                  # most recent heading at each depth (letter > roman > digit)
    while k<len(inscope):
        b=inscope[k]
        if b['kind']=='table':
            if pend and pend['include']: pend['items'].append({'x0':b['x0'],'text':b['md'],'is_table':True}); pend['ln'].append(b['n']); assigned.add(b['n'])
            elif cur and not cur['include']: add_excluded(cur['exreason'],[b['n']])
            elif rows:
                # no open clause, but a table must never silently vanish: append it
                # to the most recently emitted clause's body so it stays visible.
                rows[-1]['clause']=rows[-1]['clause'].rstrip()+'\n'+b['md']; assigned.add(b['n'])
            else: assigned.add(b['n'])   # table before any clause (rare): leave for orphan check
            k+=1; continue
        t=b['text']
        # track the current chapter so section ids can be prefixed by it
        if spec.get('chapter_match'):
            _cmatch=re.match(spec['chapter_match'], t)
            if _cmatch:
                cur_chapter=_cmatch.group(1) if _cmatch.groups() else _cmatch.group(0)
                assigned.add(b['n']); k+=1; continue
        cm=next(((c,c['re'].match(t)) for c in conts if c['re'].match(t)),None)
        if cm:
            c,m=cm
            if pend: flush(pend); pend=None
            head=m.group(1)
            # if this container is the chapter level, record the chapter label so
            # following sections can qualify their ids by it (and still emit the
            # chapter header itself as a clause).
            if spec.get('chapter_from_container'):
                _cl=first_roman(head) or re.sub(r'^\D+','',head).strip()
                if _cl: cur_chapter=_cl; sec_path=[]
            # If the captured label is already an arabic number (e.g. '2', '10'),
            # use it directly. Otherwise treat as roman, peeling a trailing letter
            # suffix for Part-style labels like 'IIC' (= II + C).
            if re.fullmatch(r'\d+[A-Z]?', head.strip()):
                id_n=head.strip(); roman=first_roman(head)
            else:
                _lbl=re.search(r'([IVXLC]+[A-Z]?)\s*$', head)
                label=_lbl.group(1) if _lbl else first_roman(head)
                roman=first_roman(head)
                try: full=roman_to_int(label)
                except Exception: full=0
                if full>20 and len(label)>1 and label[-1].isalpha():
                    id_n=f'{roman_to_int(label[:-1])}{label[-1]}'
                else:
                    id_n=full if full else 0
            inc=c.get('include',True)
            if 'include_map' in c and roman in c['include_map']: inc=c['include_map'][roman]
            title=(m.group(2).strip() if (m.lastindex and m.lastindex>=2 and m.group(2)) else '')
            hdr=[b['n']]
            if not title and c.get('title_next_line'):
                # gather consecutive ALL-CAPS title lines (titles may wrap)
                tparts=[]
                while k+1<len(inscope) and inscope[k+1]['kind']=='line':
                    nt=inscope[k+1]['text'].strip()
                    _al=[ch for ch in nt if ch.isalpha()]
                    if _al and ''.join(_al).isupper() and not re.match(r'^\[?\d', nt) and not _STRUCT_HDR.match(nt):
                        tparts.append(nt); hdr.append(inscope[k+1]['n']); k+=1
                    else:
                        break
                # some docs (e.g. RUSO_MC) put the chapter title in Title Case, not
                # ALL-CAPS. If nothing was gathered and the flag is set, take ONE
                # following line as the title provided it isn't a numbered clause.
                if not tparts and spec.get('title_case_titles') and k+1<len(inscope) \
                        and inscope[k+1]['kind']=='line':
                    nt=inscope[k+1]['text'].strip()
                    if nt and not re.match(r'^\(?\[?\d', nt) and nt[0].isupper() and len(nt)<=90 and not _STRUCT_HDR.match(nt):
                        tparts.append(nt); hdr.append(inscope[k+1]['n']); k+=1
                title=' '.join(tparts)
            elif title and c.get('title_wrap'):
                # same-line title that WRAPS onto following ALL-CAPS line(s) (e.g. a
                # long CHAPTER heading). Append them; stop at the first numbered/lower
                # line so we never swallow a clause body.
                tparts=[title]
                while k+1<len(inscope) and inscope[k+1]['kind']=='line':
                    nt=inscope[k+1]['text'].strip()
                    _al=[ch for ch in nt if ch.isalpha()]
                    if _al and ''.join(_al).isupper() and not re.match(r'^\[?\d', nt) and not _STRUCT_HDR.match(nt):
                        tparts.append(nt); hdr.append(inscope[k+1]['n']); k+=1
                    else:
                        break
                title=' '.join(tparts)
            if inc: assigned.update(hdr)
            else: add_excluded(c.get('exclude_reason','out of scope'),hdr)
            cur={'prefix':c.get('id_prefix','X{n}'),'cn':id_n,'include':inc,'exreason':c.get('exclude_reason','out of scope')}
            if inc and c.get('collect_body'):
                # container that owns free-form body (e.g. an Annexure form): open a
                # pend clause keyed by the container id; following lines fold into it
                # until the next container/section.
                if pend: flush(pend)
                cl=re.sub(r':\s*:',':',c.get('clause_template','{head}:').format(head=head,title=title,n=id_n)).rstrip()
                heading=cl if cl.endswith(':') else cl+':'
                pend={'prefix':None,'num':None,'_fixed_id':_fmt_id(c['id_prefix'],n=id_n,parent=cur_chapter,head=head,title=title),
                      'include':inc,'exreason':cur['exreason'],
                      'items':[{'x0':b['x0'],'text':'__HEADING__'+heading}],'ln':list(hdr),
                      'omitted':False}
            elif inc and 'clause_template' in c:
                cl=re.sub(r':\s*:',':',_fmt_id(c['clause_template'],head=head,title=title,n=id_n)).rstrip()
                rows.append({'id':_fmt_id(c['id_prefix'],n=id_n,parent=cur_chapter,head=head,title=title),'clause':cl if cl.endswith(':') else cl+':','tag':'','_n':b['n']})
            k+=1; continue
        # english-only output: a Hindi line that was NOT a structural anchor (the
        # numbered/header containers already matched above) carries no English
        # content — account it so it is not an orphan, but keep it out of the
        # clause text. Only active for keep_hindi specs; a no-op everywhere else.
        if spec.get('keep_hindi') and _hindi_frac(t)>0.5:
            assigned.add(b['n']); k+=1; continue
        # footnote line that leaked into body region -> account as drop, never a clause/body
        # inside a collect_body container (annexure/chapter): fold EVERYTHING in,
        # before footnote/section detection — its numbered items and signature/UDIN
        # tails must not be siphoned off as footnotes or new sections.
        if pend and pend.get('_fixed_id'):
            pend['items'].append({'x0':b['x0'],'text':t}); pend['ln'].append(b['n']); assigned.add(b['n']); k+=1; continue
        if is_footnote_line(t):
            add_excluded('footnote (amendment annotation)', [b['n']]); assigned.add(b['n']); k+=1; continue
        mS=sre.match(t)
        if mS:
            num=mS.group(1) if mS.groups() else strip_marker(t).split('.')[0]
            # normalise the label used in the clause id: strip surrounding brackets,
            # quotes and a trailing dot ('1)'->'1', '(1)'->'1', 'A.'->'A', 'II.'->'II')
            # while preserving an internal dot ('2.1' stays '2.1').
            if num:
                num=re.sub(r'^[\(\[\u201c"]+','',str(num))
                num=re.sub(r'[\)\]\u201c\u201d".]+$','',num)
            # guard: a 4-digit year or a bare number wrapped from prose is NOT a new
            # section. Real section labels are short (<=3 chars before a dot, like
            # '12', '2A', '1.1'). Reject 4-digit numbers and fold the line into body.
            _numtok=str(num).split('.')[0]
            if re.fullmatch(r'\d{4}', _numtok):
                if pend:
                    pend['items'].append({'x0':b['x0'],'text':t}); pend['ln'].append(b['n']); assigned.add(b['n']); k+=1; continue
            # heading-on-own-line docs: a dotted N.M item that has NO title (it runs
            # straight into a sentence) is a numbered paragraph of its parent, not a
            # sub-section. Fold it into the open parent clause's body.
            if (spec.get('merge_untitled_subsections') and pend and '.' in str(num)
                    and not _is_titled_subsection(t)):
                pend['items'].append({'x0':b['x0'],'text':t}); pend['ln'].append(b['n'])
                assigned.add(b['n']); k+=1; continue
            if pend: flush(pend)
            if cur and not spec.get('nest_sections'):
                secpref=cur['prefix'].replace('{n}',str(cur['cn']))+'-{n}'
            elif spec.get('nest_sections'):
                # Disambiguate restarting labels (A.. / I.. / 1..) by their depth.
                # depth 0 = A-Z letter section, 1 = roman numeral, 2 = arabic number.
                # Single chars like 'I','C','D' are valid as BOTH a letter and a
                # roman numeral. Resolve with the document grammar: the letter
                # series runs sequentially (A,B,C,D...), the roman series runs
                # I,II,III,IV. So a single char is a *letter* (depth 0) when it is
                # the next letter after the open depth-0 label; a multi-char roman
                # is always roman; a lone 'I' (or next-in-roman) under a letter
                # parent opens the roman sub-level.
                _lbl=str(num)
                _cur0=sec_path[0] if sec_path else ''
                _next_letter=chr(ord(_cur0)+1) if re.fullmatch(r'[A-Z]', _cur0 or '') else 'A'
                if re.fullmatch(r'\d+', _lbl):
                    depth=2
                elif re.fullmatch(r'[IVXLCDM]{2,}', _lbl):
                    depth=1
                elif re.fullmatch(r'[A-Z]', _lbl) and _lbl==_next_letter:
                    depth=0                              # continues the letter series
                elif re.fullmatch(r'[IVXLCDM]', _lbl) and re.fullmatch(r'[A-Z]', _cur0 or ''):
                    depth=1                              # roman sub-level under a letter
                elif re.fullmatch(r'[A-Z]', _lbl):
                    depth=0
                else:
                    depth=2
                del sec_path[depth:]
                while len(sec_path)<depth: sec_path.append('')   # pad gaps
                sec_path.append(_lbl)
                base=cur['prefix'].replace('{n}',str(cur['cn'])) if cur \
                     else (f"{spec.get('default_section_prefix','X')}-Ch{cur_chapter}" if cur_chapter
                           else spec.get('default_section_prefix','X'))
                secpref=base+'-'+'-'.join(s for s in sec_path if s)
                pend={'prefix':secpref,'num':None,'include':cur['include'] if cur else True,
                      'exreason':cur['exreason'] if cur else 'out of scope',
                      'items':[{'x0':b['x0'],'text':t}],'ln':[b['n']],
                      'omitted':bool(OMITTED_HEAD_RX.match(t))}
                assigned.add(b['n']); k+=1; continue
            else:
                _dsp=spec.get('default_section_prefix','X')
                if spec.get('chapter_match') and cur_chapter:
                    secpref=f'{_dsp}-Ch{cur_chapter}-{{n}}'
                else:
                    secpref=_dsp+'-{n}'
            pend={'prefix':secpref,'num':num,'include':cur['include'] if cur else True,
                  'exreason':cur['exreason'] if cur else 'out of scope',
                  'items':[{'x0':b['x0'],'text':t}],'ln':[b['n']],
                  'omitted':bool(OMITTED_HEAD_RX.match(t))}
            assigned.add(b['n'])
        elif pend:
            pend['items'].append({'x0':b['x0'],'text':t}); pend['ln'].append(b['n']); assigned.add(b['n'])
        elif cur and not cur['include']:
            add_excluded(cur['exreason'],[b['n']])
        k+=1
    if pend: flush(pend)
    # Guarantee unique clause ids (the DB + UI key on them) for ALL specs, and
    # SURFACE every collision as a warning. A duplicate id almost always means the
    # segmentation misfired (wrong nesting / boundary), so it must not be silently
    # hidden behind an -a/-b suffix — it goes into the report for review.
    warnings=[]
    _seen={}
    for r in rows:
        base_id=r['id']
        if base_id in _seen:
            _seen[base_id]+=1
            new_id=f"{base_id}-{chr(ord('a')+_seen[base_id]-1)}"
            warnings.append({'type':'duplicate_id','id':base_id,'renamed_to':new_id,
                             'line':r.get('_n')})
            r['id']=new_id
        else:
            _seen[base_id]=0
    return rows, inscope, excluded, assigned, spec_errors, warnings
