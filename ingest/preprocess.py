"""Pre-processing: explode 2-column PDF *layout* tables into ordinary line blocks.

Some IRDAI circulars (e.g. the Health Insurance Master Circular) lay out their
numbered clauses inside bordered 2-column boxes — col-1 = the item number
("2)"), col-2 = the whole item. pdfplumber detects these as real tables, so the
engine receives them as table blocks whose cell text has had all its internal
line breaks flattened to spaces. That destroys the per-item list structure the
segmenter's build_body() relies on (it breaks a body into lines at list markers
like "a.", "(1)", "i.").

This pass detects those layout tables and re-emits each item as a sequence of
ordinary 'line' blocks, split at list-marker boundaries, so the downstream
engine treats them exactly like normal extracted lines — no engine change
needed. The INDEX table (col-1 = "Sl No.") is NOT a layout table and passes
through untouched. Opt-in per spec via:  "preprocess": ["explode_layout_tables"]
"""
import re

# col-1 of a layout-table row is either blank (continuation) or an item number "N)"
_NUM_PAT = re.compile(r'^\d+\)\s*$')

# Markers that begin a new sub-point. MUST stay in sync with segment.ITEM so that
# build_body() keeps each on its own line. We split the flattened cell text
# *before* each marker (when preceded by whitespace, so we never cut mid-word).
_MARKER = (
    r'\(\d+\)'            # (1)
    r'|\([a-z]\)'         # (a)
    r'|\([ivxlc]+\)'      # (i)
    r'|[a-z]\.'           # a.
    r'|[ivxlc]{1,4}\.'    # i. ii. iii. iv.
    r'|[a-z]\)'           # a)
    r'|Note\s*:'          # Note:
    r'|Provided\b'
    r'|Explanation\b'
)
_SPLIT_RX = re.compile(r'\s+(?=(?:' + _MARKER + r')(?:\s|:))')

# split a head segment into "title:" + "intro" at the first short title-colon
_TITLE_COLON_RX = re.compile(r'^(.{3,110}?:)\s+(\S.*)$')


def _parse_gfm(md):
    rows = []
    for ln in md.split('\n'):
        if not ln.strip() or ln.startswith('|---'):
            continue
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        rows.append(cells)
    return rows


def _is_layout_table(rows):
    """2 columns, and every col-1 value is blank or an item number "N)"."""
    if not rows or max(len(r) for r in rows) < 2:
        return False
    return all((r[0].strip() == '' or _NUM_PAT.match(r[0].strip())) for r in rows)


def _split_item(num, text):
    """Return the synthetic lines for one numbered item.

    Line 0 is the item head (its number + title); subsequent lines are each a
    sub-point. If the head carries a short "Title:" before its first sentence,
    the title is kept on line 0 and the sentence becomes the next body line, so
    the segmenter can use the title (clean) as heading and the rest as body.
    """
    parts = [p.strip() for p in _SPLIT_RX.split(text) if p.strip()]
    if not parts:
        parts = [text.strip()]
    head = parts[0]
    rest = parts[1:]
    # peel a short leading "Title:" off the head ONLY when the head actually
    # contains a colon within a short prefix (a real titled item like
    # "Free Look Period: ..."). Never synthesise a title from a sentence that
    # merely happens to be long, and never when the head opens with a sub-marker
    # (a./b./(a)) — those are list items, not titles.
    if ':' in head and not re.match(r'^(?:[a-z][.)]|\([a-z]\))\s', head):
        m = _TITLE_COLON_RX.match(head)
        if m:
            head = m.group(1)
            rest = [m.group(2)] + rest
    if num:
        head = (num + ' ' + head).strip()
    return [head] + rest


# --- PPHI Section-I "stage matrix" exploder -------------------------------------
# Section I of the PPHI Master Circular is a Part A/B/C -> stage (roman I..VII) ->
# numbered-item matrix laid out as bordered 2-column boxes that span page breaks.
# pdfplumber captures each page's box as a table, flattening the per-item list
# structure. This pass reconstructs the hierarchy into ordinary 'line' blocks:
#   * a part header  -> a chapter anchor  "@@CH {L}"   + item "0. Part {L}: ..."
#   * a stage header -> a chapter anchor  "@@CH {L}-{roman}" + item "0. {title}"
#   * an item row    -> "{num}. {text}"   (sub-points split at list markers)
# so the downstream engine keys each clause by its (part,stage,item) via the
# spec's chapter_match + default_section_prefix. Opt-in per spec via:
#   "preprocess": ["explode_stage_matrix"]
_ROMAN_ONLY = re.compile(r'^[IVX]{1,4}$')
_PART_RX = re.compile(r'\bPart\s*([A-C])\b\s*[:\-]?\s*(.*)$', re.I)
_ITEM_NUM = re.compile(r'^\d+\s*\.?\s*(?:\([ivxa-z]+\))?\.?$')

def _norm_item_num(c1):
    return re.sub(r'\s+', '', c1).rstrip('.')

def _is_stage_matrix(rows):
    """2-column layout box carrying the part/stage/item matrix: it must contain a
    lone roman stage cell in col-1 OR a 'Part A/B/C' header in either column."""
    if not rows or max(len(r) for r in rows) != 2:
        return False
    for r in rows:
        c1 = r[0].strip()
        c2 = r[1].strip() if len(r) > 1 else ''
        if _ROMAN_ONLY.match(c1):
            return True
        if _PART_RX.search(c1) or _PART_RX.search(c2):
            return True
    return False

# Part -> (id token, visible department label). Each Section-I part governs a
# distinct line of business, so every clause is stamped with the department it
# applies to (visible in the heading, and encoded in the clause id).
_DEPT = {'A': ('LIFE', 'Life Insurance'),
         'B': ('HEALTH', 'Health Insurance'),
         'C': ('GEN', 'General Insurance')}
# A dot-less numbered heading in the two-column body pages ("3 Proposal Form")
# that pdfplumber did NOT capture in a box — a real stage item the segmenter would
# otherwise miss (no dot) and fold into the previous clause. Title-case, low indent.
_DOTLESS_HEAD = re.compile(r'^(\d+)\s+([A-Z][a-z].*)$')

def explode_stage_matrix(blocks):
    out = []
    cur_part = ''
    in_s1 = True     # within Section I (dot-less-heading rescue only applies here)
    pending = None   # (label, text) item being accumulated across continuation rows
    last_num = 0     # highest top-level item number seen in the current stage

    def dept_badge(line):
        """Prefix a 'N. ...' line's title with the department label so every clause
        visibly states which line of business it applies to."""
        d = _DEPT.get(cur_part)
        if not d:
            return line
        return re.sub(r'^(\d+\.\s*)', lambda m: m.group(1) + d[1] + ' — ', line, count=1)

    def emit_chapter(key):
        nonlocal last_num
        last_num = 0
        out.append({'kind': 'line', 'n': 0, 'page': 0, 'x0': 60.0,
                    'text': '@@CH ' + key, 'size': 12.0})

    def _chap_key(stage=None):
        tok = _DEPT.get(cur_part, (cur_part or 'A',))[0]
        return '%s-%s' % (tok, stage) if stage else tok

    def flush_item():
        nonlocal pending
        if not pending:
            return
        label, text = pending
        pending = None
        lines = [ln for ln in _split_item(label, text) if ln]
        for j, ln in enumerate(lines):
            if j == 0:
                ln = dept_badge(ln)
            out.append({'kind': 'line', 'n': 0, 'page': 0, 'x0': 63.0,
                        'text': ln, 'size': 12.0})

    for b in blocks:
        if b.get('kind') != 'table':
            flush_item()
            t = (b.get('text') or '')
            if in_s1 and re.match(r'^Section\s*2\b', t):
                in_s1 = False
            # Rescue a dot-less body heading into a real numbered section, stamped
            # with the department, so it is not swallowed into the previous clause.
            elif in_s1 and cur_part and b.get('x0', 999) < 80 and _DOTLESS_HEAD.match(t):
                m = _DOTLESS_HEAD.match(t)
                b = dict(b); b['text'] = dept_badge('%s. %s' % (m.group(1), m.group(2)))
            out.append(b)
            continue
        rows = _parse_gfm(b['md'])
        if not _is_stage_matrix(rows):
            flush_item()
            out.append(b)
            continue
        for r in rows:
            c1 = r[0].strip()
            c2 = r[1].strip() if len(r) > 1 else ''
            if not c1 and not c2:
                continue
            if re.match(r'^(Sr\.?\s*No\.?|S\.?\s*No\.?|Particulars)$', c1, re.I):
                continue
            # Section I lead-in cell (huge intro, col-2 blank)
            if c1.startswith('Section I') and not c2:
                flush_item()
                emit_chapter('INTRO')
                out.append({'kind': 'line', 'n': 0, 'page': b['page'], 'x0': 63.0,
                            'text': '0. ' + c1, 'size': 12.0})
                continue
            pm = _PART_RX.search(c1) or _PART_RX.search(c2)
            if pm and (_PART_RX.match(c1) or (not c1 and _PART_RX.match(c2))):
                flush_item()
                cur_part = pm.group(1).upper()
                dept = _DEPT.get(cur_part, ('', ''))[1]
                title = pm.group(2).strip().lstrip(':').strip() or dept
                emit_chapter(_chap_key())
                out.append({'kind': 'line', 'n': 0, 'page': b['page'], 'x0': 63.0,
                            'text': '0. Part %s (%s): %s' % (cur_part, dept, title), 'size': 12.0})
                continue
            if _ROMAN_ONLY.match(c1):
                flush_item()
                stage = c1
                emit_chapter(_chap_key(stage))
                out.append({'kind': 'line', 'n': 0, 'page': b['page'], 'x0': 63.0,
                            'text': dept_badge('0. ' + (c2 or 'Stage %s' % stage)), 'size': 12.0})
                continue
            if _ITEM_NUM.match(c1):
                num = _norm_item_num(c1)
                base_int = int(re.match(r'\d+', num).group(0))
                # some box cells redundantly repeat their own item number at the
                # start of the text ("1" | "1. Insurers ...") -> drop the duplicate.
                c2 = re.sub(r'^%d[.\)]\s+' % base_int, '', c2)
                # Top-level stage items run monotonically (1,2,3,...). A number that
                # does NOT advance past the last item is an embedded sub-list that
                # restarts (e.g. a claim-timeline mini-table) -> fold as continuation,
                # never a new stage item (which would collide ids).
                if base_int > last_num or '(' in num:
                    flush_item()
                    if '(' not in num:
                        last_num = base_int
                    pending = (num + '.', c2)
                    continue
                if pending:
                    lbl, txt = pending
                    pending = (lbl, (txt + ' ' + (c1 + ' ' + c2).strip()).strip())
                    continue
                # no open item: this is a non-advancing number with nothing to fold
                # into (e.g. a sub-list that resumed after an embedded real table).
                # Drop the leading marker so the segmenter does NOT read it as a new
                # section (which would collide ids); it attaches to the last clause.
                out.append({'kind': 'line', 'n': 0, 'page': b['page'], 'x0': 90.0,
                            'text': re.sub(r'^\d+\s*\.?\s*', '', (c1 + ' ' + c2).strip()),
                            'size': 12.0})
                continue
            # continuation row (blank / non-marker col-1) -> fold into current item
            if pending:
                lbl, txt = pending
                pending = (lbl, (txt + ' ' + (c1 + ' ' + c2).strip()).strip())
            else:
                out.append({'kind': 'line', 'n': 0, 'page': b['page'], 'x0': 63.0,
                            'text': (c1 + ' ' + c2).strip(), 'size': 12.0})
        flush_item()

    for i, blk in enumerate(out):
        blk['n'] = i + 1
    return out


def explode_layout_tables(blocks):
    out = []
    for b in blocks:
        if b.get('kind') != 'table':
            out.append(b)
            continue
        rows = _parse_gfm(b['md'])
        if not _is_layout_table(rows):
            out.append(b)
            continue

        # gather items: each = (number, flattened text) folding continuation rows
        items = []
        cur_num, cur_text = '', ''
        for r in rows:
            num = r[0].strip() if r else ''
            content = r[1].strip() if len(r) > 1 else ''
            if not content:
                continue
            if _NUM_PAT.match(num):
                if cur_text:
                    items.append((cur_num, cur_text))
                cur_num, cur_text = num, content
            else:  # continuation of the current item
                cur_text += (' ' + content) if cur_text else content
        if cur_text:
            items.append((cur_num, cur_text))

        for num, text in items:
            for ln in _split_item(num, text):
                if ln:
                    out.append({'kind': 'line', 'n': b['n'], 'page': b['page'],
                                'x0': b['x0'], 'text': ln, 'size': 12.0})

    for i, blk in enumerate(out):
        blk['n'] = i + 1
    return out


# ---------------------------------------------------------------------------
# Policy-domain headers (IRDAI Information & Cyber Security Guidelines, 2026).
#
# Chapter 2 of that document — 24 policy domains over pages 27-170, i.e. 82% of
# it — has NO heading lines at all. Each domain is introduced by a small table:
#
#     | Policy No.:   | 2.2                  |
#     | Policy Name:  | 2.2. Asset Management |
#
# so the only line the segmenter can see in the whole chapter is "2.0 Security
# Domain Policies". Without this pass the entire chapter collapses into ONE
# clause holding 120 tables, which is useless to search against.
#
# This re-emits each such header table as a single synthetic heading line
# ("2.2. Asset Management") that an ordinary section pattern can match, and
# drops the header table itself (its content is entirely reproduced in the line).
# The CONTENT tables that follow are left untouched, so the body keeps its
# markdown. Opt-in per spec via: "preprocess": ["explode_policy_headers"]
_POLICY_NO_RX = re.compile(r'policy\s*no', re.I)
# The number cell is reliable across every variant seen (2.1, 2.7, 2.8, 2.24);
# it is the NAME cell that is unreliable, so the number is what we key on.
_POLICY_NUM_RX = re.compile(r'\d+\s*\.\s*\d+')
# The 2026 PDF's text layer breaks a word after its first letter in one heading
# ("2 .8 B ring your own device"). Rejoin a stray single letter onto the
# lowercase run that follows it, excluding "a"/"i", which are real words.
_STRAY_LETTER_RX = re.compile(r'\b(?![AaIi]\b)([A-Za-z])\s+(?=[a-z])')


# A row that is part of the header label block, not policy content. The label is
# sometimes split over two rows ("Policy" / "Name:") and the value duplicated
# across columns.
_POLICY_LABEL_RX = re.compile(r'^(policy\s*no\.?:?|policy\s*name:?|policy|name:?)?$', re.I)


def _policy_header_line(rows):
    """(number, name, header_only) for a 'Policy No.' header table, else None.

    header_only is False when the table ALSO carries policy content rows. In the
    2026 PDF the 2.9 header and its Purpose/Scope/Policy rows land in one table,
    so replacing that table wholesale would silently delete the content — the
    caller keeps the table in that case and only ADDS the heading line.
    """
    if not rows or not rows[0] or not _POLICY_NO_RX.search(rows[0][0]):
        return None
    num = None
    name_cells = []
    header_only = True
    for r in rows:
        first = (r[0] if r else '').strip()
        rest = [c for c in r[1:] if c.strip()]
        if not _POLICY_LABEL_RX.match(first):
            header_only = False          # a content row (col-0 is "1", "2", ...)
            continue
        if _POLICY_NO_RX.search(first):
            m = _POLICY_NUM_RX.search(' '.join(rest) or first)
            if m:
                num = re.sub(r'\s+', '', m.group(0))
            continue
        for c in rest:
            if c not in name_cells:
                name_cells.append(c)
    if not num:
        return None
    name = ' '.join(name_cells).strip()
    name = re.sub(r'^\d+\s*\.\s*\d+\.?\s*', '', name)      # drop the repeated number
    name = _STRAY_LETTER_RX.sub(r'\1', name)
    name = re.sub(r'\s+', ' ', name).strip(' .:')
    return num, name, header_only


def explode_policy_headers(blocks):
    out = []
    for b in blocks:
        if b.get('kind') != 'table':
            out.append(b)
            continue
        parsed = _policy_header_line(_parse_gfm(b.get('md') or ''))
        if not parsed:
            out.append(b)
            continue
        num, name, header_only = parsed
        out.append({'kind': 'line', 'n': b['n'], 'page': b['page'], 'x0': b['x0'],
                    'text': f'{num}. {name}' if name else f'{num}.', 'size': 12.0})
        if not header_only:
            out.append(b)                # content rides along in this table - keep it
    for i, blk in enumerate(out):
        blk['n'] = i + 1
    return out
