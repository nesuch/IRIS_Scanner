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
