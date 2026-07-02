import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Extension } from '@tiptap/core';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Underline from '@tiptap/extension-underline';
import TextAlign from '@tiptap/extension-text-align';
import { Table } from '@tiptap/extension-table';
import TableRow from '@tiptap/extension-table-row';
import TableHeader from '@tiptap/extension-table-header';
import TableCell from '@tiptap/extension-table-cell';
import { TextStyle, FontFamily, FontSize } from '@tiptap/extension-text-style';
import { Color } from '@tiptap/extension-color';
import { Spinner } from './UI.jsx';
import { useToast } from './Toast.jsx';
import { api } from '../api.js';
import './clauseEditor.css';

// Add a vertical-align attribute to table cells/headers (top / middle / bottom),
// rendered as an inline style so it round-trips through the stored HTML.
const vAlignAttr = {
  verticalAlign: {
    default: null,
    parseHTML: (el) => el.style.verticalAlign || null,
    renderHTML: (attrs) => (attrs.verticalAlign ? { style: `vertical-align: ${attrs.verticalAlign}` } : {}),
  },
};
const CellWithVAlign = TableCell.extend({ addAttributes() { return { ...this.parent?.(), ...vAlignAttr }; } });
const HeaderWithVAlign = TableHeader.extend({ addAttributes() { return { ...this.parent?.(), ...vAlignAttr }; } });

// Block indentation: a margin-left step on paragraphs/headings, round-tripped as
// an inline style. indent()/outdent() bump the level on the current block.
const INDENT_STEP = 26;   // px per level
const Indent = Extension.create({
  name: 'indent',
  addOptions() { return { types: ['paragraph', 'heading'], max: 10 }; },
  addGlobalAttributes() {
    return [{
      types: this.options.types,
      attributes: {
        indent: {
          default: 0,
          parseHTML: (el) => Math.round((parseInt(el.style.marginLeft, 10) || 0) / INDENT_STEP) || 0,
          renderHTML: (attrs) => (attrs.indent ? { style: `margin-left: ${attrs.indent * INDENT_STEP}px` } : {}),
        },
      },
    }];
  },
  addCommands() {
    const bump = (dir) => () => ({ editor, commands }) => {
      const type = editor.isActive('heading') ? 'heading' : 'paragraph';
      const cur = editor.getAttributes(type).indent || 0;
      const next = Math.max(0, Math.min(this.options.max, cur + dir));
      return commands.updateAttributes(type, { indent: next });
    };
    return { indent: bump(1), outdent: bump(-1) };
  },
});

// The editor surface is white-space:pre-wrap, so typed alignment gaps (e.g. a
// signature block with columns spaced apart) are visible while editing — but a
// normal HTML render collapses runs of spaces, gluing the words together. Walk
// only the text nodes and convert each run of 2+ spaces to non-breaking spaces
// so the gap survives rendering. Tags/attributes are never touched.
export function preserveSpaces(html) {
  const doc = new DOMParser().parseFromString(`<body>${html}</body>`, 'text/html');
  const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT);
  let n;
  // eslint-disable-next-line no-cond-assign
  while ((n = walker.nextNode())) {
    if (/ {2,}/.test(n.nodeValue)) {
      n.nodeValue = n.nodeValue.replace(/ {2,}/g, (run) => ' '.repeat(run.length));
    }
  }
  return doc.body.innerHTML;
}

export const EXTENSIONS = [
  StarterKit,
  Underline,
  TextAlign.configure({ types: ['heading', 'paragraph'] }),
  // Wider grab handle + a resizable last column make column-edge dragging reliable.
  Table.configure({ resizable: true, handleWidth: 8, cellMinWidth: 40, lastColumnResizable: true }),
  TableRow, HeaderWithVAlign, CellWithVAlign,
  TextStyle, Color, FontFamily, FontSize, Indent,
];

// Font choices for the editor toolbars.
export const FONT_FAMILIES = [
  { label: 'Font', value: '' },
  { label: 'Serif', value: 'Georgia, "Times New Roman", serif' },
  { label: 'Sans', value: 'Inter, Arial, sans-serif' },
  { label: 'Mono', value: 'ui-monospace, "Courier New", monospace' },
  { label: 'Times', value: '"Times New Roman", serif' },
  { label: 'Arial', value: 'Arial, sans-serif' },
];
export const FONT_SIZES = ['', '11px', '12px', '13px', '14px', '16px', '18px', '20px', '24px'];

// Font family + size dropdowns shared by both toolbars (Studio + modal).
export function FontControls({ editor }) {
  const fam = editor.getAttributes('textStyle').fontFamily || '';
  const size = editor.getAttributes('textStyle').fontSize || '';
  return (
    <>
      <select className="ce-select" title="Font family" value={fam}
        onChange={(e) => { const v = e.target.value; const c = editor.chain().focus(); (v ? c.setFontFamily(v) : c.unsetFontFamily()).run(); }}>
        {FONT_FAMILIES.map((f) => <option key={f.label} value={f.value}>{f.label}</option>)}
      </select>
      <select className="ce-select ce-select-sm" title="Font size" value={size}
        onChange={(e) => { const v = e.target.value; const c = editor.chain().focus(); (v ? c.setFontSize(v) : c.unsetFontSize()).run(); }}>
        <option value="">Size</option>
        {FONT_SIZES.filter(Boolean).map((s) => <option key={s} value={s}>{parseInt(s, 10)}</option>)}
      </select>
    </>
  );
}

// Text colours offered in the editor (label + value). "Default" clears the
// inline colour so the text inherits the clause styling (fixes pasted teal text).
export const TEXT_COLORS = [
  { label: 'Default', value: null },
  { label: 'Navy', value: '#1a237e' },
  { label: 'Black', value: '#1e293b' },
  { label: 'Teal', value: '#0d9488' },
  { label: 'Red', value: '#dc2626' },
  { label: 'Amber', value: '#b45309' },
];

// --- Format painter: capture the formatting at the caret/selection, then apply
// it verbatim to another selection (like Word's format painter). ---
export function captureFormat(editor) {
  if (!editor) return null;
  const align = ['left', 'center', 'right', 'justify'].find((a) => editor.isActive({ textAlign: a }));
  const ts = editor.getAttributes('textStyle');
  return {
    bold: editor.isActive('bold'),
    italic: editor.isActive('italic'),
    underline: editor.isActive('underline'),
    color: ts.color || null,
    fontFamily: ts.fontFamily || null,
    fontSize: ts.fontSize || null,
    align: align || null,
  };
}
export function applyFormat(editor, fmt) {
  if (!editor || !fmt) return;
  const c = editor.chain().focus();
  fmt.bold ? c.setBold() : c.unsetBold();
  fmt.italic ? c.setItalic() : c.unsetItalic();
  fmt.underline ? c.setUnderline() : c.unsetUnderline();
  fmt.color ? c.setColor(fmt.color) : c.unsetColor();
  fmt.fontFamily ? c.setFontFamily(fmt.fontFamily) : c.unsetFontFamily();
  fmt.fontSize ? c.setFontSize(fmt.fontSize) : c.unsetFontSize();
  if (fmt.align) c.setTextAlign(fmt.align);
  c.run();
}

// Compact text-colour picker (palette button + swatch popover) for the toolbars.
export function ColorPicker({ editor }) {
  const [open, setOpen] = useState(false);
  const pick = (v) => {
    const ch = editor.chain().focus();
    (v ? ch.setColor(v) : ch.unsetColor()).run();
    setOpen(false);
  };
  return (
    <span className="ce-color">
      <button type="button" className="ce-tool" title="Text colour"
        onMouseDown={(e) => { e.preventDefault(); setOpen((o) => !o); }}><i className="fas fa-palette" /></button>
      {open && (
        <span className="ce-color-pop" onMouseLeave={() => setOpen(false)}>
          {TEXT_COLORS.map((c) => (
            <button type="button" key={c.label} className="ce-swatch" title={c.label}
              style={{ background: c.value || '#fff' }}
              onMouseDown={(e) => { e.preventDefault(); pick(c.value); }}>
              {!c.value && <i className="fas fa-ban" />}
            </button>
          ))}
        </span>
      )}
    </span>
  );
}

// Merge the table the cursor is in with the next table (only an empty
// paragraph may sit between them). Rows are concatenated; shorter rows are
// padded with empty cells so the combined grid stays valid. Returns false
// (no-op) if the cursor isn't in a table or there's no joinable table below.
export function joinTableBelow(editor) {
  if (!editor) return false;
  const { state } = editor;
  const { doc, selection } = state;
  const tables = [];
  doc.descendants((node, pos) => {
    if (node.type.name === 'table') { tables.push({ node, pos, end: pos + node.nodeSize }); return false; }
    return true;
  });
  if (tables.length < 2) return false;
  let ai = tables.findIndex((t) => selection.from >= t.pos && selection.from <= t.end);
  if (ai === -1) ai = tables.filter((t) => t.pos <= selection.from).length - 1; // nearest table above the caret
  if (ai < 0 || ai + 1 >= tables.length) return false;
  const A = tables[ai], B = tables[ai + 1];
  if (doc.textBetween(A.end, B.pos, '', '').trim() !== '') return false; // only blank space allowed between

  const rows = [];
  A.node.forEach((r) => rows.push(r));
  B.node.forEach((r) => rows.push(r));
  const colsOf = (row) => { let c = 0; row.forEach((cell) => { c += cell.attrs.colspan || 1; }); return c; };
  const maxCols = Math.max(...rows.map(colsOf));
  const cellType = state.schema.nodes.tableCell;
  const fixed = rows.map((row) => {
    const deficit = maxCols - colsOf(row);
    if (deficit <= 0 || !cellType) return row;
    const cells = [];
    row.forEach((c) => cells.push(c));
    for (let i = 0; i < deficit; i += 1) cells.push(cellType.createAndFill());
    return row.type.create(row.attrs, cells);
  });
  const merged = A.node.type.create(A.node.attrs, fixed);
  // No .scrollIntoView() — the merged table stays where it was; forcing the view
  // to the change made the whole editor jump on every merge.
  editor.view.dispatch(state.tr.replaceWith(A.pos, B.end, merged));
  return true;
}

// Reusable editor core: toolbar + editable area + save. Used inline by the
// Document Studio module and inside the modal below.
// Remove only Word/Office structural cruft (conditional comments, <o:p>, w: tags)
// so a pasted table parses — but KEEP inline styles so bold/italic/alignment from
// the original document survive (ProseMirror drops styles it doesn't understand).
export function cleanPastedHTML(html) {
  let out = html;
  if (/mso-|MsoNormal|schemas-microsoft|<o:p/i.test(out)) {
    out = out
      .replace(/<!--[\s\S]*?-->/g, '')
      .replace(/<\/?o:p[^>]*>/gi, '')
      .replace(/<\/?w:[^>]*>/gi, '');
  }
  // Strip explicit text colours. PDF/Office copy often carries a white or
  // near-invisible colour, so pasted text shows up blank until manually
  // recoloured. Removing `color` (but not `background-color`) lets pasted text
  // inherit the clause's own colour. Other styles (bold/italic/align) are kept.
  out = out.replace(/(?<![\w-])color\s*:\s*[^;"']*;?/gi, '');
  return out;
}

export function ClauseEditorPanel({ clause, initialHtml, onSaved, onCancel }) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const editor = useEditor({
    extensions: EXTENSIONS,
    content: initialHtml || '<p></p>',
    editorProps: { transformPastedHTML: cleanPastedHTML },
  });

  async function save() {
    if (!editor) return;
    setBusy(true);
    try {
      const r = await api.post('/clause/edit', {
        id: clause.id, source: clause.source,
        html: preserveSpaces(editor.getHTML()), text: editor.getText(),
      });
      toast.success('Clause saved');
      onSaved?.(r.html);
    } catch (e) { toast.error(e.message || 'Save failed'); }
    finally { setBusy(false); }
  }

  const B = ({ run, active, icon, label }) => (
    <button type="button" className={`ce-tool ${active ? 'is-active' : ''}`} title={label}
      onMouseDown={(e) => { e.preventDefault(); run(); }}><i className={`fas ${icon}`} /></button>
  );

  return (
    <div className="ce-panel">
      <div className="ce-toolbar">
        {editor && (
          <>
            <B run={() => editor.chain().focus().toggleBold().run()} active={editor.isActive('bold')} icon="fa-bold" label="Bold" />
            <B run={() => editor.chain().focus().toggleItalic().run()} active={editor.isActive('italic')} icon="fa-italic" label="Italic" />
            <B run={() => editor.chain().focus().toggleUnderline().run()} active={editor.isActive('underline')} icon="fa-underline" label="Underline" />
            <span className="ce-divide" />
            <B run={() => editor.chain().focus().setTextAlign('left').run()} active={editor.isActive({ textAlign: 'left' })} icon="fa-align-left" label="Align left" />
            <B run={() => editor.chain().focus().setTextAlign('center').run()} active={editor.isActive({ textAlign: 'center' })} icon="fa-align-center" label="Center" />
            <B run={() => editor.chain().focus().setTextAlign('right').run()} active={editor.isActive({ textAlign: 'right' })} icon="fa-align-right" label="Align right" />
            <B run={() => editor.chain().focus().setTextAlign('justify').run()} active={editor.isActive({ textAlign: 'justify' })} icon="fa-align-justify" label="Justify" />
            <ColorPicker editor={editor} />
            <FontControls editor={editor} />
            <span className="ce-divide" />
            <B run={() => editor.chain().focus().toggleBulletList().run()} active={editor.isActive('bulletList')} icon="fa-list-ul" label="Bulleted list" />
            <B run={() => editor.chain().focus().toggleOrderedList().run()} active={editor.isActive('orderedList')} icon="fa-list-ol" label="Numbered list" />
            <B run={() => editor.chain().focus().outdent().run()} icon="fa-outdent" label="Decrease indent" />
            <B run={() => editor.chain().focus().indent().run()} icon="fa-indent" label="Increase indent" />
            <span className="ce-divide" />
            <B run={() => editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()} icon="fa-table" label="Insert table" />
            <B run={() => editor.chain().focus().toggleHeaderRow().run()} icon="fa-heading" label="Toggle header row (bold first row on/off)" />
            <B run={() => editor.chain().focus().addRowAfter().run()} icon="fa-grip-lines" label="Add row below" />
            <B run={() => editor.chain().focus().addColumnAfter().run()} icon="fa-grip-lines-vertical" label="Add column right" />
            <B run={() => editor.chain().focus().deleteRow().run()} icon="fa-delete-left" label="Delete row" />
            <B run={() => editor.chain().focus().deleteColumn().run()} icon="fa-delete-left fa-rotate-90" label="Delete column" />
            <B run={() => editor.chain().focus().mergeCells().run()} icon="fa-object-group" label="Merge selected cells" />
            <B run={() => editor.chain().focus().splitCell().run()} icon="fa-object-ungroup" label="Split cell" />
            <B run={() => joinTableBelow(editor) || toast.error('Put the cursor in the upper table — the one directly below it will be joined on.')} icon="fa-down-left-and-up-right-to-center" label="Join with table below" />
            <B run={() => editor.chain().focus().deleteTable().run()} icon="fa-trash-can" label="Delete whole table" />
            <span className="ce-divide" />
            <B run={() => editor.chain().focus().setCellAttribute('verticalAlign', 'top').run()} icon="fa-angle-up" label="Cell align top" />
            <B run={() => editor.chain().focus().setCellAttribute('verticalAlign', 'middle').run()} icon="fa-equals" label="Cell align middle" />
            <B run={() => editor.chain().focus().setCellAttribute('verticalAlign', 'bottom').run()} icon="fa-angle-down" label="Cell align bottom" />
          </>
        )}
      </div>
      <div className="ce-body">
        <EditorContent editor={editor} className="clause-html ce-content" />
      </div>
      <div className="ce-foot">
        <span className="ce-hint">Formatting + tables preserved. The previous version is kept in history.</span>
        <span className="ce-actions">
          {onCancel && <button className="btn btn-ghost btn-sm" onClick={onCancel} disabled={busy}>Cancel</button>}
          <button className="btn btn-primary btn-sm" onClick={save} disabled={busy || !editor}>
            {busy ? <Spinner size={14} color="#fff" /> : <i className="fas fa-check" />} Save clause
          </button>
        </span>
      </div>
    </div>
  );
}

// Modal wrapper (used by the inline "Edit clause" in search).
export default function ClauseEditorModal({ clause, onClose, onSaved }) {
  const toast = useToast();
  const [html, setHtml] = useState(null);

  useEffect(() => {
    api.get(`/clause/edit?id=${encodeURIComponent(clause.id)}&source=${encodeURIComponent(clause.source)}`)
      .then((d) => setHtml(d.html || '<p></p>'))
      .catch((e) => { toast.error(e.message || 'Could not open editor'); onClose(); });
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  // Portal to <body> so the fixed-position overlay is centred on the viewport
  // rather than trapped inside a transformed/animated results ancestor.
  return createPortal(
    <div className="ce-overlay" onMouseDown={onClose}>
      <div className="ce-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="ce-head">
          <div>
            <div className="ce-title">Edit clause</div>
            <div className="ce-sub">{clause.source} · {clause.id}</div>
          </div>
          <button className="ce-close" onClick={onClose} aria-label="Close">&times;</button>
        </div>
        {html === null
          ? <div className="ce-loading"><Spinner size={16} /> Loading…</div>
          : <ClauseEditorPanel clause={clause} initialHtml={html} onSaved={onSaved} onCancel={onClose} />}
      </div>
    </div>,
    document.body,
  );
}
