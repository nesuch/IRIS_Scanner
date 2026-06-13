import { useEffect, useState } from 'react';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Underline from '@tiptap/extension-underline';
import TextAlign from '@tiptap/extension-text-align';
import { Table } from '@tiptap/extension-table';
import TableRow from '@tiptap/extension-table-row';
import TableHeader from '@tiptap/extension-table-header';
import TableCell from '@tiptap/extension-table-cell';
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

export const EXTENSIONS = [
  StarterKit,
  Underline,
  TextAlign.configure({ types: ['heading', 'paragraph'] }),
  Table.configure({ resizable: true }),
  TableRow, HeaderWithVAlign, CellWithVAlign,
];

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
  editor.view.dispatch(state.tr.replaceWith(A.pos, B.end, merged).scrollIntoView());
  return true;
}

// Reusable editor core: toolbar + editable area + save. Used inline by the
// Document Studio module and inside the modal below.
// Remove only Word/Office structural cruft (conditional comments, <o:p>, w: tags)
// so a pasted table parses — but KEEP inline styles so bold/italic/alignment from
// the original document survive (ProseMirror drops styles it doesn't understand).
export function cleanPastedHTML(html) {
  if (!/mso-|MsoNormal|schemas-microsoft|<o:p/i.test(html)) return html;
  return html
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<\/?o:p[^>]*>/gi, '')
    .replace(/<\/?w:[^>]*>/gi, '');
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
        html: editor.getHTML(), text: editor.getText(),
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
            <B run={() => editor.chain().focus().setTextAlign('justify').run()} active={editor.isActive({ textAlign: 'justify' })} icon="fa-align-justify" label="Justify" />
            <span className="ce-divide" />
            <B run={() => editor.chain().focus().toggleBulletList().run()} active={editor.isActive('bulletList')} icon="fa-list-ul" label="Bulleted list" />
            <B run={() => editor.chain().focus().toggleOrderedList().run()} active={editor.isActive('orderedList')} icon="fa-list-ol" label="Numbered list" />
            <span className="ce-divide" />
            <B run={() => editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()} icon="fa-table" label="Insert table" />
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

  return (
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
    </div>
  );
}
