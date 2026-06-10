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

// Admin rich-text editor for a clause. Loads the current content (existing HTML,
// or a seed converted from the legacy plain-text/markdown clause), lets the admin
// format it, and saves HTML + plain text back.
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
          : <Inner clause={clause} initialHtml={html} onClose={onClose} onSaved={onSaved} />}
      </div>
    </div>
  );
}

function Inner({ clause, initialHtml, onClose, onSaved }) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const editor = useEditor({
    extensions: [
      StarterKit,
      Underline,
      TextAlign.configure({ types: ['heading', 'paragraph'] }),
      Table.configure({ resizable: true }),
      TableRow, TableHeader, TableCell,
    ],
    content: initialHtml,
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
      onSaved(r.html);
    } catch (e) { toast.error(e.message || 'Save failed'); }
    finally { setBusy(false); }
  }

  const B = ({ run, active, icon, label }) => (
    <button type="button" className={`ce-tool ${active ? 'is-active' : ''}`} title={label}
      onMouseDown={(e) => { e.preventDefault(); run(); }}><i className={`fas ${icon}`} /></button>
  );

  return (
    <>
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
            <B run={() => editor.chain().focus().addRowAfter().run()} icon="fa-grip-lines" label="Add row" />
            <B run={() => editor.chain().focus().addColumnAfter().run()} icon="fa-grip-lines-vertical" label="Add column" />
            <B run={() => editor.chain().focus().deleteTable().run()} icon="fa-eraser" label="Delete table" />
          </>
        )}
      </div>
      <div className="ce-body">
        <EditorContent editor={editor} className="clause-html ce-content" />
      </div>
      <div className="ce-foot">
        <span className="ce-hint">Formatting and tables are preserved. The original is kept in version history.</span>
        <span className="ce-actions">
          <button className="btn btn-ghost btn-sm" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-primary btn-sm" onClick={save} disabled={busy || !editor}>
            {busy ? <Spinner size={14} color="#fff" /> : <i className="fas fa-check" />} Save clause
          </button>
        </span>
      </div>
    </>
  );
}
