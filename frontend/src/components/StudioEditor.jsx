import { useEffect } from 'react';
import { useEditor, EditorContent } from '@tiptap/react';
import { DOMSerializer } from '@tiptap/pm/model';
import { EXTENSIONS, cleanPastedHTML, joinTableBelow } from './ClauseEditor.jsx';

// Split the current document at the caret into two HTML fragments.
function splitAtCursor(editor) {
  const { state } = editor;
  const pos = state.selection.from;
  const doc = state.doc;
  const ser = DOMSerializer.fromSchema(editor.schema);
  const toHtml = (frag) => { const div = document.createElement('div'); div.appendChild(ser.serializeFragment(frag.content)); return div.innerHTML || '<p></p>'; };
  return { before: toHtml(doc.cut(0, pos)), after: toHtml(doc.cut(pos, doc.content.size)) };
}

// Content editor for the active clause. Reports edits via onChange and exposes
// splitAtCursor through apiRef so the Studio action bar can split at the caret.
export default function StudioEditor({ value, onChange, apiRef }) {
  const editor = useEditor({
    extensions: EXTENSIONS,
    content: value || '<p></p>',
    editorProps: { transformPastedHTML: cleanPastedHTML },
    onUpdate: ({ editor: ed }) => onChange?.(ed.getHTML()),
  });

  useEffect(() => {
    if (apiRef) apiRef.current = editor ? { split: () => splitAtCursor(editor) } : null;
    return () => { if (apiRef) apiRef.current = null; };
  }, [editor, apiRef]);

  const B = ({ run, active, icon, label }) => (
    <button type="button" className={`ce-tool ${active ? 'is-active' : ''}`} title={label}
      onMouseDown={(e) => { e.preventDefault(); run(); }}><i className={`fas ${icon}`} /></button>
  );
  if (!editor) return null;

  return (
    <div className="ce-panel">
      <div className="ce-toolbar">
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
        <B run={() => editor.chain().focus().mergeCells().run()} icon="fa-object-group" label="Merge cells" />
        <B run={() => editor.chain().focus().splitCell().run()} icon="fa-object-ungroup" label="Split cell" />
        <B run={() => joinTableBelow(editor)} icon="fa-down-left-and-up-right-to-center" label="Join with table below (cursor in upper table)" />
        <B run={() => editor.chain().focus().deleteTable().run()} icon="fa-trash-can" label="Delete whole table" />
        <span className="ce-divide" />
        <B run={() => editor.chain().focus().setCellAttribute('verticalAlign', 'top').run()} icon="fa-angle-up" label="Cell align top" />
        <B run={() => editor.chain().focus().setCellAttribute('verticalAlign', 'middle').run()} icon="fa-equals" label="Cell align middle" />
        <B run={() => editor.chain().focus().setCellAttribute('verticalAlign', 'bottom').run()} icon="fa-angle-down" label="Cell align bottom" />
      </div>
      <div className="ce-body">
        <EditorContent editor={editor} className="clause-html ce-content" />
      </div>
    </div>
  );
}
