// Slash-command parsing, shared by the clause search and the PQ search.
//
// The two pages offer different commands over different corpora (a clause has an
// id like PPHI-S1-GEN-II-4; a PQ has a number like 9000), so the command list is a
// parameter rather than a constant here.

export const CLAUSE_SLASH_COMMANDS = [
  { cmd: 'clause', label: 'Clause number', icon: 'fa-hashtag',
    desc: 'Jump to a clause by its ID — e.g. /clause PPHI-S1-GEN-II-4' },
  { cmd: 'deep', label: 'Deep scan', icon: 'fa-wave-square',
    desc: 'Scan full clause text for your exact terms (no concept expansion)' },
];

export const PQ_SLASH_COMMANDS = [
  { cmd: 'num', label: 'PQ number', icon: 'fa-hashtag',
    desc: 'Jump to a reply by its question number — e.g. /num 9000' },
  { cmd: 'deep', label: 'Deep scan', icon: 'fa-wave-square',
    desc: 'Scan full reply text for your exact terms (no concept expansion)' },
];

// Parse a "/..." query. Returns {kind:'menu',list,word} while a command is still
// being chosen, {kind:'cmd',cmd,arg} once a command + space is committed, or null
// for a bare id lookup (e.g. "/64", "/PPHI-S1-GEN-II-4", "/9000") — back-compat.
// A bare number returns null because no command starts with a digit, which is what
// keeps the pre-existing "/9000" shortcut working on both pages.
export function parseSlash(v, commands) {
  const s = (v || '').trimStart();
  if (!s.startsWith('/')) return null;
  const sp = s.indexOf(' ');
  if (sp === -1) {
    const word = s.slice(1).toLowerCase();
    const list = commands.filter((c) => c.cmd.startsWith(word));
    if (word && list.length === 0) return null;   // an id, not a command
    return { kind: 'menu', list, word };
  }
  const word = s.slice(1, sp).toLowerCase();
  const exact = commands.find((c) => c.cmd === word);
  return exact ? { kind: 'cmd', cmd: exact.cmd, arg: s.slice(sp + 1) } : null;
}
