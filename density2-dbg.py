#!/usr/bin/env python3
"""
Density 2 Debugger module.

Provides an interactive AST debugger for the Density 2 compiler.
This module intentionally avoids importing density2_compiler at import time
to prevent circular imports. It imports it lazily inside functions.
"""

import os
import shlex
import traceback
from typing import Any, Optional, List, Dict, Callable


def _import_compiler():
    import importlib
    # Import by module name to reuse the already-loaded module if present.
    return importlib.import_module('density2_compiler')


def _is_program(obj: Any) -> bool:
    try:
        d2c = _import_compiler()
        return isinstance(obj, d2c.Program)
    except Exception:
        return False


def _ast_pretty(n: Any) -> str:
    p = getattr(n, 'pretty', None)
    if callable(p):
        try:
            return p()
        except Exception:
            pass
    return repr(n)


def _ast_dodecagram(n: Any) -> str:
    f = getattr(n, 'to_dodecagram', None)
    if callable(f):
        try:
            return f()
        except Exception:
            pass
    return ''


def _ast_encode_history(n: Any, recursive: bool = False) -> str:
    f = getattr(n, 'encode_history', None)
    if callable(f):
        try:
            return f(recursive=recursive)
        except Exception:
            pass
    return ''


def _ast_enable_tracking(n: Any, enable: bool, recursive: bool) -> None:
    f = getattr(n, 'enable_mutation_tracking', None)
    if callable(f):
        try:
            f(enable=enable, recursive=recursive)
        except Exception:
            pass


def _get_fn_name(fn: Any) -> str:
    return getattr(fn, 'name', '<unnamed>')


def _get_fn_body(fn: Any) -> List[Any]:
    return getattr(fn, 'body', []) or []


def _node_type(n: Any) -> str:
    return n.__class__.__name__


def _safe_eval(expr: str, env: Dict[str, Any]) -> Any:
    safe_globals = {"__builtins__": {}}
    safe_globals.update(env)
    return eval(expr, safe_globals, {})


class DebugSession:
    """
    Interactive debugger for Density 2 ASTs and sources.
    Commands:
      help|?                  Show help.
      quit|exit               Quit debugger.
      info                    Show AST summary and cursor.
      tree [raw|expanded]     Pretty AST (default expanded).
      funcs                   List functions.
      use <idx|name>          Select current function.
      list [count]            List statements from cursor.
      show [index]            Show statement details.
      step [n]                Move cursor by n statements.
      reset                   Reset cursor to start.
      dodecagram [raw|expanded]  Show dodecagram encoding.
      track on|off [recursive]   Toggle mutation tracking.
      history [recursive]     Show mutation history (compact).
      walk [ClassName]        Preorder walk (optional type filter).
      find <regex>            Search values in AST.
      eval <expr>             Eval limited Python against the AST.
      emit                    Print NASM output.
      write <path>            Write NASM to path.
      reload [file]           Reload from file.
      rules                   Show language ruleset summary (if available).
      dict                    Show language dictionary summary (if available).
    """
    def __init__(self, ast_or_source: Any, filename: Optional[str] = None):
        self.filename = filename
        self.source: Optional[str] = None
        self.program_raw = None
        self.program = None
        self.fn_index: int = 0
        self.stmt_index: int = 0
        self.commands: Dict[str, Callable[[List[str]], None]] = {
            'help': self.cmd_help,
            '?': self.cmd_help,
            'quit': self.cmd_quit,
            'exit': self.cmd_quit,
            'info': self.cmd_info,
            'tree': self.cmd_tree,
            'funcs': self.cmd_funcs,
            'use': self.cmd_use,
            'list': self.cmd_list,
            'show': self.cmd_show,
            'step': self.cmd_step,
            'reset': self.cmd_reset,
            'dodecagram': self.cmd_dodecagram,
            'track': self.cmd_track,
            'history': self.cmd_history,
            'walk': self.cmd_walk,
            'find': self.cmd_find,
            'eval': self.cmd_eval,
            'emit': self.cmd_emit,
            'write': self.cmd_write,
            'reload': self.cmd_reload,
            'rules': self.cmd_rules,
            'dict': self.cmd_dict,
        }
        self._initialize(ast_or_source)

    def _initialize(self, ast_or_source: Any):
        d2c = _import_compiler()
        if _is_program(ast_or_source):
            self.program = ast_or_source
            self.program_raw = None
            self.source = None
        elif isinstance(ast_or_source, str):
            self.source = ast_or_source
            tokens = d2c.tokenize(ast_or_source)
            parser = d2c.Parser(tokens)
            prog_raw = parser.parse()
            self.program_raw = prog_raw
            self.program = d2c.expand_macros(prog_raw, parser.macro_table)
        else:
            # Fallback: stringify and parse
            text = str(ast_or_source)
            self.source = text
            tokens = d2c.tokenize(text)
            parser = d2c.Parser(tokens)
            prog_raw = parser.parse()
            self.program_raw = prog_raw
            self.program = d2c.expand_macros(prog_raw, parser.macro_table)

        if not getattr(self.program, 'functions', []):
            self.fn_index = -1
            self.stmt_index = -1
        else:
            self.fn_index = 0
            self.stmt_index = 0

    def _functions(self) -> List[Any]:
        return getattr(self.program, 'functions', []) or []

    def _raw_functions(self) -> List[Any]:
        return getattr(self.program_raw, 'functions', []) or []

    def _current_fn(self) -> Optional[Any]:
        fns = self._functions()
        if 0 <= self.fn_index < len(fns):
            return fns[self.fn_index]
        return None

    def _current_stmt(self) -> Optional[Any]:
        fn = self._current_fn()
        if not fn:
            return None
        body = _get_fn_body(fn)
        if 0 <= self.stmt_index < len(body):
            return body[self.stmt_index]
        return None

    def _print(self, s: str = ''):
        print(s, flush=True)

    def _print_error(self, msg: str, ex: Optional[BaseException] = None):
        self._print(f'[error] {msg}')
        if ex:
            traceback.print_exc()

    def repl(self):
        self._print('Density 2 Debugger. Type "help" for commands. "quit" to exit.')
        while True:
            try:
                line = input('(den2) ').strip()
            except (EOFError, KeyboardInterrupt):
                self._print()
                break
            if not line:
                continue
            try:
                parts = shlex.split(line)
            except ValueError as ex:
                self._print_error('Parse error', ex)
                continue
            if not parts:
                continue
            cmd, *args = parts
            fn = self.commands.get(cmd.lower())
            if fn is None:
                self._print(f'Unknown command: {cmd}. Try "help".')
                continue
            try:
                fn(args)
            except SystemExit:
                raise
            except Exception as ex:
                self._print_error('Command failed', ex)

    # Commands
    def cmd_help(self, args: List[str]):
        for ln in self.__class__.__doc__.splitlines():
            if ln.strip():
                self._print(ln)

    def cmd_quit(self, args: List[str]):
        raise SystemExit(0)

    def cmd_info(self, args: List[str]):
        fns = self._functions()
        raw_fns = self._raw_functions()
        self._print(f'File: {self.filename or "<memory>"}')
        self._print(f'Functions (expanded): {len(fns)} | (raw): {len(raw_fns)}')
        if 0 <= self.fn_index < len(fns):
            fn = fns[self.fn_index]
            body = _get_fn_body(fn)
            self._print(f'Cursor: fn[{self.fn_index}]={_get_fn_name(fn)} stmt[{self.stmt_index}] of {len(body)}')
        else:
            self._print('Cursor: <no function>')

    def cmd_tree(self, args: List[str]):
        which = (args[0].lower() if args else 'expanded')
        if which == 'raw':
            if self.program_raw is None:
                self._print('<no raw AST available>')
                return
            self._print(_ast_pretty(self.program_raw))
        else:
            self._print(_ast_pretty(self.program))

    def cmd_funcs(self, args: List[str]):
        for i, f in enumerate(self._functions()):
            self._print(f'[{i}] {_get_fn_name(f)}')

    def cmd_use(self, args: List[str]):
        if not args:
            self._print('usage: use <idx|name>')
            return
        key = args[0]
        fns = self._functions()
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < len(fns):
                self.fn_index = idx
                self.stmt_index = 0
                self.cmd_info([])
            else:
                self._print('index out of range')
            return
        for i, f in enumerate(fns):
            if _get_fn_name(f) == key:
                self.fn_index = i
                self.stmt_index = 0
                self.cmd_info([])
                return
        self._print('function not found')

    def cmd_list(self, args: List[str]):
        count = int(args[0]) if args and args[0].isdigit() else 10
        fn = self._current_fn()
        if not fn:
            self._print('<no function selected>')
            return
        body = _get_fn_body(fn)
        start = max(0, self.stmt_index)
        end = min(len(body), start + count)
        for i in range(start, end):
            st = body[i]
            ty = _node_type(st)
            self._print(f'{i:4}: {ty} {repr(st)}')

    def cmd_show(self, args: List[str]):
        fn = self._current_fn()
        if not fn:
            self._print('<no function selected>')
            return
        body = _get_fn_body(fn)
        idx = self.stmt_index
        if args and args[0].isdigit():
            idx = int(args[0])
        if not (0 <= idx < len(body)):
            self._print('index out of range')
            return
        st = body[idx]
        self._print(f'[{idx}] type={_node_type(st)}')
        self._print(_ast_pretty(st))

    def cmd_step(self, args: List[str]):
        n = int(args[0]) if args and args[0].lstrip('-').isdigit() else 1
        fn = self._current_fn()
        if not fn:
            self._print('<no function selected>')
            return
        body = _get_fn_body(fn)
        self.stmt_index = max(0, min(len(body) - 1, self.stmt_index + n))
        self.cmd_show([])

    def cmd_reset(self, args: List[str]):
        self.fn_index = 0 if self._functions() else -1
        self.stmt_index = 0
        self.cmd_info([])

    def cmd_dodecagram(self, args: List[str]):
        which = (args[0].lower() if args else 'expanded')
        if which == 'raw':
            if self.program_raw is None:
                self._print('<no raw AST>')
            else:
                self._print(_ast_dodecagram(self.program_raw))
            return
        self._print(_ast_dodecagram(self.program))

    def cmd_track(self, args: List[str]):
        if not args:
            self._print('usage: track on|off [recursive]')
            return
        mode = args[0].lower()
        recursive = (len(args) > 1 and args[1].lower() in ('1', 'true', 'yes', 'y', 'rec', 'recursive', 'on'))
        if mode not in ('on', 'off'):
            self._print('usage: track on|off [recursive]')
            return
        _ast_enable_tracking(self.program, enable=(mode == 'on'), recursive=recursive)
        self._print(f'mutation tracking: {mode} (recursive={recursive})')

    def cmd_history(self, args: List[str]):
        recursive = (len(args) > 0 and args[0].lower() in ('1', 'true', 'yes', 'y', 'rec', 'recursive', 'on'))
        enc = _ast_encode_history(self.program, recursive=recursive)
        if enc:
            self._print('history (compact):')
            self._print(enc)
        else:
            self._print('<no history>')

    def cmd_walk(self, args: List[str]):
        type_filter = args[0] if args else None

        def walk(n: Any):
            w = getattr(n, 'walk', None)
            if callable(w):
                for it in w():
                    yield it
                return
            yield n

        count = 0
        for node in walk(self.program):
            if type_filter and _node_type(node) != type_filter:
                continue
            self._print(f'{_node_type(node)} | {repr(node)}')
            count += 1
            if count > 1000:
                self._print('... truncated (1000 shown)')
                break

    def cmd_find(self, args: List[str]):
        if not args:
            self._print('usage: find <regex>')
            return
        import re
        pat = re.compile(args[0])

        def iter_items(n: Any):
            it = getattr(n, '_iter_fields', None)
            if callable(it):
                for k, v in it():
                    yield k, v

        hits = 0
        walker = getattr(self.program, 'walk', None)
        iterable = walker() if callable(walker) else [self.program]
        for node in iterable:
            for k, v in (iter_items(node) or []):
                if isinstance(v, str):
                    if pat.search(v):
                        self._print(f'{_node_type(node)}.{k}: {v!r}')
                        hits += 1
                elif not hasattr(v, 'walk'):
                    s = repr(v)
                    if pat.search(s):
                        self._print(f'{_node_type(node)}.{k}: {s}')
                        hits += 1
        if hits == 0:
            self._print('<no matches>')

    def cmd_eval(self, args: List[str]):
        if not args:
            self._print('usage: eval <expr>')
            return
        expr = ' '.join(args)
        env = {
            'program': self.program,
            'program_raw': self.program_raw,
            'fn': self._current_fn(),
            'stmt': self._current_stmt(),
            'len': len,
            'type': type,
            'repr': repr,
            'str': str,
        }
        try:
            out = _safe_eval(expr, env)
            self._print(repr(out))
        except Exception as ex:
            self._print_error('eval failed', ex)

    def cmd_emit(self, args: List[str]):
        try:
            d2c = _import_compiler()
            asm = d2c.generate_nasm(self.program)
            self._print(asm)
        except Exception as ex:
            self._print_error('emit failed', ex)

    def cmd_write(self, args: List[str]):
        if not args:
            self._print('usage: write <path>')
            return
        path = args[0]
        try:
            d2c = _import_compiler()
            asm = d2c.generate_nasm(self.program)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(asm)
            self._print(f'written: {path}')
        except Exception as ex:
            self._print_error('write failed', ex)

    def cmd_reload(self, args: List[str]):
        path = args[0] if args else self.filename
        if not path:
            self._print('usage: reload <file>')
            return
        if not os.path.exists(path):
            self._print(f'file not found: {path}')
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.filename = path
            self._initialize(text)
            self._print(f'reloaded: {path}')
            self.cmd_info([])
        except Exception as ex:
            self._print_error('reload failed', ex)

    def cmd_rules(self, args: List[str]):
        try:
            d2c = _import_compiler()
            if not hasattr(d2c, 'get_ruleset'):
                self._print('<ruleset not available>')
                return
            rs = d2c.get_ruleset()
            self._print('Ruleset summary:')
            self._print(f"- tokens: {len(rs.get('tokens', []))}")
            self._print(f"- keywords: {len(rs.get('keywords', []))}")
            self._print(f"- inline_languages: {len(rs.get('inline_languages', []))}")
            self._print(f"- nasm_directives: {len(rs.get('nasm_directives', []))}")
            self._print(f"- registers_x86_64: {len(rs.get('registers_x86_64', []))}")
            self._print('Statement syntax:')
            for k, v in rs.get('statement_syntax', {}).items():
                self._print(f"  {k}: {v}")
        except Exception as ex:
            self._print_error('rules failed', ex)

    def cmd_dict(self, args: List[str]):
        try:
            d2c = _import_compiler()
            if not hasattr(d2c, 'get_dictionary'):
                self._print('<dictionary not available>')
                return
            di = d2c.get_dictionary()
            self._print('Dictionary summary:')
            self._print(f"- syscalls_amd64_linux: {len(di.get('syscalls_amd64_linux', {}))}")
            self._print(f"- inline_languages: {len(di.get('inline_languages', []))}")
            self._print(f"- nasm_directives: {len(di.get('nasm_directives', []))}")
            self._print(f"- registers_x86_64: {len(di.get('registers_x86_64', []))}")
        except Exception as ex:
            self._print_error('dict failed', ex)


def start_debugger(ast_or_source: Any, filename: Optional[str] = None) -> None:
    """
    Launch the interactive debugger.
    - ast_or_source: Program AST or Density 2 source string.
    - filename: Optional file path (display only).
    """
    sess = DebugSession(ast_or_source, filename=filename)
    sess.repl()


__all__ = ['start_debugger', 'DebugSession']
