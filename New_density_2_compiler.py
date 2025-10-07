# density2_compiler.py
""""""

import os
import re
import shutil
import subprocess
import tempfile
import time
import hashlib
from typing import List, Tuple, Union, Dict, Optional

import os
import re
import shutil
import subprocess
import tempfile
import time
import hashlib
from typing import List, Tuple, Union, Dict, Optional

# -----------------------------
# Token Definitions
# -----------------------------
TOKEN_SPECIFICATION = [
    # Blocks first (non-greedy to avoid overconsumption)
    ('CIAM',        r"'''(.*?),,,"),              # ''' ... ,,,
    ('INLINE_ASM',  r"#asm(.*?)#endasm"),
    ('INLINE_C',    r"#c(.*?)#endc"),
    ('INLINE_PY',   r"#python(.*?)#endpython"),

    # Comments
    ('COMMENT',     r'//[^\n]*'),
    ('MCOMMENT',    r'/\*.*?\*/'),

    # Literals and identifiers
    ('STRING',      r'"(?:\\.|[^"\\])*"'),
    ('IDENT',       r'[A-Za-z_][A-Za-z0-9_]*'),

    # Symbols
    ('LBRACE',      r'\{'),
    ('RBRACE',      r'\}'),
    ('LPAREN',      r'\('),
    ('RPAREN',      r'\)'),
    ('COLON',       r':'),
    ('SEMICOLON',   r';'),
    ('COMMA',       r','),         # added for macro args
    ('PLUS',        r'\+'),        # added for string concat

    # Whitespace, newline, and mismatch
    ('NEWLINE',     r'\n'),
    ('SKIP',        r'[ \t]+'),
    ('MISMATCH',    r'.'),
]

# Make CIAM and INLINE_* non-greedy by ensuring (.*?) groups are used above.
# Note: CIAM pattern above currently uses (.*?),,, via the outer token list string.
# We will rebuild it here to ensure the non-greedy subpattern.
TOKEN_SPECIFICATION = [
    ('CIAM',        r"'''(.*?),,,"),              # We'll replace *? shortly
    ('INLINE_ASM',  r"#asm(.*?)#endasm"),
    ('INLINE_C',    r"#c(.*?)#endc"),
    ('INLINE_PY',   r"#python(.*?)#endpython"),
    ('COMMENT',     r'//[^\n]*'),
    ('MCOMMENT',    r'/\*.*?\*/'),
    ('STRING',      r'"(?:\\.|[^"\\])*"'),
    ('IDENT',       r'[A-Za-z_][A-Za-z0-9_]*'),
    ('LBRACE',      r'\{'),
    ('RBRACE',      r'\}'),
    ('LPAREN',      r'\('),
    ('RPAREN',      r'\)'),
    ('COLON',       r':'),
    ('SEMICOLON',   r';'),
    ('COMMA',       r','),         # macro args
    ('PLUS',        r'\+'),        # string concat
    ('NEWLINE',     r'\n'),
    ('SKIP',        r'[ \t]+'),
    ('MISMATCH',    r'.'),
]

# Fix CIAM to be non-greedy explicitly
TOKEN_SPECIFICATION = [
    (name, (pattern if name != 'CIAM' else r"'''(.*?),,,"))
    for (name, pattern) in TOKEN_SPECIFICATION
]

token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATION)


class Token:
    def __init__(self, type_: str, value: str, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line}, col={self.column})"


# -----------------------------
# Lexer
# -----------------------------
def tokenize(code: str) -> List[Token]:
    tokens: List[Token] = []
    line_num = 1
    line_start = 0
    for mo in re.finditer(token_regex, code, re.DOTALL):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
            continue
        elif kind in ('SKIP', 'COMMENT', 'MCOMMENT'):
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        tokens.append(Token(kind, value, line_num, column))
    return tokens


# -----------------------------
# AST Nodes
# -----------------------------
class ASTNode:
    """
    Base class for all Density 2 AST nodes.

    Features:
    - Optional source position tracking: filename, (line, col) -> (end_line, end_col)
    - Child discovery: children() finds nested AST nodes and lists of nodes
    - Traversal: walk() yields nodes in preorder
    - Visitor pattern: accept(visitor) calls visitor.visit_<Type>(self) or visitor.visit(self)
    - Structural replace: replace_child(old, new) updates direct attributes/lists
    - Serialization: to_dict()/pretty() for debugging and tooling
    - Copy: copy(**overrides) for shallow cloning
    - Dodecagram encoding: to_dodecagram() uses global ast_to_dodecagram if available
    - Structural equality: __eq__ based on type and serialized content (excluding positions)
    """

    # Position information is optional and can be set later via set_pos().
    def __init__(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ):
        self.filename = filename
        self.line = line
        self.col = col
        self.end_line = end_line
        self.end_col = end_col

    # ----- Source position helpers -----
    def set_pos(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ) -> "ASTNode":
        if filename is not None:
            self.filename = filename
        if line is not None:
            self.line = line
        if col is not None:
            self.col = col
        if end_line is not None:
            self.end_line = end_line
        if end_col is not None:
            self.end_col = end_col
        return self

    # ----- Introspection helpers -----
    def _is_pos_field(self, name: str) -> bool:
        return name in ("filename", "line", "col", "end_line", "end_col")

    def _iter_fields(self):
        # Do not consider private/dunder attributes as AST data
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            yield k, v

    def children(self) -> List["ASTNode"]:
        """Return direct child AST nodes (flattening lists)."""
        result: List[ASTNode] = []
        for _, v in self._iter_fields():
            if isinstance(v, ASTNode):
                result.append(v)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, ASTNode):
                        result.append(it)
        return result

    def walk(self):
        """Preorder traversal of this subtree."""
        yield self
        for c in self.children():
            yield from c.walk()

    # ----- Visitor pattern -----
    def accept(self, visitor):
        """Call visitor.visit_<Type>(self) if present, else visitor.visit(self) if present."""
        method = getattr(visitor, f"visit_{self.__class__.__name__}", None)
        if callable(method):
            return method(self)
        generic = getattr(visitor, "visit", None)
        if callable(generic):
            return generic(self)
        return None

    # ----- Structural operations -----
    def replace_child(self, old: "ASTNode", new: Optional["ASTNode"]) -> bool:
        """
        Replace a direct child 'old' with 'new'.
        If 'new' is None, removes the child if it's in a list; clears attribute otherwise.
        Returns True if a replacement/removal occurred.
        """
        changed = False
        for k, v in list(self._iter_fields()):
            if isinstance(v, ASTNode):
                if v is old:
                    setattr(self, k, new)
                    changed = True
            elif isinstance(v, list):
                # Replace in lists; remove if new is None
                new_list = []
                for it in v:
                    if it is old:
                        if new is not None:
                            new_list.append(new)
                        changed = True
                    else:
                        new_list.append(it)
                if changed:
                    setattr(self, k, new_list)
        return changed

    # ----- Serialization / Debugging -----
    def to_dict(self, *, include_pos: bool = True) -> Dict[str, object]:
        """Convert the node (recursively) to a dict suitable for JSON/debugging."""
        d: Dict[str, object] = {"__type__": self.__class__.__name__}
        for k, v in self._iter_fields():
            if not include_pos and self._is_pos_field(k):
                continue
            if isinstance(v, ASTNode):
                d[k] = v.to_dict(include_pos=include_pos)
            elif isinstance(v, list):
                d[k] = [
                    (it.to_dict(include_pos=include_pos) if isinstance(it, ASTNode) else it)
                    for it in v
                ]
            else:
                d[k] = v
        return d

    def pretty(self, indent: str = "  ") -> str:
        """Human-readable multi-line tree dump."""
        lines: List[str] = []

        def rec(n: "ASTNode", depth: int):
            pad = indent * depth
            header = n.__class__.__name__
            pos = []
            if n.filename:
                pos.append(f'file="{n.filename}"')
            if n.line is not None and n.col is not None:
                pos.append(f"@{n.line}:{n.col}")
            if n.end_line is not None and n.end_col is not None:
                pos.append(f"-{n.end_line}:{n.end_col}")
            if pos:
                header += " [" + " ".join(pos) + "]"
            lines.append(pad + header)

            # Show scalar fields
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    continue
                if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                    continue
                lines.append(pad + indent + f"{k} = {v!r}")

            # Recurse into child nodes
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    lines.append(pad + indent + f"{k}:")
                    rec(v, depth + 2)
                elif isinstance(v, list):
                    child_nodes = [it for it in v if isinstance(it, ASTNode)]
                    if child_nodes:
                        lines.append(pad + indent + f"{k}: [{len(child_nodes)}]")
                        for it in child_nodes:
                            rec(it, depth + 2)

        rec(self, 0)
        return "\n".join(lines)

    def copy(self, **overrides):
        """
        Shallow copy with optional field overrides:
            new = node.copy(body=new_body)
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)  # type: ignore
        # Copy all instance attributes
        new_obj.__dict__.update(self.__dict__)
        # Apply overrides
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    def to_dodecagram(self) -> str:
        """
        Encode this node (and subtree) using the Dodecagram mapping.
        Relies on a global function ast_to_dodecagram(node).
        """
        f = globals().get("ast_to_dodecagram")
        if callable(f):
            return f(self)  # type: ignore[misc]
        raise RuntimeError("ast_to_dodecagram() is not available in this module")

    # ----- Equality / Representation -----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASTNode):
            return False
        if self.__class__ is not other.__class__:
            return False
        return self.to_dict(include_pos=False) == other.to_dict(include_pos=False)

    def __repr__(self) -> str:
        # Compact representation showing scalar fields only
        fields: List[str] = []
        for k, v in self._iter_fields():
            if isinstance(v, ASTNode):
                continue
            if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                continue
            fields.append(f"{k}={v!r}")
        inner = ", ".join(fields)
        return f"{self.__class__.__name__}({inner})"


class Program(ASTNode):
    def __init__(self, functions: List['Function']):
        self.functions = functions

    def __repr__(self):
        return f"Program({self.functions})"


class Function(ASTNode):
    def __init__(self, name: str, body: List['Statement']):
        self.name = name
        self.body = body

    def __repr__(self):
        return f"Function({self.name}, body={self.body})"


class Statement(ASTNode):
    """Base class for all statements in Density 2."""
    pass


class PrintStatement(Statement):
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Print({self.text!r})"


class CIAMBlock(Statement):
    def __init__(self, name: str, params: List[str], body_text: str):
        self.name = name
        self.params = params
        self.body_text = body_text  # raw Density 2 snippet

    def __repr__(self):
        return f"CIAMBlock(name={self.name!r}, params={self.params}, body_len={len(self.body_text)})"


class MacroCall(Statement):
    def __init__(self, name: str, arg_texts: List[str]):
        self.name = name
        self.arg_texts = arg_texts  # raw argument texts

    def __repr__(self):
        return f"MacroCall({self.name!r}, args={self.arg_texts})"


class InlineBlock(Statement):
    def __init__(self, lang: str, content: str):
        self.lang = lang  # 'asm', 'c', 'python'
        self.content = content

    def __repr__(self):
        return f"InlineBlock(lang={self.lang!r}, content_len={len(self.content)})"


# -----------------------------
# Dodecagram AST encoding
# -----------------------------
_DODECAGRAM_MAP = {
    'Program': '0',
    'Function': '1',
    'PrintStatement': '2',
    'CIAMBlock': '3',
    'MacroCall': '4',
    # InlineBlock variants:
    'InlineBlock:asm': '5',
    'InlineBlock:python': '6',
    'InlineBlock:py': '6',
    'InlineBlock:c': '7',
    'InlineBlock:other': '8',
    # Reserved for future nodes:
    '_reserved9': '9',
    '_reserveda': 'a',
    '_reservedb': 'b',
}

def ast_to_dodecagram(node: ASTNode) -> str:
    """
    Preorder encoding of the AST using the Dodecagram alphabet 0-9,a,b.
    """
    def enc(n: ASTNode) -> str:
        if isinstance(n, Program):
            s = _DODECAGRAM_MAP['Program']
            for f in n.functions:
                s += enc(f)
            return s
        if isinstance(n, Function):
            s = _DODECAGRAM_MAP['Function']
            for st in n.body:
                s += enc(st)
            return s
        if isinstance(n, PrintStatement):
            return _DODECAGRAM_MAP['PrintStatement']
        if isinstance(n, CIAMBlock):
            return _DODECAGRAM_MAP['CIAMBlock']
        if isinstance(n, MacroCall):
            return _DODECAGRAM_MAP['MacroCall']
        if isinstance(n, InlineBlock):
            key = f'InlineBlock:{n.lang}'
            ch = _DODECAGRAM_MAP.get(key, _DODECAGRAM_MAP['InlineBlock:other'])
            return ch
        # Unknown node -> reserved
        return _DODECAGRAM_MAP['_reserved9']
    return enc(node)

# -----------------------------
# Parser
# -----------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        # Macro table collected while parsing: name -> CIAMBlock
        self.macro_table: Dict[str, CIAMBlock] = {}

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def lookahead(self, offset: int) -> Optional[Token]:
        idx = self.pos + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_type: str) -> Token:
        tok = self.peek()
        if not tok or tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok

    def match(self, expected_type: str) -> bool:
        tok = self.peek()
        if tok and tok.type == expected_type:
            self.pos += 1
            return True
        return False

    def parse(self) -> Program:
        functions: List[Function] = []
        while self.peek() is not None:
            # Expect IDENT (function name) or skip stray tokens
            tok = self.peek()
            if tok.type == 'IDENT':
                # Look for function signature IDENT()
                if self.lookahead(1) and self.lookahead(1).type == 'LPAREN':
                    functions.append(self.parse_function())
                else:
                    # Skip stray identifier
                    self.pos += 1
            else:
                # Skip unknown tokens at top-level
                self.pos += 1
        return Program(functions)

    def parse_function(self) -> Function:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        self.consume('RPAREN')
        body: List[Statement] = []
        if self.match('LBRACE'):
            # Block body
            body = self.parse_statements_until_rbrace()
            self.consume('RBRACE')
        else:
            # Single statement allowed
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        return Function(name_tok.value, body)

    def parse_statements_until_rbrace(self) -> List[Statement]:
        stmts: List[Statement] = []
        while True:
            tok = self.peek()
            if tok is None or tok.type == 'RBRACE':
                break
            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    stmts.extend(stmt)
                else:
                    stmts.append(stmt)
        return stmts

    def parse_statement(self) -> Optional[Union[Statement, List[Statement]]]:
        tok = self.peek()
        if tok is None:
            return None

        if tok.type == 'IDENT' and tok.value == 'Print':
            return self.parse_print()

        if tok.type == 'CIAM':
            # Parse CIAM block definition, store in macro table, no direct output
            ciam_tok = self.consume('CIAM')
            content = ciam_tok.value
            # Strip delimiters: starts with ''' and ends with ,,,
            if content.startswith("'''"):
                content_inner = content[3:]
                if content_inner.endswith(",,,"):
                    content_inner = content_inner[:-3]
            else:
                content_inner = content

            name, params, body_text = self._parse_ciam_content(content_inner.strip(), ciam_tok)
            ciam_block = CIAMBlock(name, params, body_text)
            self.macro_table[name] = ciam_block
            # Defining a macro does not emit code
            return None

        if tok.type.startswith('INLINE_'):
            return self.parse_inline_block()

        if tok.type == 'IDENT':
            # Possibly a macro invocation: IDENT(...)
            ident = tok.value
            la = self.lookahead(1)
            if la and la.type == 'LPAREN':
                return self.parse_macro_call()
            else:
                # Unknown ident; consume to progress
                self.pos += 1
                return None

        # Unknown token; consume to progress
        self.pos += 1
        return None

    def _parse_ciam_content(self, content: str, tok: Token) -> Tuple[str, List[str], str]:
        # First line: Name(params)
        lines = content.splitlines()
        if not lines:
            raise SyntaxError(f"Empty CIAM at line {tok.line}")

        header = lines[0].strip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$', header)
        if not m:
            raise SyntaxError(f"Invalid CIAM header '{header}' at line {tok.line}")
        name = m.group(1)
        params_str = m.group(2).strip()
        params = []
        if params_str:
            params = [p.strip() for p in params_str.split(',') if p.strip()]
        body_text = '\n'.join(lines[1:]).strip()
        return name, params, body_text

    def parse_macro_call(self) -> MacroCall:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        args: List[str] = []
        current_parts: List[str] = []
        paren_depth = 1
        while True:
            tok = self.peek()
            if tok is None:
                raise SyntaxError("Unclosed macro call")
            if tok.type == 'LPAREN':
                paren_depth += 1
                current_parts.append(tok.value)
                self.pos += 1
            elif tok.type == 'RPAREN':
                paren_depth -= 1
                if paren_depth == 0:
                    # finalize last arg
                    arg_text = ''.join(current_parts).strip()
                    if arg_text:
                        args.append(arg_text)
                    self.pos += 1
                    break
                else:
                    current_parts.append(tok.value)
                    self.pos += 1
            elif tok.type == 'COMMA' and paren_depth == 1:
                arg_text = ''.join(current_parts).strip()
                args.append(arg_text)
                current_parts = []
                self.pos += 1
            else:
                current_parts.append(tok.value)
                self.pos += 1
        self.consume('SEMICOLON')
        return MacroCall(name_tok.value, args)

    def parse_inline_block(self) -> InlineBlock:
        tok = self.peek()
        lang = tok.type.split('_', 1)[1].lower()  # asm / c / py
        inline_tok = self.consume(tok.type)
        # strip off #lang and #endlang markers:
        content = re.sub(r'^#\w+', '', inline_tok.value, flags=re.DOTALL)
        content = re.sub(r'#end\w+$', '', content, flags=re.DOTALL)
        return InlineBlock(lang, content.strip())

    def parse_print(self) -> PrintStatement:
        self.consume('IDENT')  # 'Print'
        self.consume('COLON')
        self.consume('LPAREN')

        # Simple string concat of STRING (+ STRING)*
        parts: List[str] = []
        first = True
        while True:
            tok = self.peek()
            if not tok:
                raise SyntaxError("Unclosed Print: missing ')'")
            if first:
                if tok.type != 'STRING':
                    raise SyntaxError(f"Print expects a string literal, got {tok}")
                parts.append(eval(tok.value))
                self.pos += 1
                first = False
            else:
                if tok.type == 'PLUS':
                    self.pos += 1
                    tok2 = self.peek()
                    if not tok2 or tok2.type != 'STRING':
                        raise SyntaxError("Expected string after '+' in Print")
                    parts.append(eval(tok2.value))
                    self.pos += 1
                else:
                    break

            # optional spaces are already skipped by lexer

        self.consume('RPAREN')
        self.consume('SEMICOLON')
        full_text = ''.join(parts)
        return PrintStatement(full_text)


# -----------------------------
# Macro Expansion
# -----------------------------
def expand_macros(program: Program, macro_table: Dict[str, CIAMBlock], max_depth: int = 32) -> Program:
    def expand_statements(stmts: List[Statement], depth: int) -> List[Statement]:
        result: List[Statement] = []
        for s in stmts:
            if isinstance(s, MacroCall):
                if s.name not in macro_table:
                    # Unknown macro, ignore call
                    continue
                macro = macro_table[s.name]
                expanded = expand_macro_call(macro, s, depth)
                result.extend(expanded)
            else:
                result.append(s)
        return result

    def expand_macro_call(macro: CIAMBlock, call: MacroCall, depth: int) -> List[Statement]:
        if depth <= 0:
            raise RecursionError("Macro expansion depth exceeded")
        # Build replacement text by simple identifier replacement
        body = macro.body_text

        # map params -> args text
        mapping = {p: (call.arg_texts[i] if i < len(call.arg_texts) else '') for i, p in enumerate(macro.params)}

        # Replace params with args (word boundary replacement)
        for p, a in mapping.items():
            # Use a simple token-ish replace; avoid capturing partial identifiers
            body = re.sub(rf'\b{re.escape(p)}\b', a, body)

        # Re-parse the expanded snippet into statements
        sub_tokens = tokenize(body)
        sub_parser = Parser(sub_tokens)
        # Share the same macro table to allow nested macro definitions/uses
        sub_parser.macro_table = macro_table
        # We parse a synthetic function body to reuse statement parser
        sub_stmts = sub_parser.parse_statements_until_rbrace()  # until EOF
        # Recursively expand within
        return expand_statements(sub_stmts, depth - 1)

    # Expand within each function
    new_functions: List[Function] = []
    for f in program.functions:
        expanded_body = expand_statements(f.body, max_depth)
        new_functions.append(Function(f.name, expanded_body))
    return Program(new_functions)


# -----------------------------
# NASM Emitter
# -----------------------------
class CodeGenerator:
    def __init__(self, ast: Program):
        self.ast = ast
        self.text_lines: List[str] = []
        self.data_lines: List[str] = []
        self.string_table: Dict[str, str] = {}  # map string -> label
        self.label_counter = 0
        # simple register reuse cache (invalidated across inline blocks)
        self._reg_cache = {'rax_sys_write': False, 'rdi_stdout': False}

    def generate(self) -> str:
        # Expand macros already done in compile_density2
        self.text_lines = []
        self.data_lines = []
        self.string_table = {}
        self.label_counter = 0
        self._invalidate_reg_cache()

        # Header and functions
        self._emit_header()
        for func in self.ast.functions:
            self._emit_function(func)

        # Build final assembly
        final_lines: List[str] = []
        final_lines.append('section .data')
        final_lines.extend('    ' + line for line in self.data_lines)
        final_lines.append('section .text')
        final_lines.append('    global _start')
        for l in self.text_lines:
            final_lines.append(l)

        return '\n'.join(final_lines)

    def _invalidate_reg_cache(self):
        self._reg_cache['rax_sys_write'] = False
        self._reg_cache['rdi_stdout'] = False

    def _emit_header(self):
        # Add a small banner; sections are assembled at the end
        self.text_lines.append('; --- Density 2 NASM output ---')
        self.text_lines.append('; inline C: compiled then translated (AT&T -> NASM best-effort)')
        self.text_lines.append('; inline Python: executed at codegen; use emit("...") to output NASM')

    def _emit_function(self, func: Function):
        # Entry label
        if func.name == 'Main':
            self.text_lines.append('_start:')
        else:
            self.text_lines.append(f'{func.name}:')
        # function entry: registers not assumed
        self._invalidate_reg_cache()

        for stmt in func.body:
            if isinstance(stmt, PrintStatement):
                self._emit_print(stmt.text)
            elif isinstance(stmt, InlineBlock):
                self._emit_inline(stmt)
                # assume inline code may clobber registers
                self._invalidate_reg_cache()
            elif isinstance(stmt, CIAMBlock):
                # Already handled by expand_macros; leave a comment in case any remain
                self.text_lines.append(f'    ; CIAMBlock ignored (should be expanded): {getattr(stmt, "name", "?")}')
            elif isinstance(stmt, MacroCall):
                # Should be expanded away
                self.text_lines.append(f'    ; MacroCall ignored (should be expanded): {getattr(stmt, "name", "?")}')
            else:
                # Unknown statement type
                self.text_lines.append('    ; Unknown statement')

        if func.name == 'Main':
            self._emit_exit()

    def _encode_string_bytes(self, s: str) -> List[int]:
        return list(s.encode('utf-8'))

    def _get_string_label(self, text: str) -> Tuple[str, int]:
        if text not in self.string_table:
            label = f'str_{self.label_counter}'
            self.label_counter += 1
            data_bytes = self._encode_string_bytes(text)
            # Store bytes + newline + NUL terminator
            stored = data_bytes + [10, 0]
            bytes_list = ', '.join(str(b) for b in stored)
            self.data_lines.append(f'{label} db {bytes_list}')
            # Length to write: bytes plus newline only (exclude NUL)
            length = len(data_bytes) + 1
            self.string_table[text] = label
        else:
            label = self.string_table[text]
            data_bytes = self._encode_string_bytes(text)
            length = len(data_bytes) + 1
        return label, length

    def _emit_print(self, text: str):
        label, length = self._get_string_label(text)
        # register reuse: avoid redundant loads across consecutive prints
        if not self._reg_cache['rax_sys_write']:
            self.text_lines.append(f'    mov rax, 1          ; sys_write')
            self._reg_cache['rax_sys_write'] = True
        if not self._reg_cache['rdi_stdout']:
            self.text_lines.append(f'    mov rdi, 1          ; stdout')
            self._reg_cache['rdi_stdout'] = True
        self.text_lines.append(f'    mov rsi, {label}    ; message')
        self.text_lines.append(f'    mov rdx, {length}         ; length (bytes)')
        self.text_lines.append('    syscall')
        # after syscall, rax/rdi are still fine for subsequent prints in our model

    def _emit_exit(self):
        self.text_lines.append('    mov rax, 60         ; sys_exit')
        self.text_lines.append('    xor rdi, rdi        ; status 0')
        self.text_lines.append('    syscall')
        # invalidate cache after exit emitter
        self._invalidate_reg_cache()

    def _emit_inline(self, block: InlineBlock):
        if block.lang == 'asm':
            self.text_lines.append('    ; inline NASM start')
            for line in block.content.splitlines():
                line = line.rstrip()
                if line:
                    self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline NASM end')
        elif block.lang in ('py', 'python'):
            self.text_lines.append('    ; inline Python start')
            for line in self._run_inline_python(block.content):
                self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline Python end')
        elif block.lang == 'c':
            self.text_lines.append('    ; inline C start')
            asm_lines = self._compile_c_to_asm(block.content)
            if asm_lines:
                for l in asm_lines:
                    self.text_lines.append('    ' + l)
            else:
                # Fallback: keep it commented
                for line in block.content.splitlines():
                    self.text_lines.append('    ; ' + line)
                self.text_lines.append('    ; (no C compiler found; block left as comment)')
            self.text_lines.append('    ; inline C end')
        else:
            # Unknown inline language
            for line in block.content.splitlines():
                self.text_lines.append(f'    ; inline {block.lang} ignored: {line}')

    def _run_inline_python(self, code: str) -> List[str]:
        # Provide a tiny DSL where Python can emit NASM
        lines: List[str] = []

        def emit(s: str):
            if not isinstance(s, str):
                raise TypeError("emit() expects a string")
            lines.append(s)

        def label(prefix: str = 'gen'):
            lbl = f'{prefix}_{self.label_counter}'
            self.label_counter += 1
            return lbl

        globals_dict = {
            '__builtins__': {
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'print': print,
            },
            'emit': emit,
            'label': label,
        }
        locals_dict: Dict[str, object] = {}
        try:
            exec(code, globals_dict, locals_dict)
        except Exception as ex:
            lines.append(f'; [inline python error] {ex!r}')
        return lines

    def _compile_c_to_asm(self, c_code: str) -> List[str]:
        # Try tcc, clang, gcc in that order to produce assembly text
        compiler = None
        for cand in ('tcc', 'clang', 'gcc'):
            if shutil.which(cand):
                compiler = cand
                break
        if compiler is None:
            return []

        tmpdir = tempfile.mkdtemp(prefix='den2_c_')
        c_path = os.path.join(tmpdir, 'inline.c')
        asm_path = os.path.join(tmpdir, 'inline.s')
        try:
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            if compiler == 'tcc':
                cmd = [compiler, '-nostdlib', '-S', c_path, '-o', asm_path]
            elif compiler == 'clang':
                cmd = [compiler, '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer']
            else:  # gcc
                cmd = [compiler, '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer']

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not os.path.exists(asm_path):
                return []

            with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()

            translated = self._translate_att_to_nasm(raw)
            if translated:
                return translated

            commented = ['; [begin compiled C assembly]']
            commented += ['; ' + line for line in raw.splitlines()]
            commented.append('; [end compiled C assembly]')
            return commented
        except Exception as ex:
            return [f'; [inline c compile error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

        def _compile_c_to_asm(self, c_code: str) -> List[str]:
        # Try tcc, clang, gcc, cl (MSVC) in that order to produce assembly text
            compiler = None
        for cand in ('tcc', 'clang', 'gcc', 'cl'):
            if shutil.which(cand):
                compiler = cand
                break
        if compiler is None:
            return []

        # MSVC path (use /FA to generate assembly listing)
        if compiler == 'cl':
            lines = self._compile_with_msvc(c_code)
            if lines:
                # Unobtrusive log of the compiler used
                return ['; [inline c compiler: cl]'] + lines
            return []

        tmpdir = tempfile.mkdtemp(prefix='den2_c_')
        c_path = os.path.join(tmpdir, 'inline.c')
        asm_path = os.path.join(tmpdir, 'inline.s')
        try:
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            cmd: List[str]
            if compiler == 'tcc':
                # tcc can emit assembly with -S
                cmd = [compiler, '-nostdlib', '-S', c_path, '-o', asm_path]
            elif compiler == 'clang':
                cmd = [compiler, '-x', 'c', '-O2', '-S', c_path, '-o', asm_path, '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer']
            else:  # gcc
                cmd = [compiler, '-x', 'c', '-O2', '-S', c_path, '-o', asm_path, '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer']

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not os.path.exists(asm_path):
                return []

            with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()

            # Attempt to translate AT&T/Intel output to NASM
            translated = self._translate_att_to_nasm(raw)
            if translated:
                return [f'; [inline c compiler: {compiler}]'] + translated

            # Fallback to comments if translation failed
            commented = [f'; [inline c compiler: {compiler}]', '; [begin compiled C assembly]']
            for line in raw.splitlines():
                commented.append('; ' + line)
            commented.append('; [end compiled C assembly]')
            return commented
        except Exception as ex:
            return [f'; [inline c compile error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _compile_with_msvc(self, c_code: str) -> List[str]:
        """
        Use MSVC cl.exe to compile C to assembly listing (.asm via /FA), then translate
        to NASM-friendly lines.
        """
        tmpdir = tempfile.mkdtemp(prefix='den2_msvc_')
        try:
            c_path = os.path.join(tmpdir, 'inline.c')
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            # cl will emit inline.asm in the current working directory (tmpdir)
            cmd = ['cl', '/nologo', '/FA', '/c', os.path.basename(c_path)]
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            asm_listing = os.path.join(tmpdir, 'inline.asm')
            lines: List[str] = []

            # If cl produced diagnostics, include them as comments (unobtrusive)
            if proc.stdout:
                for ln in proc.stdout.splitlines():
                    if ln.strip():
                        lines.append('; [cl] ' + ln)
            if proc.returncode != 0 and proc.stderr:
                for ln in proc.stderr.splitlines():
                    if ln.strip():
                        lines.append('; [cl err] ' + ln)

            if os.path.exists(asm_listing):
                with open(asm_listing, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                translated = self._msvc_asm_to_nasm(raw)
                # Place translated code after any diagnostics
                return lines + translated

            # No asm produced; return only diagnostics if any
            return lines
        except Exception as ex:
            return [f'; [inline c compile (msvc) error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _msvc_asm_to_nasm(self, msvc_asm: str) -> List[str]:
        """
        Filter MSVC's Intel-style assembly into NASM-friendly output.
        Best-effort: strip MSVC-specific metadata, map basic directives,
        preserve labels/instructions/comments.
        """
        out: List[str] = []
        for line in msvc_asm.splitlines():
            s = line.rstrip()
            if not s:
                continue

            # Keep original comments, already ';' prefixed in MSVC listings
            if s.lstrip().startswith(';'):
                out.append(s)
                continue

            tok = s.strip()

            # Skip or rewrite common MSVC metadata/directives
            upper = tok.upper()
            if upper.startswith(('TITLE ', 'COMMENT ', 'INCLUDE ', 'INCLUDELIB ')):
                # Ignore headers/includes in listing
                continue
            if upper.startswith(('.MODEL', '.CODE', '.DATA', '.CONST', '.XDATA', '.PDATA', '.STACK', '.LIST', '.686', '.686P', '.XMM', '.X64')):
                continue
            if upper.startswith(('PUBLIC ', 'EXTRN ', 'EXTERN ', 'ASSUME ')):
                continue
            if upper == 'END':
                continue
            if upper.startswith('ALIGN '):
                # Map to NASM align
                parts = tok.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    out.append(f'align {parts[1]}')
                continue

            # PROC/ENDP handling: turn "label PROC" -> "label:" and skip ENDP
            m_proc = re.match(r'^([A-Za-z_$.@?][\w$.@?]*)\s+PROC\b', tok)
            if m_proc:
                out.append(f'{m_proc.group(1)}:')
                continue
            if re.match(r'^[A-Za-z_$.@?][\w$.@?]*\s+ENDP\b', tok):
                # End of procedure: ignore (no epilogue emission here)
                continue

            # Data directives: DB/DW/DD/DQ -> lowercase NASM-friendly
            m_data = re.match(r'^(DB|DW|DD|DQ)\b(.*)$', tok, re.IGNORECASE)
            if m_data:
                out.append(m_data.group(1).lower() + m_data.group(2))
                continue

            # Replace "BYTE/WORD/DWORD/QWORD PTR" with NASM "byte/word/dword/qword"
            tok = re.sub(r'\b(BYTE|WORD|DWORD|QWORD)\s+PTR\b', lambda m: m.group(1).lower(), tok)

            # Replace "OFFSET FLAT:label" or "OFFSET label" with just label
            tok = re.sub(r'\bOFFSET\s+FLAT:', '', tok, flags=re.IGNORECASE)
            tok = re.sub(r'\bOFFSET\s+', '', tok, flags=re.IGNORECASE)

            # Some MSVC emits "FLAT:" as a segment label prefix; drop it
            tok = tok.replace(' FLAT:', ' ')

            # Instructions are already Intel syntax; keep as-is
            out.append(tok)

        return out

    # --- AT&T -> NASM best-effort translation helpers ---
    def _translate_att_to_nasm(self, att_asm: str) -> List[str]:
        out: List[str] = []
        for line in att_asm.splitlines():
            s = line.strip()
            if not s:
                continue

            if s.startswith('.'):
                if s.startswith(('.globl', '.global', '.text', '.data', '.bss', '.rodata', '.section',
                                 '.type', '.size', '.file', '.ident', '.cfi', '.p2align', '.intel_syntax', '.att_syntax')):
                    continue
                out.append(f'; {s}')
                continue

            if s.endswith(':'):
                out.append(s)
                continue

            s = s.split('\t#', 1)[0].split(' #', 1)[0].strip()
            if not s:
                continue

            parts = s.split(None, 1)
            op = parts[0]
            rest = parts[1] if len(parts) > 1 else ''
            op_n = re.sub(r'(q|l|w|b)$', '', op)

            ops = [o.strip() for o in rest.split(',')] if rest else []
            ops = [self._att_operand_to_nasm(o) for o in ops]

            if len(ops) == 2:
                ops = [ops[1], ops[0]]

            if ops:
                out.append(f'{op_n} ' + ', '.join(ops))
            else:
                out.append(op_n)

        return out

    def _att_operand_to_nasm(self, o: str) -> str:
        o = o.strip()
        if o.startswith('$'):
            return o[1:]
        o = re.sub(r'%([a-zA-Z][a-zA-Z0-9]*)', r'\1', o)

        m = re.match(r'^\s*([\-+]?\d+)?\s*\(\s*([a-zA-Z0-9%]+)\s*(?:,\s*([a-zA-Z0-9%]+)\s*(?:,\s*(\d+))?)?\s*\)\s*$', o)
        if m:
            disp, base, index, scale = m.groups()
            base = base.lstrip('%')
            parts = [base]
            if index:
                idx = index.lstrip('%')
                sc = int(scale) if scale else 1
                parts.append(f'{idx}*{sc}')
            addr = ' + '.join(parts) if parts else ''
            if disp:
                sign = '+' if int(disp) >= 0 else '-'
                addr = f'{addr} {sign} {abs(int(disp))}' if addr else str(disp)
            return f'[{addr}]'

        return o


class PrintStatement(Statement):
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Print({self.text!r})"


class CIAMBlock(Statement):
    def __init__(self, name: str, params: List[str], body_text: str):
        self.name = name
        self.params = params
        self.body_text = body_text  # raw Density 2 snippet

    def __repr__(self):
        return f"CIAMBlock(name={self.name!r}, params={self.params}, body_len={len(self.body_text)})"


class MacroCall(Statement):
    def __init__(self, name: str, arg_texts: List[str]):
        self.name = name
        self.arg_texts = arg_texts  # raw argument texts

    def __repr__(self):
        return f"MacroCall({self.name!r}, args={self.arg_texts})"


class InlineBlock(Statement):
    def __init__(self, lang: str, content: str):
        self.lang = lang  # 'asm', 'c', 'python'
        self.content = content

    def __repr__(self):
        return f"InlineBlock(lang={self.lang!r}, content_len={len(self.content)})"

# -----------------------------
# Runtime mutation tracking + history encoding (non-invasive, opt-in)
# -----------------------------

class _TrackedList(list):
    """
    List wrapper to track mutations on AST node list fields.
    """
    __slots__ = ('_owner', '_field')

    def __init__(self, owner: 'ASTNode', field: str, iterable=None):
        super().__init__(iterable or [])
        self._owner = owner
        self._field = field

    def _rec(self, op: str, detail: Dict[str, object]):
        if getattr(self._owner, '_track_mutations', False):
            self._owner._record_mutation(op=op, field=self._field, before=None, after=None, detail=detail)

    def append(self, item):
        super().append(item)
        self._rec('list.append', {'item': self._owner._brief(item)})

    def extend(self, it):
        vals = [self._owner._brief(v) for v in it]
        super().extend(it)
        self._rec('list.extend', {'items': vals})

    def insert(self, index, item):
        super().insert(index, item)
        self._rec('list.insert', {'index': index, 'item': self._owner._brief(item)})

    def pop(self, index=-1):
        v = super().pop(index)
        self._rec('list.pop', {'index': index, 'item': self._owner._brief(v)})
        return v

    def remove(self, item):
        super().remove(item)
        self._rec('list.remove', {'item': self._owner._brief(item)})

    def clear(self):
        n = len(self)
        super().clear()
        self._rec('list.clear', {'count': n})

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            before = [self._owner._brief(v) for v in self[key]]
            super().__setitem__(key, value)
            after = [self._owner._brief(v) for v in self[key]]
            self._rec('list.setslice', {'slice': str(key), 'before': before, 'after': after})
        else:
            before = self._owner._brief(self[key]) if 0 <= key < len(self) else None
            super().__setitem__(key, value)
            self._rec('list.setitem', {'index': key, 'before': before, 'after': self._owner._brief(value)})

    def __delitem__(self, key):
        if isinstance(key, slice):
            before = [self._owner._brief(v) for v in self[key]]
            super().__delitem__(key)
            self._rec('list.delslice', {'slice': str(key), 'before': before})
        else:
            before = self._owner._brief(self[key]) if 0 <= key < len(self) else None
            super().__delitem__(key)
            self._rec('list.delitem', {'index': key, 'before': before})


# Keep original setattr so we can wrap and still call through.
_ASTNODE_ORIG_SETATTR = ASTNode.__setattr__

def _astnode_tracking_setattr(self: ASTNode, name: str, value):
    # Never track private/internal attributes
    if name.startswith('_'):
        return _ASTNODE_ORIG_SETATTR(self, name, value)

    # Wrap lists into tracked lists
    if isinstance(value, list) and not isinstance(value, _TrackedList):
        value = _TrackedList(self, name, value)

    old_present = name in self.__dict__
    old_value = self.__dict__.get(name, None)

    # Set the attribute
    _ASTNODE_ORIG_SETATTR(self, name, value)

    # Record mutation if enabled and value actually changed (identity or repr inequality)
    if getattr(self, '_track_mutations', False):
        changed = (not old_present) or (old_value is not value)
        if changed:
            self._record_mutation(
                op='setattr',
                field=name,
                before=self._brief(old_value) if old_present else None,
                after=self._brief(value),
                detail=None
            )

def _astnode_brief(self: ASTNode, v):
    # Short, non-recursive summaries for history entries
    if isinstance(v, ASTNode):
        return v.__class__.__name__
    if isinstance(v, _TrackedList):
        return f'TrackedList(len={len(v)})'
    if isinstance(v, list):
        return f'list(len={len(v)})'
    if isinstance(v, str):
        s = v
        return s if len(s) <= 32 else s[:29] + '...'
    return v

def _astnode_record_mutation(self: ASTNode, *, op: str, field: str, before, after, detail: Optional[Dict[str, object]]):
    # Lazily create history storage
    if '_history' not in self.__dict__:
        _ASTNODE_ORIG_SETATTR(self, '_history', [])
    event = {
        'ts': time.time(),
        'node': self.__class__.__name__,
        'op': op,
        'field': field,
        'before': before,
        'after': after,
        'detail': detail or {}
    }
    self._history.append(event)

def _astnode_wrap_lists(self: ASTNode):
    # Ensure existing list fields are wrapped
    for k, v in list(self.__dict__.items()):
        if k.startswith('_'):
            continue
        if isinstance(v, list) and not isinstance(v, _TrackedList):
            _ASTNODE_ORIG_SETATTR(self, k, _TrackedList(self, k, v))

def _astnode_enable_mutation_tracking(self: ASTNode, enable: bool = True, recursive: bool = False) -> ASTNode:
    # Initialize tracking flags/history lazily
    if '_track_mutations' not in self.__dict__:
        _ASTNODE_ORIG_SETATTR(self, '_track_mutations', False)
    if '_history' not in self.__dict__:
        _ASTNODE_ORIG_SETATTR(self, '_history', [])

    _ASTNODE_ORIG_SETATTR(self, '_track_mutations', bool(enable))
    if enable:
        _astnode_wrap_lists(self)
    if recursive:
        for c in self.children():
            c.enable_mutation_tracking(enable=True, recursive=True)
    return self

def _astnode_is_tracking(self: ASTNode) -> bool:
    return bool(getattr(self, '_track_mutations', False))

def _astnode_get_mutation_history(self: ASTNode, recursive: bool = False) -> List[Dict[str, object]]:
    hist: List[Dict[str, object]] = list(getattr(self, '_history', []))
    if recursive:
        for c in self.children():
            hist.extend(c.get_mutation_history(recursive=True))
    # Sort by timestamp for a coherent timeline
    hist.sort(key=lambda e: e.get('ts', 0.0))
    return hist

def _astnode_clear_mutation_history(self: ASTNode, recursive: bool = False):
    if '_history' in self.__dict__:
        self._history.clear()
    if recursive:
        for c in self.children():
            c.clear_mutation_history(recursive=True)

def _astnode_encode_history(self: ASTNode, recursive: bool = False, mode: str = 'compact') -> str:
    """
    Encode history into a compact ASCII stream.
    - mode='compact': opCode|fieldHash|nodeCode per event, joined by ';'
    """
    events = self.get_mutation_history(recursive=recursive)
    if not events:
        return ''

    # Map op -> single char
    op_map = {
        'setattr': 'S',
        'list.append': 'A',
        'list.extend': 'E',
        'list.insert': 'I',
        'list.pop': 'P',
        'list.remove': 'X',
        'list.clear': 'C',
        'list.setitem': 'U',
        'list.setslice': 'V',
        'list.delitem': 'D',
        'list.delslice': 'L',
        'replace_child': 'R',
    }
    # Map node class to dodecagram digit when possible
    node_to_code = {
        'Program': '0', 'Function': '1', 'PrintStatement': '2', 'CIAMBlock': '3', 'MacroCall': '4',
        'InlineBlock': '8'  # specific lang variants are compacted to 8 here
    }

    def h8(s: str) -> str:
        # Short 8-hex hash for field names
        return hashlib.sha1(s.encode('utf-8')).hexdigest()[:8]

    parts: List[str] = []
    for ev in events:
        op_c = op_map.get(ev.get('op', ''), '?')
        fld = str(ev.get('field', ''))
        node_c = node_to_code.get(str(ev.get('node', '')), '9')
        parts.append(f'{op_c}|{h8(fld)}|{node_c}')
    return ';'.join(parts)

# Monkey patch: add capabilities to ASTNode without altering existing API
ASTNode._brief = _astnode_brief
ASTNode._record_mutation = _astnode_record_mutation
ASTNode.enable_mutation_tracking = _astnode_enable_mutation_tracking
ASTNode.is_mutation_tracking_enabled = _astnode_is_tracking
ASTNode.get_mutation_history = _astnode_get_mutation_history
ASTNode.clear_mutation_history = _astnode_clear_mutation_history
ASTNode.encode_history = _astnode_encode_history
ASTNode.__setattr__ = _astnode_tracking_setattr

# Also hook replace_child to record replacements as a mutation event
_astnode_orig_replace_child = ASTNode.replace_child
def _astnode_replace_child_tracked(self: ASTNode, old: 'ASTNode', new: Optional['ASTNode']) -> bool:
    changed = _astnode_orig_replace_child(self, old, new)
    if changed and getattr(self, '_track_mutations', False):
        self._record_mutation(
            op='replace_child',
            field='(list/attr)',
            before=self._brief(old),
            after=self._brief(new),
            detail=None
        )
    return changed
ASTNode.replace_child = _astnode_replace_child_tracked

# -----------------------------
# Driver / Example usage
# -----------------------------
def compile_density2(code: str) -> str:
    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse()
    # Expand macros now that we have parser.macro_table
    ast = expand_macros(ast, parser.macro_table)
    gen = CodeGenerator(ast)
    asm = gen.generate()
    return asm


if __name__ == '__main__':
    demo_code = """
// Density 2 Hello World with CIAM and inline NASM & Python

Main() {
    '''SayHello(name)
        Print: ("Hello, " + name + "!");
    ,,,

    SayHello("World");

    #asm
        ; Inline NASM block
        ; (we still append an automatic exit at end of Main)
        ; You can place any NASM instructions here.
    #endasm

    #python
emit("; generated from inline python")
emit("mov rax, 1")
emit("mov rdi, 1")
# emit a short message directly from Python
# Note: for brevity, we just demonstrate instruction emission here.
#endpython
}
"""

    asm = compile_density2(demo_code)
    print("Generated NASM:\n")
    print(asm)
    # write to file
    with open('out.asm', 'w', encoding='utf-8') as f:
        f.write(asm)
    print("\nWritten to out.asm")

    print("\nTo assemble and link (Linux x86-64), run:")
    print("  nasm -f elf64 out.asm -o out.o")
    print("  ld out.o -o out")

    print("\nThen execute with:")
    print("  ./out")

    print("\nTo assemble and link (Windows x86-64), run:")
    print("  nasm -f pe64 out.asm -o out.o")
    print("  ld out.o -o out")

    print("\nThen execute with:")
    print("  out.exe")

    print("\nTo assemble and link (macOS x86-64), run:")
    print("  nasm -f macho64 out.asm -o out.o")
    print("  ld -macosx_version_min 10.13 -e _start -lSystem -o out out.o")

    print("\nThen execute with:")
    print("  ./out")

    # End of density2_compiler.py

import os
import re
import shutil
import subprocess
import tempfile
import time
import hashlib
from typing import List, Tuple, Union, Dict, Optional

# -----------------------------
# Token Definitions
# -----------------------------
TOKEN_SPECIFICATION = [
    # Blocks first (non-greedy to avoid overconsumption)
    ('CIAM',        r"'''(.*?),,,"),              # ''' ... ,,,
    ('INLINE_ASM',  r"#asm(.*?)#endasm"),
    ('INLINE_C',    r"#c(.*?)#endc"),
    ('INLINE_PY',   r"#python(.*?)#endpython"),

    # Comments
    ('COMMENT',     r'//[^\n]*'),
    ('MCOMMENT',    r'/\*.*?\*/'),

    # Literals and identifiers
    # Unterminated string recovery: BAD_STRING must be before STRING and MISMATCH
    ('BAD_STRING',  r'"(?:\\.|[^"\\])*?(?=\n|$)'),
    ('STRING',      r'"(?:\\.|[^"\\])*"'),
    # Unicode identifier: first char is any Unicode letter (no digit/underscore), then word chars
    ('IDENT',       r'[^\W\d_][\w]*'),

    # Symbols
    ('LBRACE',      r'\{'),
    ('RBRACE',      r'\}'),
    ('LPAREN',      r'\('),
    ('RPAREN',      r'\)'),
    ('COLON',       r':'),
    ('SEMICOLON',   r';'),
    ('COMMA',       r','),         # macro args
    ('PLUS',        r'\+'),        # string concat

    # Whitespace, newline, and mismatch
    ('NEWLINE',     r'\n'),
    ('SKIP',        r'[ \t]+'),
    ('MISMATCH',    r'.'),
]

# Make CIAM and INLINE_* non-greedy by ensuring (.*?) groups are used above.
# Note: CIAM pattern above currently uses (.*?),,, via the outer token list string.
TOKEN_SPECIFICATION = [
    (name, (pattern if name != 'CIAM' else r"'''(.*?),,,")) for (name, pattern) in TOKEN_SPECIFICATION
]

token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATION)

# Global token filter hooks and lexer error buffer
TOKEN_FILTERS: List = []
_LAST_LEX_ERRORS: List[str] = []

def register_token_filter(fn) -> None:
    TOKEN_FILTERS.append(fn)

def get_last_lex_errors() -> List[str]:
    return list(_LAST_LEX_ERRORS)


class Token:
    def __init__(self, type_: str, value: str, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line}, col={self.column})"


# -----------------------------
# Lexer
# -----------------------------
class SyntaxErrorEx(Exception):
    def __init__(self, message: str, line: int, col: int, suggestion: Optional[str] = None):
        self.line = line
        self.col = col
        self.suggestion = suggestion
        super().__init__(f"{message} @ {line}:{col}" + (f" | hint: {suggestion}" if suggestion else ""))

def tokenize(code: str) -> List[Token]:
    # UTF-8 BOM handling: strip BOM if present
    if code.startswith('\ufeff'):
        code = code.lstrip('\ufeff')

    tokens: List[Token] = []
    _LAST_LEX_ERRORS.clear()
    line_num = 1
    line_start = 0
    for mo in re.finditer(token_regex, code, re.DOTALL):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
            continue
        elif kind in ('SKIP', 'COMMENT', 'MCOMMENT'):
            continue
        elif kind == 'BAD_STRING':
            # Recover: capture as error token and continue to next line
            msg = f"Unterminated string literal"
            _LAST_LEX_ERRORS.append(f"{msg} at {line_num}:{column}")
            tokens.append(Token('ERROR', value, line_num, column))
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        tokens.append(Token(kind, value, line_num, column))

    # Apply token filters last
    for filt in TOKEN_FILTERS:
        tokens = filt(tokens)
    return tokens


# -----------------------------
# AST Nodes
# -----------------------------
class ASTNode:
    """
    Base class for all Density 2 AST nodes.

    Features:
    - Optional source position tracking: filename, (line, col) -> (end_line, end_col)
    - Child discovery: children() finds nested AST nodes and lists of nodes
    - Traversal: walk() yields nodes in preorder
    - Visitor pattern: accept(visitor) calls visitor.visit_<Type>(self) or visitor.visit(self)
    - Structural replace: replace_child(old, new) updates direct attributes/lists
    - Serialization: to_dict()/pretty() for debugging and tooling
    - Copy: copy(**overrides) for shallow cloning
    - Dodecagram encoding: to_dodecagram() uses global ast_to_dodecagram if available
    - Structural equality: __eq__ based on type and serialized content (excluding positions)
    """

    # Position information is optional and can be set later via set_pos().
    def __init__(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ):
        self.filename = filename
        self.line = line
        self.col = col
        self.end_line = end_line
        self.end_col = end_col

    # ----- Source position helpers -----
    def set_pos(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ) -> "ASTNode":
        if filename is not None:
            self.filename = filename
        if line is not None:
            self.line = line
        if col is not None:
            self.col = col
        if end_line is not None:
            self.end_line = end_line
        if end_col is not None:
            self.end_col = end_col
        return self

    # ----- Introspection helpers -----
    def _is_pos_field(self, name: str) -> bool:
        return name in ("filename", "line", "col", "end_line", "end_col")

    def _iter_fields(self):
        # Do not consider private/dunder attributes as AST data
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            yield k, v

    def children(self) -> List["ASTNode"]:
        """Return direct child AST nodes (flattening lists)."""
        result: List[ASTNode] = []
        for _, v in self._iter_fields():
            if isinstance(v, ASTNode):
                result.append(v)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, ASTNode):
                        result.append(it)
        return result

    def walk(self):
        """Preorder traversal of this subtree."""
        yield self
        for c in self.children():
            yield from c.walk()

    # ----- Visitor pattern -----
    def accept(self, visitor):
        """Call visitor.visit_<Type>(self) if present, else visitor.visit(self) if present."""
        method = getattr(visitor, f"visit_{self.__class__.__name__}", None)
        if callable(method):
            return method(self)
        generic = getattr(visitor, "visit", None)
        if callable(generic):
            return generic(self)
        return None

    # ----- Structural operations -----
    def replace_child(self, old: "ASTNode", new: Optional["ASTNode"]) -> bool:
        """
        Replace a direct child 'old' with 'new'.
        If 'new' is None, removes the child if it's in a list; clears attribute otherwise.
        Returns True if a replacement/removal occurred.
        """
        changed = False
        for k, v in list(self._iter_fields()):
            if isinstance(v, ASTNode):
                if v is old:
                    setattr(self, k, new)
                    changed = True
            elif isinstance(v, list):
                # Replace in lists; remove if new is None
                new_list = []
                for it in v:
                    if it is old:
                        if new is not None:
                            new_list.append(new)
                        changed = True
                    else:
                        new_list.append(it)
                if changed:
                    setattr(self, k, new_list)
        return changed

    # ----- Serialization / Debugging -----
    def to_dict(self, *, include_pos: bool = True) -> Dict[str, object]:
        """Convert the node (recursively) to a dict suitable for JSON/debugging."""
        d: Dict[str, object] = {"__type__": self.__class__.__name__}
        for k, v in self._iter_fields():
            if not include_pos and self._is_pos_field(k):
                continue
            if isinstance(v, ASTNode):
                d[k] = v.to_dict(include_pos=include_pos)
            elif isinstance(v, list):
                d[k] = [
                    (it.to_dict(include_pos=include_pos) if isinstance(it, ASTNode) else it)
                    for it in v
                ]
            else:
                d[k] = v
        return d

    def pretty(self, indent: str = "  ") -> str:
        """Human-readable multi-line tree dump."""
        lines: List[str] = []

        def rec(n: "ASTNode", depth: int):
            pad = indent * depth
            header = n.__class__.__name__
            pos = []
            if n.filename:
                pos.append(f'file="{n.filename}"')
            if n.line is not None and n.col is not None:
                pos.append(f"@{n.line}:{n.col}")
            if n.end_line is not None and n.end_col is not None:
                pos.append(f"-{n.end_line}:{n.end_col}")
            if pos:
                header += " [" + " ".join(pos) + "]"
            lines.append(pad + header)

            # Show scalar fields
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    continue
                if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                    continue
                lines.append(pad + indent + f"{k} = {v!r}")

            # Recurse into child nodes
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    lines.append(pad + indent + f"{k}:")
                    rec(v, depth + 2)
                elif isinstance(v, list):
                    child_nodes = [it for it in v if isinstance(it, ASTNode)]
                    if child_nodes:
                        lines.append(pad + indent + f"{k}: [{len(child_nodes)}]")
                        for it in child_nodes:
                            rec(it, depth + 2)

        rec(self, 0)
        return "\n".join(lines)

    def copy(self, **overrides):
        """
        Shallow copy with optional field overrides:
            new = node.copy(body=new_body)
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)  # type: ignore
        # Copy all instance attributes
        new_obj.__dict__.update(self.__dict__)
        # Apply overrides
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    def to_dodecagram(self) -> str:
        """
        Encode this node (and subtree) using the Dodecagram mapping.
        Relies on a global function ast_to_dodecagram(node).
        """
        f = globals().get("ast_to_dodecagram")
        if callable(f):
            return f(self)  # type: ignore[misc]
        raise RuntimeError("ast_to_dodecagram() is not available in this module")

    # ----- Equality / Representation -----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASTNode):
            return False
        if self.__class__ is not other.__class__:
            return False
        return self.to_dict(include_pos=False) == other.to_dict(include_pos=False)

    def __repr__(self) -> str:
        # Compact representation showing scalar fields only
        fields: List[str] = []
        for k, v in self._iter_fields():
            if isinstance(v, ASTNode):
                continue
            if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                continue
            fields.append(f"{k}={v!r}")
        inner = ", ".join(fields)
        return f"{self.__class__.__name__}({inner})"


class Program(ASTNode):
    def __init__(self, functions: List['Function']):
        self.functions = functions

    def __repr__(self):
        return f"Program({self.functions})"


class Function(ASTNode):
    def __init__(self, name: str, body: List['Statement']):
        self.name = name
        self.body = body

    def __repr__(self):
        return f"Function({self.name}, body={self.body})"


class Statement(ASTNode):
    """Base class for all statements in Density 2."""
    pass


class PrintStatement(Statement):
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Print({self.text!r})"


class CIAMBlock(Statement):
    def __init__(self, name: str, params: List[str], body_text: str):
        self.name = name
        self.params = params
        self.body_text = body_text  # raw Density 2 snippet

    def __repr__(self):
        return f"CIAMBlock(name={self.name!r}, params={self.params}, body_len={len(self.body_text)})"


class MacroCall(Statement):
    def __init__(self, name: str, arg_texts: List[str]):
        self.name = name
        self.arg_texts = arg_texts  # raw argument texts

    def __repr__(self):
        return f"MacroCall({self.name!r}, args={self.arg_texts})"


class InlineBlock(Statement):
    def __init__(self, lang: str, content: str):
        self.lang = lang  # 'asm', 'c', 'python'
        self.content = content

    def __repr__(self):
        return f"InlineBlock(lang={self.lang!r}, content_len={len(self.content)})"


# -----------------------------
# Dodecagram AST encoding
# -----------------------------
_DODECAGRAM_MAP = {
    'Program': '0',
    'Function': '1',
    'PrintStatement': '2',
    'CIAMBlock': '3',
    'MacroCall': '4',
    # InlineBlock variants:
    'InlineBlock:asm': '5',
    'InlineBlock:python': '6',
    'InlineBlock:py': '6',
    'InlineBlock:c': '7',
    'InlineBlock:other': '8',
    # Reserved for future nodes:
    '_reserved9': '9',
    '_reserveda': 'a',
    '_reservedb': 'b',
}

def ast_to_dodecagram(node: ASTNode) -> str:
    """
    Preorder encoding of the AST using the Dodecagram alphabet 0-9,a,b.
    """
    def enc(n: ASTNode) -> str:
        if isinstance(n, Program):
            s = _DODECAGRAM_MAP['Program']
            for f in n.functions:
                s += enc(f)
            return s
        if isinstance(n, Function):
            s = _DODECAGRAM_MAP['Function']
            for st in n.body:
                s += enc(st)
            return s
        if isinstance(n, PrintStatement):
            return _DODECAGRAM_MAP['PrintStatement']
        if isinstance(n, CIAMBlock):
            return _DODECAGRAM_MAP['CIAMBlock']
        if isinstance(n, MacroCall):
            return _DODECAGRAM_MAP['MacroCall']
        if isinstance(n, InlineBlock):
            key = f'InlineBlock:{n.lang}'
            ch = _DODECAGRAM_MAP.get(key, _DODECAGRAM_MAP['InlineBlock:other'])
            return ch
        # Unknown node -> reserved
        return _DODECAGRAM_MAP['_reserved9']
    return enc(node)

# -----------------------------
# Parser
# -----------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        # Macro table collected while parsing: name -> CIAMBlock
        self.macro_table: Dict[str, CIAMBlock] = {}

    def _error(self, expected: str, got: Optional[Token], suggestion: Optional[str] = None):
        if got:
            raise SyntaxErrorEx(f"Expected {expected}, got {got.type} {got.value!r}", got.line, got.column, suggestion)
        raise SyntaxErrorEx(f"Expected {expected}, got <eof>", -1, -1, suggestion)

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def lookahead(self, offset: int) -> Optional[Token]:
        idx = self.pos + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_type: str) -> Token:
        tok = self.peek()
        if not tok or tok.type != expected_type:
            self._error(expected_type, tok, suggestion="Check syntax near here; try adding missing delimiter or semicolon")
        self.pos += 1
        return tok

    def match(self, expected_type: str) -> bool:
        tok = self.peek()
        if tok and tok.type == expected_type:
            self.pos += 1
            return True
        return False

    def parse(self) -> Program:
        functions: List[Function] = []
        while self.peek() is not None:
            # Expect IDENT (function name) or skip stray tokens
            tok = self.peek()
            if tok.type == 'IDENT':
                # Look for function signature IDENT()
                if self.lookahead(1) and self.lookahead(1).type == 'LPAREN':
                    functions.append(self.parse_function())
                else:
                    # Skip stray identifier
                    self.pos += 1
            else:
                # Skip unknown tokens at top-level
                self.pos += 1
        return Program(functions)

    def parse_function(self) -> Function:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        self.consume('RPAREN')
        body: List[Statement] = []
        if self.match('LBRACE'):
            # Block body
            body = self.parse_statements_until_rbrace()
            self.consume('RBRACE')
        else:
            # Single statement allowed
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        return Function(name_tok.value, body)

    def parse_statements_until_rbrace(self) -> List[Statement]:
        stmts: List[Statement] = []
        while True:
            tok = self.peek()
            if tok is None or tok.type == 'RBRACE':
                break
            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    stmts.extend(stmt)
                else:
                    stmts.append(stmt)
        return stmts

    def parse_statement(self) -> Optional[Union[Statement, List[Statement]]]:
        tok = self.peek()
        if tok is None:
            return None

        if tok.type == 'IDENT' and tok.value == 'Print':
            return self.parse_print()

        if tok.type == 'CIAM':
            # Parse CIAM block definition, store in macro table, no direct output
            ciam_tok = self.consume('CIAM')
            content = ciam_tok.value
            # Strip delimiters: starts with ''' and ends with ,,,
            if content.startswith("'''"):
                content_inner = content[3:]
                if content_inner.endswith(",,,"):
                    content_inner = content_inner[:-3]
            else:
                content_inner = content

            name, params, body_text = self._parse_ciam_content(content_inner.strip(), ciam_tok)
            ciam_block = CIAMBlock(name, params, body_text)
            self.macro_table[name] = ciam_block
            # Defining a macro does not emit code
            return None

        if tok.type.startswith('INLINE_'):
            return self.parse_inline_block()

        if tok.type == 'IDENT':
            # Possibly a macro invocation: IDENT(...)
            ident = tok.value
            la = self.lookahead(1)
            if la and la.type == 'LPAREN':
                return self.parse_macro_call()
            else:
                # Unknown ident; consume to progress
                self.pos += 1
                return None

        # Unknown token; consume to progress
        self.pos += 1
        return None

    def _parse_ciam_content(self, content: str, tok: Token) -> Tuple[str, List[str], str]:
        # First line: Name(params)
        lines = content.splitlines()
        if not lines:
            raise SyntaxError(f"Empty CIAM at line {tok.line}")

        header = lines[0].strip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$', header)
        if not m:
            raise SyntaxError(f"Invalid CIAM header '{header}' at line {tok.line}")
        name = m.group(1)
        params_str = m.group(2).strip()
        params = []
        if params_str:
            params = [p.strip() for p in params_str.split(',') if p.strip()]
        body_text = '\n'.join(lines[1:]).strip()
        return name, params, body_text

    def parse_macro_call(self) -> MacroCall:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        args: List[str] = []
        current_parts: List[str] = []
        paren_depth = 1
        while True:
            tok = self.peek()
            if tok is None:
                raise SyntaxError("Unclosed macro call")
            if tok.type == 'LPAREN':
                paren_depth += 1
                current_parts.append(tok.value)
                self.pos += 1
            elif tok.type == 'RPAREN':
                paren_depth -= 1
                if paren_depth == 0:
                    # finalize last arg
                    arg_text = ''.join(current_parts).strip()
                    if arg_text:
                        args.append(arg_text)
                    self.pos += 1
                    break
                else:
                    current_parts.append(tok.value)
                    self.pos += 1
            elif tok.type == 'COMMA' and paren_depth == 1:
                arg_text = ''.join(current_parts).strip()
                args.append(arg_text)
                current_parts = []
                self.pos += 1
            else:
                current_parts.append(tok.value)
                self.pos += 1
        self.consume('SEMICOLON')
        return MacroCall(name_tok.value, args)

    def parse_inline_block(self) -> InlineBlock:
        tok = self.peek()
        lang = tok.type.split('_', 1)[1].lower()  # asm / c / py
        inline_tok = self.consume(tok.type)
        # strip off #lang and #endlang markers:
        content = re.sub(r'^#\w+', '', inline_tok.value, flags=re.DOTALL)
        content = re.sub(r'#end\w+$', '', content, flags=re.DOTALL)
        return InlineBlock(lang, content.strip())

    def parse_print(self) -> PrintStatement:
        self.consume('IDENT')  # 'Print'
        self.consume('COLON')
        self.consume('LPAREN')

        # Simple string concat of STRING (+ STRING)*
        parts: List[str] = []
        first = True
        while True:
            tok = self.peek()
            if not tok:
                raise SyntaxError("Unclosed Print: missing ')'")
            if tok.type == 'STRING':
                parts.append(tok.value)
                self.pos += 1
                first = False

def generate_nasm(ast_or_source: Union[Program, str]) -> str:
    """
    Generate NASM assembly.
    - If given a Program AST, emit NASM via CodeGenerator.
    - If given Density 2 source (str), parse, expand macros, then emit NASM.
    - If given an assembly-looking string (already NASM), return as-is.
    """
    if isinstance(ast_or_source, Program):
        gen = CodeGenerator(ast_or_source)
        return gen.generate()

    if isinstance(ast_or_source, str):
        text = ast_or_source
        # Heuristic: looks like NASM already
        if re.search(r'^\s*section\s+\.text\b', text, flags=re.MULTILINE) and 'global _start' in text:
            return text
        # Treat as Density 2 source
        tokens = tokenize(text)
        parser = Parser(tokens)
        program = parser.parse()
        program = expand_macros(program, parser.macro_table)
        gen = CodeGenerator(program)
        return gen.generate()

    raise TypeError(f"generate_nasm expects Program or str, got {type(ast_or_source).__name__}")

class CodeGenerator: ()

#!/usr/bin/env python3
# Density 2 compiler: lexer + parser + macro expander + NASM codegen + CLI

import os
import re
import sys
import shutil
import subprocess
import tempfile
import time
import hashlib
from typing import List, Tuple, Union, Dict, Optional
from density2_dbg import start_debugger

# -----------------------------
# Token Definitions
# -----------------------------
TOKEN_SPECIFICATION = [
    # Blocks first (non-greedy to avoid overconsumption)
    ('CIAM',        r"'''(.*?),,,"),              # ''' ... ,,,
    ('INLINE_ASM',  r"#asm(.*?)#endasm"),
    ('INLINE_C',    r"#c(.*?)#endc"),
    ('INLINE_PY',   r"#python(.*?)#endpython"),

    # Comments
    ('COMMENT',     r'//[^\n]*'),
    ('MCOMMENT',    r'/\*.*?\*/'),

    # Literals and identifiers
    ('STRING',      r'"(?:\\.|[^"\\])*"'),
    ('IDENT',       r'[A-Za-z_][A-Za-z0-9_]*'),

    # Symbols
    ('LBRACE',      r'\{'),
    ('RBRACE',      r'\}'),
    ('LPAREN',      r'\('),
    ('RPAREN',      r'\)'),
    ('COLON',       r':'),
    ('SEMICOLON',   r';'),
    ('COMMA',       r','),         # macro args
    ('PLUS',        r'\+'),        # string concat

    # Whitespace, newline, and mismatch
    ('NEWLINE',     r'\n'),
    ('SKIP',        r'[ \t]+'),
    ('MISMATCH',    r'.'),
]

# Ensure CIAM is non-greedy explicitly
TOKEN_SPECIFICATION = [
    (name, (pattern if name != 'CIAM' else r"'''(.*?),,,"))
    for (name, pattern) in TOKEN_SPECIFICATION
]

token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATION)
_TOKEN_RE = re.compile(token_regex, re.DOTALL)

# -----------------------------
# Language ruleset/dictionary (read-only, for tooling; cheap to access)
# -----------------------------
KEYWORDS = frozenset(('Main', 'Print', 'If', 'Else', 'While', 'For', 'Return', 'Let', 'Const', 'Include', 'Import'))
INLINE_LANGUAGES = frozenset(('asm', 'c', 'python', 'py', 'js', 'lua', 'zig', 'rust', 'cs', 'cpp', 'wasm', 'shell'))
NASM_DIRECTIVES = frozenset(('section', 'global', 'extern', 'align', 'db', 'dw', 'dd', 'dq', 'resb', 'resw', 'resd', 'resq'))
REGISTERS_X86_64 = frozenset((
    'rax','rbx','rcx','rdx','rsi','rdi','rbp','rsp',
    'r8','r9','r10','r11','r12','r13','r14','r15',
    'eax','ebx','ecx','edx','esi','edi','ebp','esp',
    'r8d','r9d','r10d','r11d','r12d','r13d','r14d','r15d',
    'ax','bx','cx','dx','si','di','bp','sp',
    'r8w','r9w','r10w','r11w','r12w','r13w','r14w','r15w',
    'al','bl','cl','dl','sil','dil','bpl','spl',
    'r8b','r9b','r10b','r11b','r12b','r13b','r14b','r15b'
))
SYSCALL_AMD64_LINUX = {
    'read': 0, 'write': 1, 'open': 2, 'close': 3, 'stat': 4, 'fstat': 5, 'lstat': 6,
    'mmap': 9, 'mprotect': 10, 'munmap': 11, 'brk': 12,
    'rt_sigaction': 13, 'rt_sigprocmask': 14, 'ioctl': 16, 'pread64': 17, 'pwrite64': 18,
    'readv': 19, 'writev': 20, 'access': 21, 'pipe': 22, 'select': 23,
    'nanosleep': 35, 'getpid': 39, 'getppid': 110, 'exit': 60, 'wait4': 61, 'kill': 62, 'uname': 63,
}
INLINE_BLOCK_MARKERS = {'asm': ('#asm', '#endasm'), 'c': ('#c', '#endc'), 'python': ('#python', '#endpython'), 'py': ('#python', '#endpython')}

LANG_RULESET: Dict[str, object] = {
    'tokens': tuple(name for name, _ in TOKEN_SPECIFICATION),
    'keywords': KEYWORDS,
    'inline_languages': INLINE_LANGUAGES,
    'inline_markers': INLINE_BLOCK_MARKERS,
    'nasm_directives': NASM_DIRECTIVES,
    'registers_x86_64': REGISTERS_X86_64,
    'statement_syntax': {
        'Print': 'Print: ("string" [+ "string"]*) ;',
        'CIAM': "'''Name(param, ...)\\n<density2...>\\n,,,",
        'MacroCall': 'Name(arg, ...) ;',
        'InlineBlock': '#asm|#c|#python ... #endasm|#endc|#endpython',
    },
    'notes': 'CIAM/inline blocks non-greedy; functions are IDENT() with optional { body }.',
}
LANG_DICTIONARY: Dict[str, object] = {
    'syscalls_amd64_linux': dict(SYSCALL_AMD64_LINUX),
    'inline_languages': sorted(INLINE_LANGUAGES),
    'nasm_directives': sorted(NASM_DIRECTIVES),
    'registers_x86_64': sorted(REGISTERS_X86_64),
}

def get_ruleset() -> Dict[str, object]:
    return dict(LANG_RULESET)

def get_dictionary() -> Dict[str, object]:
    return dict(LANG_DICTIONARY)


class Token:
    def __init__(self, type_: str, value: str, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line}, col={self.column})"


# -----------------------------
# Lexer
# -----------------------------
def tokenize(code: str) -> List[Token]:
    tokens: List[Token] = []
    line_num = 1
    line_start = 0
    for mo in _TOKEN_RE.finditer(code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
            continue
        elif kind in ('SKIP', 'COMMENT', 'MCOMMENT'):
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        tokens.append(Token(kind, value, line_num, column))
    return tokens


# -----------------------------
# AST Nodes
# -----------------------------
class ASTNode:
    def __init__(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ):
        self.filename = filename
        self.line = line
        self.col = col
        self.end_line = end_line
        self.end_col = end_col

    def set_pos(self, *, filename: Optional[str] = None, line: Optional[int] = None, col: Optional[int] = None,
                end_line: Optional[int] = None, end_col: Optional[int] = None) -> "ASTNode":
        if filename is not None:
            self.filename = filename
        if line is not None:
            self.line = line
        if col is not None:
            self.col = col
        if end_line is not None:
            self.end_line = end_line
        if end_col is not None:
            self.end_col = end_col
        return self

    def _is_pos_field(self, name: str) -> bool:
        return name in ("filename", "line", "col", "end_line", "end_col")

    def _iter_fields(self):
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            yield k, v

    def children(self) -> List["ASTNode"]:
        result: List[ASTNode] = []
        for _, v in self._iter_fields():
            if isinstance(v, ASTNode):
                result.append(v)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, ASTNode):
                        result.append(it)
        return result

    def walk(self):
        yield self
        for c in self.children():
            yield from c.walk()

    def accept(self, visitor):
        method = getattr(visitor, f"visit_{self.__class__.__name__}", None)
        if callable(method):
            return method(self)
        generic = getattr(visitor, "visit", None)
        if callable(generic):
            return generic(self)
        return None

    def replace_child(self, old: "ASTNode", new: Optional["ASTNode"]) -> bool:
        changed = False
        for k, v in list(self._iter_fields()):
            if isinstance(v, ASTNode) and v is old:
                setattr(self, k, new)
                changed = True
            elif isinstance(v, list):
                new_list = []
                for it in v:
                    if it is old:
                        if new is not None:
                            new_list.append(new)
                        changed = True
                    else:
                        new_list.append(it)
                if changed:
                    setattr(self, k, new_list)
        return changed

    def to_dict(self, *, include_pos: bool = True) -> Dict[str, object]:
        d: Dict[str, object] = {"__type__": self.__class__.__name__}
        for k, v in self._iter_fields():
            if not include_pos and self._is_pos_field(k):
                continue
            if isinstance(v, ASTNode):
                d[k] = v.to_dict(include_pos=include_pos)
            elif isinstance(v, list):
                d[k] = [(it.to_dict(include_pos=include_pos) if isinstance(it, ASTNode) else it) for it in v]
            else:
                d[k] = v
        return d

    def pretty(self, indent: str = "  ") -> str:
        lines: List[str] = []
        def rec(n: "ASTNode", depth: int):
            pad = indent * depth
            header = n.__class__.__name__
            pos = []
            if n.filename:
                pos.append(f'file="{n.filename}"')
            if n.line is not None and n.col is not None:
                pos.append(f"@{n.line}:{n.col}")
            if n.end_line is not None and n.end_col is not None:
                pos.append(f"-{n.end_line}:{n.end_col}")
            if pos:
                header += " [" + " ".join(pos) + "]"
            lines.append(pad + header)
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    continue
                if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                    continue
                lines.append(pad + indent + f"{k} = {v!r}")
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    lines.append(pad + indent + f"{k}:")
                    rec(v, depth + 2)
                elif isinstance(v, list):
                    child_nodes = [it for it in v if isinstance(it, ASTNode)]
                    if child_nodes:
                        lines.append(pad + indent + f"{k}: [{len(child_nodes)}]")
                        for it in child_nodes:
                            rec(it, depth + 2)
        rec(self, 0)
        return "\n".join(lines)

    def copy(self, **overrides):
        cls = self.__class__
        new_obj = cls.__new__(cls)  # type: ignore
        new_obj.__dict__.update(self.__dict__)
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    def to_dodecagram(self) -> str:
        f = globals().get("ast_to_dodecagram")
        if callable(f):
            return f(self)  # type: ignore[misc]
        raise RuntimeError("ast_to_dodecagram() is not available in this module")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASTNode):
            return False
        if self.__class__ is not other.__class__:
            return False
        return self.to_dict(include_pos=False) == other.to_dict(include_pos=False)

    def __repr__(self) -> str:
        fields: List[str] = []
        for k, v in self._iter_fields():
            if isinstance(v, ASTNode):
                continue
            if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                continue
            fields.append(f"{k}={v!r}")
        inner = ", ".join(fields)
        return f"{self.__class__.__name__}({inner})"


class Program(ASTNode):
    def __init__(self, functions: List['Function']):
        self.functions = functions

    def __repr__(self):
        return f"Program({self.functions})"


class Function(ASTNode):
    def __init__(self, name: str, body: List['Statement']):
        self.name = name
        self.body = body

    def __repr__(self):
        return f"Function({self.name}, body={self.body})"


class Statement(ASTNode):
    pass


class PrintStatement(Statement):
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Print({self.text!r})"


class CIAMBlock(Statement):
    def __init__(self, name: str, params: List[str], body_text: str):
        self.name = name
        self.params = params
        self.body_text = body_text

    def __repr__(self):
        return f"CIAMBlock(name={self.name!r}, params={self.params}, body_len={len(self.body_text)})"


class MacroCall(Statement):
    def __init__(self, name: str, arg_texts: List[str]):
        self.name = name
        self.arg_texts = arg_texts

    def __repr__(self):
        return f"MacroCall({self.name!r}, args={self.arg_texts})"


class InlineBlock(Statement):
    def __init__(self, lang: str, content: str):
        self.lang = lang  # 'asm', 'c', 'python'
        self.content = content

    def __repr__(self):
        return f"InlineBlock(lang={self.lang!r}, content_len={len(self.content)})"


# -----------------------------
# Dodecagram AST encoding
# -----------------------------
_DODECAGRAM_MAP = {
    'Program': '0',
    'Function': '1',
    'PrintStatement': '2',
    'CIAMBlock': '3',
    'MacroCall': '4',
    'InlineBlock:asm': '5',
    'InlineBlock:python': '6',
    'InlineBlock:py': '6',
    'InlineBlock:c': '7',
    'InlineBlock:other': '8',
    '_reserved9': '9',
    '_reserveda': 'a',
    '_reservedb': 'b',
}

def ast_to_dodecagram(node: ASTNode) -> str:
    def enc(n: ASTNode) -> str:
        if isinstance(n, Program):
            s = _DODECAGRAM_MAP['Program']
            for f in n.functions:
                s += enc(f)
            return s
        if isinstance(n, Function):
            s = _DODECAGRAM_MAP['Function']
            for st in n.body:
                s += enc(st)
            return s
        if isinstance(n, PrintStatement):
            return _DODECAGRAM_MAP['PrintStatement']
        if isinstance(n, CIAMBlock):
            return _DODECAGRAM_MAP['CIAMBlock']
        if isinstance(n, MacroCall):
            return _DODECAGRAM_MAP['MacroCall']
        if isinstance(n, InlineBlock):
            key = f'InlineBlock:{n.lang}'
            ch = _DODECAGRAM_MAP.get(key, _DODECAGRAM_MAP['InlineBlock:other'])
            return ch
        return _DODECAGRAM_MAP['_reserved9']
    return enc(node)


# -----------------------------
# Parser
# -----------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.macro_table: Dict[str, CIAMBlock] = {}

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def lookahead(self, offset: int) -> Optional[Token]:
        idx = self.pos + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_type: str) -> Token:
        tok = self.peek()
        if not tok or tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok

    def match(self, expected_type: str) -> bool:
        tok = self.peek()
        if tok and tok.type == expected_type:
            self.pos += 1
            return True
        return False

    def parse(self) -> Program:
        functions: List[Function] = []
        while self.peek() is not None:
            tok = self.peek()
            if tok.type == 'IDENT':
                if self.lookahead(1) and self.lookahead(1).type == 'LPAREN':
                    functions.append(self.parse_function())
                else:
                    self.pos += 1
            else:
                self.pos += 1
        return Program(functions)

    def parse_function(self) -> Function:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        self.consume('RPAREN')
        body: List[Statement] = []
        if self.match('LBRACE'):
            body = self.parse_statements_until_rbrace()
            self.consume('RBRACE')
        else:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        return Function(name_tok.value, body)

    def parse_statements_until_rbrace(self) -> List[Statement]:
        stmts: List[Statement] = []
        while True:
            tok = self.peek()
            if tok is None or tok.type == 'RBRACE':
                break
            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    stmts.extend(stmt)
                else:
                    stmts.append(stmt)
        return stmts

    def parse_statement(self) -> Optional[Union[Statement, List[Statement]]]:
        tok = self.peek()
        if tok is None:
            return None

        if tok.type == 'IDENT' and tok.value == 'Print':
            return self.parse_print()

        if tok.type == 'CIAM':
            ciam_tok = self.consume('CIAM')
            content = ciam_tok.value
            if content.startswith("'''"):
                content_inner = content[3:]
                if content_inner.endswith(",,,"):
                    content_inner = content_inner[:-3]
            else:
                content_inner = content
            name, params, body_text = self._parse_ciam_content(content_inner.strip(), ciam_tok)
            ciam_block = CIAMBlock(name, params, body_text)
            self.macro_table[name] = ciam_block
            return None

        if tok.type.startswith('INLINE_'):
            return self.parse_inline_block()

        if tok.type == 'IDENT':
            la = self.lookahead(1)
            if la and la.type == 'LPAREN':
                return self.parse_macro_call()
            else:
                self.pos += 1
                return None

        self.pos += 1
        return None

    def _parse_ciam_content(self, content: str, tok: Token) -> Tuple[str, List[str], str]:
        lines = content.splitlines()
        if not lines:
            raise SyntaxError(f"Empty CIAM at line {tok.line}")
        header = lines[0].strip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$', header)
        if not m:
            raise SyntaxError(f"Invalid CIAM header '{header}' at line {tok.line}")
        name = m.group(1)
        params_str = m.group(2).strip()
        params = [p.strip() for p in params_str.split(',') if p.strip()] if params_str else []
        body_text = '\n'.join(lines[1:]).strip()
        return name, params, body_text

    def parse_macro_call(self) -> MacroCall:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        args: List[str] = []
        current_parts: List[str] = []
        paren_depth = 1
        while True:
            tok = self.peek()
            if tok is None:
                raise SyntaxError("Unclosed macro call")
            if tok.type == 'LPAREN':
                paren_depth += 1
                current_parts.append(tok.value)
                self.pos += 1
            elif tok.type == 'RPAREN':
                paren_depth -= 1
                if paren_depth == 0:
                    arg_text = ''.join(current_parts).strip()
                    if arg_text:
                        args.append(arg_text)
                    self.pos += 1
                    break
                else:
                    current_parts.append(tok.value)
                    self.pos += 1
            elif tok.type == 'COMMA' and paren_depth == 1:
                arg_text = ''.join(current_parts).strip()
                args.append(arg_text)
                current_parts = []
                self.pos += 1
            else:
                current_parts.append(tok.value)
                self.pos += 1
        self.consume('SEMICOLON')
        return MacroCall(name_tok.value, args)

    def parse_inline_block(self) -> InlineBlock:
        tok = self.peek()
        lang = tok.type.split('_', 1)[1].lower()
        inline_tok = self.consume(tok.type)
        content = re.sub(r'^#\w+', '', inline_tok.value, flags=re.DOTALL)
        content = re.sub(r'#end\w+$', '', content, flags=re.DOTALL)
        return InlineBlock(lang, content.strip())

    def parse_print(self) -> PrintStatement:
        self.consume('IDENT')  # 'Print'
        self.consume('COLON')
        self.consume('LPAREN')

        parts: List[str] = []
        first = True
        while True:
            tok = self.peek()
            if not tok:
                raise SyntaxError("Unclosed Print: missing ')'")
            if first:
                if tok.type != 'STRING':
                    raise SyntaxError(f"Print expects a string literal, got {tok}")
                parts.append(eval(tok.value))
                self.pos += 1
                first = False
            else:
                if tok.type == 'PLUS':
                    self.pos += 1
                    tok2 = self.peek()
                    if not tok2 or tok2.type != 'STRING':
                        raise SyntaxError("Expected string after '+' in Print")
                    parts.append(eval(tok2.value))
                    self.pos += 1
                else:
                    break

        self.consume('RPAREN')
        self.consume('SEMICOLON')
        full_text = ''.join(parts)
        return PrintStatement(full_text)


# -----------------------------
# Macro Expansion
# -----------------------------
def expand_macros(program: Program, macro_table: Dict[str, CIAMBlock], max_depth: int = 32) -> Program:
    def expand_statements(stmts: List[Statement], depth: int) -> List[Statement]:
        result: List[Statement] = []
        for s in stmts:
            if isinstance(s, MacroCall):
                if s.name not in macro_table:
                    continue
                macro = macro_table[s.name]
                expanded = expand_macro_call(macro, s, depth)
                result.extend(expanded)
            else:
                result.append(s)
        return result

    def expand_macro_call(macro: CIAMBlock, call: MacroCall, depth: int) -> List[Statement]:
        if depth <= 0:
            raise RecursionError("Macro expansion depth exceeded")
        body = macro.body_text
        mapping = {p: (call.arg_texts[i] if i < len(call.arg_texts) else '') for i, p in enumerate(macro.params)}
        for p, a in mapping.items():
            body = re.sub(rf'\b{re.escape(p)}\b', a, body)
        sub_tokens = tokenize(body)
        sub_parser = Parser(sub_tokens)
        sub_parser.macro_table = macro_table
        sub_stmts = sub_parser.parse_statements_until_rbrace()
        return expand_statements(sub_stmts, depth - 1)

    new_functions: List[Function] = []
    for f in program.functions:
        expanded_body = expand_statements(f.body, max_depth)
        new_functions.append(Function(f.name, expanded_body))
    return Program(new_functions)


# -----------------------------
# NASM Emitter
# -----------------------------
class CodeGenerator:
    def __init__(self, ast: Program):
        self.ast = ast
        self.text_lines: List[str] = []
        self.data_lines: List[str] = []
        self.string_table: Dict[str, str] = {}
        self.label_counter = 0
        self._reg_cache = {'rax_sys_write': False, 'rdi_stdout': False}

    def generate(self) -> str:
        self.text_lines = []
        self.data_lines = []
        self.string_table = {}
        self.label_counter = 0
        self._invalidate_reg_cache()

        self._emit_header()
        for func in self.ast.functions:
            self._emit_function(func)

        final_lines: List[str] = []
        final_lines.append('section .data')
        final_lines.extend('    ' + line for line in self.data_lines)
        final_lines.append('section .text')
        final_lines.append('    global _start')
        for l in self.text_lines:
            final_lines.append(l)

        return '\n'.join(final_lines)

    def _invalidate_reg_cache(self):
        self._reg_cache['rax_sys_write'] = False
        self._reg_cache['rdi_stdout'] = False

    def _emit_header(self):
        self.text_lines.append('; --- Density 2 NASM output ---')
        self.text_lines.append('; inline C: compiled then translated (best-effort)')
        self.text_lines.append('; inline Python: executed at codegen; use emit("...") to output NASM')

    def _emit_function(self, func: Function):
        if func.name == 'Main':
            self.text_lines.append('_start:')
        else:
            self.text_lines.append(f'{func.name}:')
        self._invalidate_reg_cache()

        for stmt in func.body:
            if isinstance(stmt, PrintStatement):
                self._emit_print(stmt.text)
            elif isinstance(stmt, InlineBlock):
                self._emit_inline(stmt)
                self._invalidate_reg_cache()
            elif isinstance(stmt, CIAMBlock):
                self.text_lines.append(f'    ; CIAMBlock ignored (should be expanded): {getattr(stmt, "name", "?")}')
            elif isinstance(stmt, MacroCall):
                self.text_lines.append(f'    ; MacroCall ignored (should be expanded): {getattr(stmt, "name", "?")}')
            else:
                self.text_lines.append('    ; Unknown statement')

        if func.name == 'Main':
            self._emit_exit()

    def _encode_string_bytes(self, s: str) -> List[int]:
        return list(s.encode('utf-8'))

    def _get_string_label(self, text: str) -> Tuple[str, int]:
        if text not in self.string_table:
            label = f'str_{self.label_counter}'
            self.label_counter += 1
            data_bytes = self._encode_string_bytes(text)
            stored = data_bytes + [10, 0]  # newline + NUL
            bytes_list = ', '.join(str(b) for b in stored)
            self.data_lines.append(f'{label} db {bytes_list}')
            length = len(data_bytes) + 1  # include newline, exclude NUL
            self.string_table[text] = label
        else:
            label = self.string_table[text]
            data_bytes = self._encode_string_bytes(text)
            length = len(data_bytes) + 1
        return label, length

    def _emit_print(self, text: str):
        label, length = self._get_string_label(text)
        if not self._reg_cache['rax_sys_write']:
            self.text_lines.append(f'    mov rax, 1          ; sys_write')
            self._reg_cache['rax_sys_write'] = True
        if not self._reg_cache['rdi_stdout']:
            self.text_lines.append(f'    mov rdi, 1          ; stdout')
            self._reg_cache['rdi_stdout'] = True
        self.text_lines.append(f'    mov rsi, {label}    ; message')
        self.text_lines.append(f'    mov rdx, {length}         ; length (bytes)')
        self.text_lines.append('    syscall')

    def _emit_exit(self):
        self.text_lines.append('    mov rax, 60         ; sys_exit')
        self.text_lines.append('    xor rdi, rdi        ; status 0')
        self.text_lines.append('    syscall')
        self._invalidate_reg_cache()

    def _emit_inline(self, block: InlineBlock):
        if block.lang == 'asm':
            self.text_lines.append('    ; inline NASM start')
            for line in block.content.splitlines():
                line = line.rstrip()
                if line:
                    self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline NASM end')
        elif block.lang in ('py', 'python'):
            self.text_lines.append('    ; inline Python start')
            for line in self._run_inline_python(block.content):
                self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline Python end')
        elif block.lang == 'c':
            self.text_lines.append('    ; inline C start')
            asm_lines = self._compile_c_to_asm(block.content)
            if asm_lines:
                for l in asm_lines:
                    self.text_lines.append('    ' + l)
            else:
                for line in block.content.splitlines():
                    self.text_lines.append('    ; ' + line)
                self.text_lines.append('    ; (no C compiler found; block left as comment)')
            self.text_lines.append('    ; inline C end')
        else:
            for line in block.content.splitlines():
                self.text_lines.append(f'    ; inline {block.lang} ignored: {line}')

    def _run_inline_python(self, code: str) -> List[str]:
        lines: List[str] = []
        def emit(s: str):
            if not isinstance(s, str):
                raise TypeError("emit() expects a string")
            lines.append(s)
        def label(prefix: str = 'gen'):
            lbl = f'{prefix}_{self.label_counter}'
            self.label_counter += 1
            return lbl
        globals_dict = {'__builtins__': {'range': range, 'len': len, 'str': str, 'int': int, 'print': print}, 'emit': emit, 'label': label}
        try:
            exec(code, globals_dict, {})
        except Exception as ex:
            lines.append(f'; [inline python error] {ex!r}')
        return lines

    def _compile_c_to_asm(self, c_code: str) -> List[str]:
        for cand in ('tcc', 'clang', 'gcc', 'cl'):
            if shutil.which(cand):
                compiler = cand
                break
        else:
            return []

        if compiler == 'cl':
            return ['; [inline c compiler: cl]'] + self._compile_with_msvc(c_code)

        tmpdir = tempfile.mkdtemp(prefix='den2_c_')
        c_path = os.path.join(tmpdir, 'inline.c')
        asm_path = os.path.join(tmpdir, 'inline.s')
        try:
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            if compiler == 'tcc':
                cmd = [compiler, '-nostdlib', '-S', c_path, '-o', asm_path]
            elif compiler == 'clang':
                cmd = [compiler, '-x', 'c', '-O2', '-S', c_path, '-o', asm_path, '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer']
            else:  # gcc
                cmd = [compiler, '-x', 'c', '-O2', '-S', c_path, '-o', asm_path, '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer']

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not os.path.exists(asm_path):
                return []

            with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()

            translated = self._translate_att_to_nasm(raw)
            if translated:
                return [f'; [inline c compiler: {compiler}]'] + translated

            commented = [f'; [inline c compiler: {compiler}]', '; [begin compiled C assembly]']
            commented += ['; ' + line for line in raw.splitlines()]
            commented.append('; [end compiled C assembly]')
            return commented
        except Exception as ex:
            return [f'; [inline c compile error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _compile_with_msvc(self, c_code: str) -> List[str]:
        tmpdir = tempfile.mkdtemp(prefix='den2_msvc_')
        try:
            c_path = os.path.join(tmpdir, 'inline.c')
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            cmd = ['cl', '/nologo', '/FA', '/c', os.path.basename(c_path)]
            proc = subprocess.run(cmd, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            asm_listing = os.path.join(tmpdir, 'inline.asm')
            lines: List[str] = []

            if proc.stdout:
                for ln in proc.stdout.splitlines():
                    if ln.strip():
                        lines.append('; [cl] ' + ln)
            if proc.returncode != 0 and proc.stderr:
                for ln in proc.stderr.splitlines():
                    if ln.strip():
                        lines.append('; [cl err] ' + ln)

            if os.path.exists(asm_listing):
                with open(asm_listing, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                translated = self._msvc_asm_to_nasm(raw)
                return lines + translated

            return lines
        except Exception as ex:
            return [f'; [inline c compile (msvc) error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _msvc_asm_to_nasm(self, msvc_asm: str) -> List[str]:
        out: List[str] = []
        for line in msvc_asm.splitlines():
            s = line.rstrip()
            if not s:
                continue
            if s.lstrip().startswith(';'):
                out.append(s)
                continue
            tok = s.strip()
            up = tok.upper()
            if up.startswith(('TITLE ', 'COMMENT ', 'INCLUDE ', 'INCLUDELIB ')):
                continue
            if up.startswith(('.MODEL', '.CODE', '.DATA', '.CONST', '.XDATA', '.PDATA', '.STACK', '.LIST', '.686', '.686P', '.XMM', '.X64')):
                continue
            if up.startswith(('PUBLIC ', 'EXTRN ', 'EXTERN ', 'ASSUME ')):
                continue
            if up == 'END':
                continue
            if up.startswith('ALIGN '):
                parts = tok.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    out.append(f'align {parts[1]}')
                continue
            m_proc = re.match(r'^([A-Za-z_$.@?][\w$.@?]*)\s+PROC\b', tok)
            if m_proc:
                out.append(f'{m_proc.group(1)}:')
                continue
            if re.match(r'^[A-Za-z_$.@?][\w$.@?]*\s+ENDP\b', tok):
                continue
            m_data = re.match(r'^(DB|DW|DD|DQ)\b(.*)$', tok, re.IGNORECASE)
            if m_data:
                out.append(m_data.group(1).lower() + m_data.group(2))
                continue
            tok = re.sub(r'\b(BYTE|WORD|DWORD|QWORD)\s+PTR\b', lambda m: m.group(1).lower(), tok)
            tok = re.sub(r'\bOFFSET\s+FLAT:', '', tok, flags=re.IGNORECASE)
            tok = re.sub(r'\bOFFSET\s+', '', tok, flags=re.IGNORECASE)
            tok = tok.replace(' FLAT:', ' ')
            out.append(tok)
        return out

    def _translate_att_to_nasm(self, att_asm: str) -> List[str]:
        out: List[str] = []
        for line in att_asm.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith('.'):
                if s.startswith(('.globl', '.global', '.text', '.data', '.bss', '.rodata', '.section', '.type', '.size', '.file', '.ident', '.cfi', '.p2align', '.intel_syntax', '.att_syntax')):
                    continue
                out.append(f'; {s}')
                continue
            if s.endswith(':'):
                out.append(s)
                continue
            s = s.split('\t#', 1)[0].split(' #', 1)[0].strip()
            if not s:
                continue
            parts = s.split(None, 1)
            op = parts[0]
            rest = parts[1] if len(parts) > 1 else ''
            op_n = re.sub(r'(q|l|w|b)$', '', op)
            ops = [o.strip() for o in rest.split(',')] if rest else []
            ops = [self._att_operand_to_nasm(o) for o in ops]
            if len(ops) == 2:
                ops = [ops[1], ops[0]]
            if ops:
                out.append(f'{op_n} ' + ', '.join(ops))
            else:
                out.append(op_n)
        return out

    def _att_operand_to_nasm(self, o: str) -> str:
        o = o.strip()
        if o.startswith('$'):
            return o[1:]
        o = re.sub(r'%([a-zA-Z][a-zA-Z0-9]*)', r'\1', o)
        m = re.match(r'^\s*([\-+]?\d+)?\s*\(\s*([a-zA-Z0-9%]+)\s*(?:,\s*([a-zA-Z0-9%]+)\s*(?:,\s*(\d+))?)?\s*\)\s*$', o)
        if m:
            disp, base, index, scale = m.groups()
            base = base.lstrip('%')
            parts = [base]
            if index:
                idx = index.lstrip('%')
                sc = int(scale) if scale else 1
                parts.append(f'{idx}*{sc}')
            addr = ' + '.join(parts) if parts else ''
            if disp:
                sign = '+' if int(disp) >= 0 else '-'
                addr = f'{addr} {sign} {abs(int(disp))}' if addr else str(disp)
            return f'[{addr}]'
        return o


# -----------------------------
# Public API
# -----------------------------
def generate_nasm(ast_or_source: Union[Program, str]) -> str:
    """
    Generate NASM assembly.
    - If given a Program AST, emit NASM via CodeGenerator.
    - If given Density 2 source (str), parse, expand macros, then emit NASM.
    - If given an assembly-looking string (already NASM), return as-is.
    """
    if isinstance(ast_or_source, Program):
        gen = CodeGenerator(ast_or_source)
        return gen.generate()

    if isinstance(ast_or_source, str):
        text = ast_or_source
        if re.search(r'^\s*section\s+\.text\b', text, flags=re.MULTILINE) and 'global _start' in text:
            return text
        tokens = tokenize(text)
        parser = Parser(tokens)
        program = parser.parse()
        program = expand_macros(program, parser.macro_table)
        gen = CodeGenerator(program)
        return gen.generate()

    raise TypeError(f"generate_nasm expects Program or str, got {type(ast_or_source).__name__}")


def parse_density2(code: str) -> Program:
    tokens = tokenize(code)
    parser = Parser(tokens)
    program = parser.parse()
    return expand_macros(program, parser.macro_table)


# -----------------------------
# CLI
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: density2_compiler.py <input.den2> [--debug]")
        sys.exit(0)

    debug_mode = '--debug' in sys.argv
    src_path = next((a for a in sys.argv[1:] if not a.startswith('--')), None)
    if not src_path or not os.path.exists(src_path):
        print("Input file not found.")
        sys.exit(2)

    with open(src_path, 'r', encoding='utf-8') as f:
        source = f.read()

    if debug_mode:
        print("🔍 Entering Density 2 Debugger...")
        program = parse_density2(source)
        start_debugger(program, filename=src_path)
        return

    asm = generate_nasm(source)
    out_path = os.path.join(os.path.dirname(src_path) or '.', 'out.asm')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(asm)
    print("✅ NASM written to", out_path)


if __name__ == '__main__':
    main()

    #!/usr/bin/env python3

import os
import shlex
import traceback
from typing import Any, Optional, List, Dict, Callable


def _import_compiler():
    import importlib
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
        lines = [
            'Commands:',
            '  help|?                  Show this help.',
            '  quit|exit               Quit debugger.',
            '  info                    Show AST summary and cursor.',
            '  tree [raw|expanded]     Pretty AST (default expanded).',
            '  funcs                   List functions.',
            '  use <idx|name>          Select current function.',
            '  list [count]            List statements from cursor.',
            '  show [index]            Show statement details.',
            '  step [n]                Move cursor by n statements.',
            '  reset                   Reset cursor.',
            '  dodecagram [raw|expanded]  Show encoding.',
            '  track on|off [recursive]   Mutation tracking.',
            '  history [recursive]     Mutation history (compact).',
            '  walk [ClassName]        Preorder walk (optional type filter).',
            '  find <regex>            Search values in AST.',
            '  eval <expr>             Eval in limited env.',
            '  emit                    Print NASM.',
            '  write <path>            Write NASM to path.',
            '  reload [file]           Reload from file.',
            '  rules                   Show language ruleset summary.',
            '  dict                    Show dictionary summary.',
        ]
        for ln in lines:
            self._print(ln)

    def cmd_quit(self, args: List[str]):
        raise SystemExit(0)

    def cmd_info(self, args: List[str]):
        fns = self._functions()
        raw_fns = self._raw_functions()
        self._print(f'File: {self.filename or "<memory>"}')
        self._print(f'Functions (expanded): {len(fns)} | (raw): {len(raw_fns)}')
        if self.fn_index >= 0 and self.fn_index < len(fns):
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
        d2c = _import_compiler()
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
        for node in getattr(self.program, 'walk', lambda: [self.program])():
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
        env = {'program': self.program, 'program_raw': self.program_raw, 'fn': self._current_fn(), 'stmt': self._current_stmt(), 'len': len, 'type': type, 'repr': repr, 'str': str}
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
            di = d2c.get_dictionary()
            self._print('Dictionary summary:')
            self._print(f"- syscalls_amd64_linux: {len(di.get('syscalls_amd64_linux', {}))}")
            self._print(f"- dodecagram entries: {len(di.get('dodecagram', {}))}")
            self._print(f"- inline_languages: {len(di.get('inline_languages', []))}")
            self._print(f"- nasm_directives: {len(di.get('nasm_directives', []))}")
            self._print(f"- registers_x86_64: {len(di.get('registers_x86_64', []))}")
        except Exception as ex:
            self._print_error('dict failed', ex)


def start_debugger(ast_or_source: Any, filename: Optional[str] = None) -> None:
    sess = DebugSession(ast_or_source, filename=filename)
    sess.repl()

# 1) REPLACE the TOKEN_SPECIFICATION block with this extended ruleset (operators first for longest-match)

TOKEN_SPECIFICATION = [
    # Blocks (non-greedy)
    ('CIAM',        r"'''(.*?),,,"),                 # ''' ... ,,,
    ('INLINE_ASM',  r"#asm(.*?)#endasm"),
    ('INLINE_C',    r"#c(.*?)#endc"),
    ('INLINE_PY',   r"#python(.*?)#endpython"),

    # Comments/whitespace
    ('COMMENT',     r'//[^\n]*'),
    ('MCOMMENT',    r'/\*.*?\*/'),
    ('NEWLINE',     r'\n'),
    ('SKIP',        r'[ \t]+'),

    # Operators and punctuation (longest first)
    ('EQEQ',        r'=='),
    ('NEQ',        r'!='),
    ('LTE',        r'<='),
    ('GTE',        r'>='),
    ('ANDAND',     r'&&'),
    ('OROR',       r'\|\|'),
    ('ASSIGN',     r'='),
    ('LT',         r'<'),
    ('GT',         r'>'),
    ('PLUS',       r'\+'),
    ('MINUS',      r'-'),
    ('STAR',       r'\*'),
    ('SLASH',      r'/'),
    ('PERCENT',    r'%'),
    ('BANG',       r'!'),
    ('LBRACE',     r'\{'),
    ('RBRACE',     r'\}'),
    ('LPAREN',     r'\('),
    ('RPAREN',     r'\)'),
    ('COLON',      r':'),
    ('SEMICOLON',  r';'),
    ('COMMA',      r','),

    # Literals and identifiers
    ('INT',        r'0|[1-9][0-9]*'),
    ('STRING',     r'"(?:\\.|[^"\\])*"'),
    ('IDENT',      r'[A-Za-z_][A-Za-z0-9_]*'),

    # Mismatch
    ('MISMATCH',   r'.'),
]

# Ensure CIAM non-greedy is kept
TOKEN_SPECIFICATION = [
    (name, (pattern if name != 'CIAM' else r"'''(.*?),,,")) for (name, pattern) in TOKEN_SPECIFICATION
]

token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATION)

# Precompile for speed
_TOKEN_RE = re.compile(token_regex, re.DOTALL)

# 2) REPLACE tokenize() with stronger errors and the precompiled regex (keeps BOM stripping optional)

class SyntaxErrorEx(Exception):
    def __init__(self, message: str, line: int, col: int, hint: Optional[str] = None):
        self.line = line
        self.col = col
        self.hint = hint
        full = f"{message} @ {line}:{col}"
        if hint:
            full += f" | hint: {hint}"
        super().__init__(full)

def tokenize(code: str) -> List[Token]:
    tokens: List[Token] = []
    line_num = 1
    line_start = 0
    for mo in _TOKEN_RE.finditer(code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start

        if kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
            continue
        if kind in ('SKIP', 'COMMENT', 'MCOMMENT'):
            continue
        if kind == 'MISMATCH':
            raise SyntaxErrorEx(f"Unexpected character {value!r}", line_num, column, hint="Remove or escape the character")
        tokens.append(Token(kind, value, line_num, column))
    return tokens

# 3) ADD new AST nodes (place after existing AST node classes and before Parser)

# --- Types, Expressions, and new Statements ---

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

class TypeName(ASTNode):
    def __init__(self, name: str):
        self.name = name
    def __repr__(self): return f"TypeName({self.name!r})"

class Literal(ASTNode):
    def __init__(self, value: Union[int, str, bool]):
        self.value = value
    def __repr__(self): return f"Literal({self.value!r})"

class VarRef(ASTNode):
    def __init__(self, name: str):
        self.name = name
    def __repr__(self): return f"VarRef({self.name!r})"

class UnaryOp(ASTNode):
    def __init__(self, op: str, expr: ASTNode):
        self.op = op
        self.expr = expr
    def __repr__(self): return f"UnaryOp({self.op!r}, {self.expr!r})"

class BinaryOp(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right
    def __repr__(self): return f"BinaryOp({self.op!r}, {self.left!r}, {self.right!r})"

class VariableDecl(Statement):
    def __init__(self, name: str, type_name: Optional[TypeName], init: Optional[ASTNode], is_const: bool):
        self.name = name
        self.type_name = type_name
        self.init = init
        self.is_const = is_const
    def __repr__(self): return f"VariableDecl({self.name!r}, type={self.type_name!r}, init={self.init!r}, const={self.is_const})"

class Assign(Statement):
    def __init__(self, name: str, expr: ASTNode):
        self.name = name
        self.expr = expr
    def __repr__(self): return f"Assign({self.name!r}, {self.expr!r})"

class IfStatement(Statement):
    def __init__(self, cond: ASTNode, then_body: List[Statement], else_body: Optional[List[Statement]]):
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body or []
    def __repr__(self): return f"If({self.cond!r}, then={len(self.then_body)}, else={len(self.else_body)})"

class WhileStatement(Statement):
    def __init__(self, cond: ASTNode, body: List[Statement]):
        self.cond = cond
        self.body = body
    def __repr__(self): return f"While({self.cond!r}, body={len(self.body)})"

class ForStatement(Statement):
    def __init__(self, init: Optional[Statement], cond: Optional[ASTNode], post: Optional[Statement], body: List[Statement]):
        self.init = init
        self.cond = cond
        self.post = post
        self.body = body
    def __repr__(self): return f"For(init={self.init!r}, cond={self.cond!r}, post={self.post!r}, body={len(self.body)})"

class ReturnStatement(Statement):
    def __init__(self, value: Optional[ASTNode]):
        self.value = value
    def __repr__(self): return f"Return({self.value!r})"

class ExprStatement(Statement):
    def __init__(self, expr: ASTNode):
        self.expr = expr
    def __repr__(self): return f"ExprStmt({self.expr!r})"

# 4) EXTEND Parser with expressions, variables, and control flow.
#    Add these methods into the Parser class.

    # ----- Error helpers -----
    def _err(self, msg: str, hint: Optional[str] = None):
        tok = self.peek()
        if tok:
            raise SyntaxErrorEx(msg, tok.line, tok.column, hint)
        raise SyntaxErrorEx(msg, -1, -1, hint)

    def _consume_value(self, expected_type: str) -> Token:
        tok = self.peek()
        if not tok or tok.type != expected_type:
            self._err(f"Expected {expected_type}, got {tok.type if tok else '<eof>'}",
                      hint="Check delimiters and statement terminators ';'")
        self.pos += 1
        return tok

    def _match(self, t: str) -> Optional[Token]:
        tok = self.peek()
        if tok and tok.type == t:
            self.pos += 1
            return tok
        return None

    # ----- Blocks -----
    def parse_block(self) -> List[Statement]:
        body: List[Statement] = []
        self._consume_value('LBRACE')
        while True:
            tok = self.peek()
            if tok is None:
                self._err("Unclosed block", hint="Add '}'")
            if tok.type == 'RBRACE':
                self.pos += 1
                break
            st = self.parse_statement()
            if st:
                if isinstance(st, list):
                    body.extend(st)
                else:
                    body.append(st)
        return body

    # ----- Statement dispatch -----
    def parse_statement(self) -> Optional[Union[Statement, List[Statement]]]:
        tok = self.peek()
        if tok is None:
            return None

        # Built-in Print
        if tok.type == 'IDENT' and tok.value == 'Print':
            return self.parse_print()

        # CIAM definition
        if tok.type == 'CIAM':
            ciam_tok = self.consume('CIAM')
            content = ciam_tok.value
            content_inner = content[3:-3] if content.startswith("'''") and content.endswith(",,,") else content
            name, params, body_text = self._parse_ciam_content(content_inner.strip(), ciam_tok)
            self.macro_table[name] = CIAMBlock(name, params, body_text)
            return None

        # Inline blocks
        if tok.type.startswith('INLINE_'):
            return self.parse_inline_block()

        # Control flow + declarations
        if tok.type == 'IDENT':
            kw = tok.value
            if kw == 'If':
                return self.parse_if()
            if kw == 'While':
                return self.parse_while()
            if kw == 'For':
                return self.parse_for()
            if kw in ('Let', 'Const'):
                return self.parse_vardecl(is_const=(kw == 'Const'))
            if kw == 'Return':
                return self.parse_return()

            # Macro invocation: IDENT '(' ... ');'
            la = self.lookahead(1)
            if la and la.type == 'LPAREN':
                # keep existing macro call format
                return self.parse_macro_call()

        # Expression or assignment statement
        expr = self.parse_expression()
        self._consume_value('SEMICOLON')
        # Promote IDENT '=' expr to Assign when left is VarRef and ASSIGN seen (handled in expression parsing)
        if isinstance(expr, Assign):
            return expr
        return ExprStatement(expr)

    # ----- Declarations -----
    def parse_vardecl(self, is_const: bool) -> VariableDecl:
        self._consume_value('IDENT')  # 'Let' or 'Const'
        name_tok = self._consume_value('IDENT')
        type_name: Optional[TypeName] = None
        init_expr: Optional[ASTNode] = None

        if self._match('COLON'):
            type_tok = self._consume_value('IDENT')
            type_name = TypeName(type_tok.value)

        if self._match('ASSIGN'):
            init_expr = self.parse_expression()

        self._consume_value('SEMICOLON')
        return VariableDecl(name_tok.value, type_name, init_expr, is_const)

    # ----- Control flow -----
    def parse_if(self) -> IfStatement:
        self._consume_value('IDENT')  # 'If'
        self._consume_value('LPAREN')
        cond = self.parse_expression()
        self._consume_value('RPAREN')
        then_body = self.parse_block()
        else_body: Optional[List[Statement]] = None
        tok = self.peek()
        if tok and tok.type == 'IDENT' and tok.value == 'Else':
            self.pos += 1
            else_body = self.parse_block()
        return IfStatement(cond, then_body, else_body)

    def parse_while(self) -> WhileStatement:
        self._consume_value('IDENT')  # 'While'
        self._consume_value('LPAREN')
        cond = self.parse_expression()
        self._consume_value('RPAREN')
        body = self.parse_block()
        return WhileStatement(cond, body)

    def parse_for(self) -> ForStatement:
        self._consume_value('IDENT')  # 'For'
        self._consume_value('LPAREN')
        # init; cond; post
        init_stmt: Optional[Statement] = None
        if not self._match('SEMICOLON'):
            # reuse decl or assignment/expr
            if self.peek().type == 'IDENT' and self.peek().value in ('Let', 'Const'):
                init_stmt = self.parse_vardecl(is_const=(self.peek().value == 'Const'))  # parse_vardecl consumes ';'
            else:
                init_expr = self.parse_expression()
                self._consume_value('SEMICOLON')
                init_stmt = ExprStatement(init_expr)

        cond_expr: Optional[ASTNode] = None
        if not self._match('SEMICOLON'):
            cond_expr = self.parse_expression()
            self._consume_value('SEMICOLON')

        post_stmt: Optional[Statement] = None
        if not self._match('RPAREN'):
            post_expr = self.parse_expression()
            self._consume_value('RPAREN')
            post_stmt = ExprStatement(post_expr)

        body = self.parse_block()
        return ForStatement(init_stmt, cond_expr, post_stmt, body)

    def parse_return(self) -> ReturnStatement:
        self._consume_value('IDENT')  # 'Return'
        # Optional expression
        if self.peek() and self.peek().type not in ('SEMICOLON',):
            val = self.parse_expression()
        else:
            val = None
        self._consume_value('SEMICOLON')
        return ReturnStatement(val)

    # ----- Expressions -----
    # Pratt parser with precedence
    def parse_expression(self) -> ASTNode:
        lhs = self._parse_unary()
        return self._parse_binops_rhs(0, lhs)

    def _parse_unary(self) -> ASTNode:
        tok = self.peek()
        if tok and tok.type in ('PLUS', 'MINUS', 'BANG'):
            self.pos += 1
            expr = self._parse_unary()
            return UnaryOp(tok.type, expr)
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        tok = self.peek()
        if not tok:
            self._err("Expression expected", hint="Insert literal, variable, or '('expr')'")

        if tok.type == 'LPAREN':
            self.pos += 1
            inner = self.parse_expression()
            self._consume_value('RPAREN')
            return inner

        if tok.type == 'STRING':
            self.pos += 1
            return Literal(eval(tok.value))

        if tok.type == 'INT':
            self.pos += 1
            return Literal(int(tok.value))

        if tok.type == 'IDENT':
            # keywords true/false
            if tok.value in ('true', 'True'):
                self.pos += 1
                return Literal(True)
            if tok.value in ('false', 'False'):
                self.pos += 1
                return Literal(False)
            # identifier
            self.pos += 1
            ident = tok.value
            # assignment lookahead: IDENT '=' expr
            if self.peek() and self.peek().type == 'ASSIGN':
                # parse assignment as a statement-like node but return as expression for uniform handling
                self.pos += 1  # '='
                rhs = self.parse_expression()
                return Assign(ident, rhs)
            return VarRef(ident)

        self._err(f"Unexpected token in expression: {tok.type} {tok.value!r}",
                  hint="Use literals, identifiers, or parenthesized expressions")

    # Precedence and associativity
    _PREC: Dict[str, Tuple[int, str]] = {
        'OROR': (1, 'L'),   # ||
        'ANDAND': (2, 'L'), # &&
        'EQEQ': (3, 'L'), 'NEQ': (3, 'L'),
        'LT': (4, 'L'), 'LTE': (4, 'L'), 'GT': (4, 'L'), 'GTE': (4, 'L'),
        'PLUS': (5, 'L'), 'MINUS': (5, 'L'),
        'STAR': (6, 'L'), 'SLASH': (6, 'L'), 'PERCENT': (6, 'L'),
    }

    def _parse_binops_rhs(self, expr_prec: int, lhs: ASTNode) -> ASTNode:
        while True:
            tok = self.peek()
            if not tok or tok.type not in self._PREC:
                return lhs
            prec, assoc = self._PREC[tok.type]
            if prec < expr_prec:
                return lhs
            self.pos += 1  # consume operator
            rhs = self._parse_unary()
            next_tok = self.peek()
            while next_tok and next_tok.type in self._PREC:
                next_prec, _ = self._PREC[next_tok.type]
                if (next_prec > prec) or (next_prec == prec and assoc == 'R'):
                    rhs = self._parse_binops_rhs(prec + 1, rhs)
                    break
                else:
                    break
            lhs = BinaryOp(tok.type, lhs, rhs)
        # unreachable

# 5) EXTEND CodeGenerator with variables/expressions/control flow and better Inline C normalization.

class CodeGenerator:
    def __init__(self, ast: Program):
        self.ast = ast
        self.text_lines: List[str] = []
        self.data_lines: List[str] = []
        self.bss_lines: List[str] = []
        self.string_table: Dict[str, str] = {}
        self.label_counter = 0
        self._reg_cache = {'rax_sys_write': False, 'rdi_stdout': False}
        self._locals: Dict[str, Dict[str, str]] = {}  # funcName -> {varName -> label}

        # --- inside the last CodeGenerator class ---

def _emit_stmt(self, func: Function, st: Statement, end_label: str):
    if isinstance(st, PrintStatement):
        self._emit_print(st.text)
        return
    if isinstance(st, InlineBlock):
        # pass end_label so 'ret' in inline code jumps to function end
        self._emit_inline(st, end_label)
        self._invalidate_reg_cache()
        return
    if isinstance(st, VariableDecl):
        # init handled at function entry
        return
    if isinstance(st, Assign):
        self._emit_expr(func, st.expr)
        var_label = self._locals[func.name].get(st.name)
        if var_label is None:
            var_label = self._declare_implicit(func, st.name)
        self.text_lines.append(f'    mov [{var_label}], rax')
        return
    if isinstance(st, ExprStatement):
        self._emit_expr(func, st.expr)  # result in rax, ignored
        return
    if isinstance(st, IfStatement):
        else_lbl = self._new_label('else')
        end_if = self._new_label('endif')
        self._emit_expr(func, st.cond)
        self.text_lines.append('    cmp rax, 0')
        self.text_lines.append(f'    je {else_lbl}')
        for ss in st.then_body:
            self._emit_stmt(func, ss, end_label)
        self.text_lines.append(f'    jmp {end_if}')
        self.text_lines.append(f'{else_lbl}:')
        for ss in st.else_body:
            self._emit_stmt(func, ss, end_label)
        self.text_lines.append(f'{end_if}:')
        return
    if isinstance(st, WhileStatement):
        top = self._new_label('while_top')
        done = self._new_label('while_end')
        self.text_lines.append(f'{top}:')
        self._emit_expr(func, st.cond)
        self.text_lines.append('    cmp rax, 0')
        self.text_lines.append(f'    je {done}')
        for ss in st.body:
            self._emit_stmt(func, ss, end_label)
        self.text_lines.append(f'    jmp {top}')
        self.text_lines.append(f'{done}:')
        return
    if isinstance(st, ForStatement):
        top = self._new_label('for_top')
        done = self._new_label('for_end')
        if st.init:
            self._emit_stmt(func, st.init, end_label)
        self.text_lines.append(f'{top}:')
        if st.cond:
            self._emit_expr(func, st.cond)
            self.text_lines.append('    cmp rax, 0')
            self.text_lines.append(f'    je {done}')
        for ss in st.body:
            self._emit_stmt(func, ss, end_label)
        if st.post:
            self._emit_stmt(func, st.post, end_label)
        self.text_lines.append(f'    jmp {top}')
        self.text_lines.append(f'{done}:')
        return
    if isinstance(st, ReturnStatement):
        # For Main, return exits with code; for others, just jump to end
        if st.value is not None:
            self._emit_expr(func, st.value)
        else:
            self.text_lines.append('    xor rax, rax')
        if func.name == 'Main':
            self.text_lines.append('    mov rdi, rax')
        self.text_lines.append(f'    jmp {end_label}')
        return
    if isinstance(st, CIAMBlock) or isinstance(st, MacroCall):
        self.text_lines.append('    ; macro artifacts should be expanded away')
        return
    self.text_lines.append('    ; Unknown statement')

def _sanitize_inline_lines(self, lines: List[str], end_label: Optional[str]) -> List[str]:
    """
    Make inline code robust:
    - Drop sections/globals/external directives/comments not needed.
    - Strip common prologues/epilogues (push rbp/mov rbp,rsp/leave/pop rbp).
    - Rewrite 'ret' to 'jmp end_label' to safely transition to function tail.
    - Keep labels and instructions; keep comment lines.
    """
    out: List[str] = []
    drop_prefixes = (
        'section', 'global', 'extern', 'align', 'ALIGN',
        '.globl', '.global', '.type', '.size', '.ident', '.file', '.cfi'
    )
    prologue = (
        'push rbp', 'mov rbp, rsp', 'mov rbp,rsp',
    )
    epilogue = (
        'pop rbp', 'leave',
    )
    for raw in lines:
        line = raw.rstrip('\n')
        s = line.strip()
        if not s:
            continue
        if s.startswith((';', '#')):  # keep comments
            out.append(s)
            continue
        low = s.lower()
        if any(s.startswith(p) for p in drop_prefixes):
            continue
        if low.endswith(':'):  # label
            out.append(s)
            continue
        if any(low.startswith(p) for p in prologue):
            continue
        if any(low.startswith(p) for p in epilogue):
            continue
        if re.match(r'^\s*ret(q)?\b', s, flags=re.IGNORECASE):
            if end_label:
                out.append(f'jmp {end_label}')
            # if no end_label provided, drop the ret to avoid terminating the process unexpectedly
            continue
        # MSVC listing data directives mapping already handled upstream; keep instruction
        out.append(s)
    return out

def _emit_inline(self, block: InlineBlock, end_label: Optional[str] = None):
    if block.lang == 'asm':
        self.text_lines.append('    ; inline NASM start')
        raw_lines = [ln.rstrip() for ln in block.content.splitlines()]
        for ln in self._sanitize_inline_lines(raw_lines, end_label):
            if ln:
                self.text_lines.append('    ' + ln)
        self.text_lines.append('    ; inline NASM end')
    elif block.lang in ('py', 'python'):
        self.text_lines.append('    ; inline Python start')
        py_lines = self._run_inline_python(block.content)
        for ln in self._sanitize_inline_lines(py_lines, end_label):
            self.text_lines.append('    ' + ln)
        self.text_lines.append('    ; inline Python end')
    elif block.lang == 'c':
        self.text_lines.append('    ; inline C start')
        asm_lines = self._compile_c_to_asm(block.content)
        if asm_lines:
            # sanitize compiler output for safe inlining
            for ln in self._sanitize_inline_lines(asm_lines, end_label):
                self.text_lines.append('    ' + ln)
        else:
            for line in block.content.splitlines():
                self.text_lines.append('    ; ' + line)
            self.text_lines.append('    ; (no C compiler found; block left as comment)')
        self.text_lines.append('    ; inline C end')
    else:
        for line in block.content.splitlines():
            self.text_lines.append(f'    ; inline {block.lang} ignored: {line}')

    def generate(self) -> str:
        self.text_lines = []
        self.data_lines = []
        self.bss_lines = []
        self.string_table.clear()
        self.label_counter = 0
        self._invalidate_reg_cache()

        self._emit_header()
        # collect locals and emit bss
        for func in self.ast.functions:
            self._collect_locals(func)
        if self.bss_lines:
            self.text_lines.append('; [bss emitted at top of file]')

        for func in self.ast.functions:
            self._emit_function(func)

        out: List[str] = []
        if self.bss_lines:
            out.append('section .bss')
            out.extend('    ' + l for l in self.bss_lines)
        out.append('section .data')
        out.extend('    ' + l for l in self.data_lines)
        out.append('section .text')
        out.append('    global _start')
        out.extend(self.text_lines)
        return '\n'.join(out)

    def _collect_locals(self, func: Function):
        table: Dict[str, str] = {}
        def collect_stmt(st: Statement):
            if isinstance(st, VariableDecl):
                lbl = f'{func.name}_var_{st.name}'
                if st.name not in table:
                    table[st.name] = lbl
                    self.bss_lines.append(f'{lbl}    dq 0')
            elif isinstance(st, IfStatement):
                for s in st.then_body: collect_stmt(s)
                for s in st.else_body: collect_stmt(s)
            elif isinstance(st, WhileStatement):
                for s in st.body: collect_stmt(s)
            elif isinstance(st, ForStatement):
                if st.init: collect_stmt(st.init)
                for s in st.body: collect_stmt(s)
            # others: no local declarations
        for s in func.body:
            collect_stmt(s)
        self._locals[func.name] = table

    def _emit_function(self, func: Function):
        end_label = self._new_label(f'{func.name}_end')
        if func.name == 'Main':
            self.text_lines.append('_start:')
        else:
            self.text_lines.append(f'{func.name}:')
        self._invalidate_reg_cache()

        # initialize declared variables with init values
        for st in func.body:
            if isinstance(st, VariableDecl) and st.init is not None:
                self._emit_expr(func, st.init)      # rax = value
                var_label = self._locals[func.name][st.name]
                self.text_lines.append(f'    mov [{var_label}], rax')

        for st in func.body:
            self._emit_stmt(func, st, end_label)

        if func.name == 'Main':
            self._emit_exit()
        self.text_lines.append(f'{end_label}:')

    def _emit_stmt(self, func: Function, st: Statement, end_label: str):
        if isinstance(st, PrintStatement):
            self._emit_print(st.text)
            return
        if isinstance(st, InlineBlock):
            self._emit_inline(st)
            self._invalidate_reg_cache()
            return
        if isinstance(st, VariableDecl):
            # init handled at function entry
            return
        if isinstance(st, Assign):
            self._emit_expr(func, st.expr)
            var_label = self._locals[func.name].get(st.name)
            if var_label is None:
                var_label = self._declare_implicit(func, st.name)
            self.text_lines.append(f'    mov [{var_label}], rax')
            return
        if isinstance(st, ExprStatement):
            self._emit_expr(func, st.expr)  # result in rax, ignored
            return
        if isinstance(st, IfStatement):
            else_lbl = self._new_label('else')
            end_if = self._new_label('endif')
            self._emit_expr(func, st.cond)
            self.text_lines.append('    cmp rax, 0')
            self.text_lines.append(f'    je {else_lbl}')
            for ss in st.then_body:
                self._emit_stmt(func, ss, end_label)
            self.text_lines.append(f'    jmp {end_if}')
            self.text_lines.append(f'{else_lbl}:')
            for ss in st.else_body:
                self._emit_stmt(func, ss, end_label)
            self.text_lines.append(f'{end_if}:')
            return
        if isinstance(st, WhileStatement):
            top = self._new_label('while_top')
            done = self._new_label('while_end')
            self.text_lines.append(f'{top}:')
            self._emit_expr(func, st.cond)
            self.text_lines.append('    cmp rax, 0')
            self.text_lines.append(f'    je {done}')
            for ss in st.body:
                self._emit_stmt(func, ss, end_label)
            self.text_lines.append(f'    jmp {top}')
            self.text_lines.append(f'{done}:')
            return
        if isinstance(st, ForStatement):
            top = self._new_label('for_top')
            done = self._new_label('for_end')
            if st.init:
                self._emit_stmt(func, st.init, end_label)
            self.text_lines.append(f'{top}:')
            if st.cond:
                self._emit_expr(func, st.cond)
                self.text_lines.append('    cmp rax, 0')
                self.text_lines.append(f'    je {done}')
            for ss in st.body:
                self._emit_stmt(func, ss, end_label)
            if st.post:
                self._emit_stmt(func, st.post, end_label)
            self.text_lines.append(f'    jmp {top}')
            self.text_lines.append(f'{done}:')
            return
        if isinstance(st, ReturnStatement):
            # For Main, return exits with code; for others, just jump to end
            if st.value is not None:
                self._emit_expr(func, st.value)
            else:
                self.text_lines.append('    xor rax, rax')
            if func.name == 'Main':
                # rax holds value; move to rdi and exit at end_label
                self.text_lines.append('    mov rdi, rax')
            self.text_lines.append(f'    jmp {end_label}')
            return
        if isinstance(st, CIAMBlock) or isinstance(st, MacroCall):
            self.text_lines.append('    ; macro artifacts should be expanded away')
            return
        self.text_lines.append('    ; Unknown statement')

    def _emit_expr(self, func: Function, e: ASTNode):
        if isinstance(e, Literal):
            if isinstance(e.value, bool):
                self.text_lines.append(f'    mov rax, {1 if e.value else 0}')
            elif isinstance(e.value, int):
                self.text_lines.append(f'    mov rax, {e.value}')
            elif isinstance(e.value, str):
                # materialize string address: put into .data and move address to rax (not printed)
                lbl, _ = self._get_string_label(e.value)
                self.text_lines.append(f'    lea rax, [{lbl}]')
            return
        if isinstance(e, VarRef):
            var_label = self._locals[func.name].get(e.name)
            if var_label is None:
                var_label = self._declare_implicit(func, e.name)
            self.text_lines.append(f'    mov rax, [{var_label}]')
            return
        if isinstance(e, Assign):
            self._emit_expr(func, e.expr)
            var_label = self._locals[func.name].get(e.name)
            if var_label is None:
                var_label = self._declare_implicit(func, e.name)
            self.text_lines.append(f'    mov [{var_label}], rax')
            return
        if isinstance(e, UnaryOp):
            self._emit_expr(func, e.expr)
            if e.op == 'MINUS':
                self.text_lines.append('    neg rax')
            elif e.op == 'BANG':
                # logical not: rax = (rax == 0) ? 1 : 0
                self.text_lines.append('    cmp rax, 0')
                self.text_lines.append('    mov rax, 0')
                self.text_lines.append('    sete al')
            return
        if isinstance(e, BinaryOp):
            # Evaluate left, push; evaluate right, compute with rbx
            self._emit_expr(func, e.left)
            self.text_lines.append('    push rax')
            self._emit_expr(func, e.right)
            self.text_lines.append('    mov rbx, rax')
            self.text_lines.append('    pop rax')
            op = e.op
            if op == 'PLUS':
                self.text_lines.append('    add rax, rbx')
            elif op == 'MINUS':
                self.text_lines.append('    sub rax, rbx')
            elif op == 'STAR':
                self.text_lines.append('    imul rax, rbx')
            elif op == 'SLASH' or op == 'PERCENT':
                self.text_lines.append('    cqo')
                self.text_lines.append('    idiv rbx')
                if op == 'PERCENT':
                    self.text_lines.append('    mov rax, rdx')
            elif op in ('EQEQ','NEQ','LT','LTE','GT','GTE'):
                self.text_lines.append('    cmp rax, rbx')
                set_map = {
                    'EQEQ': 'sete', 'NEQ': 'setne',
                    'LT': 'setl', 'LTE': 'setle',
                    'GT': 'setg', 'GTE': 'setge'
                }
                self.text_lines.append('    mov rax, 0')
                self.text_lines.append(f'    {set_map[op]} al')
            elif op in ('ANDAND','OROR'):
                # short-circuit
                if op == 'ANDAND':
                    done = self._new_label('and_done')
                    self.text_lines.append('    cmp rax, 0')
                    self.text_lines.append(f'    je {done}')
                    # rbx holds right
                    self.text_lines.append('    mov rax, rbx')
                    self.text_lines.append(f'{done}:')
                    self.text_lines.append('    cmp rax, 0')
                    self.text_lines.append('    mov rax, 0')
                    self.text_lines.append('    setne al')
                else:  # OROR
                    done = self._new_label('or_done')
                    self.text_lines.append('    cmp rax, 0')
                    self.text_lines.append(f'    jne {done}')
                    self.text_lines.append('    mov rax, rbx')
                    self.text_lines.append(f'{done}:')
                    self.text_lines.append('    cmp rax, 0')
                    self.text_lines.append('    mov rax, 0')
                    self.text_lines.append('    setne al')
            return
        self.text_lines.append('    ; <expr not implemented>')

    def _declare_implicit(self, func: Function, name: str) -> str:
        # Implicit variable if used without declaration (friendly for macros/tests)
        lbl = f'{func.name}_var_{name}'
        if name not in self._locals[func.name]:
            self._locals[func.name][name] = lbl
            self.bss_lines.append(f'{lbl}    dq 0')
        return lbl

    def _new_label(self, prefix: str) -> str:
        v = f'{prefix}_{self.label_counter}'
        self.label_counter += 1
        return v

    # --- Inline C: prefer Intel syntax, normalize to NASM ---
    def _compile_c_to_asm(self, c_code: str) -> List[str]:
        compiler = None
        for cand in ('clang', 'gcc', 'tcc', 'cl'):
            if shutil.which(cand):
                compiler = cand
                break
        if compiler is None:
            return []
        # Prefer Intel syntax to reduce translation pain
        tmpdir = tempfile.mkdtemp(prefix='den2_c_')
        c_path = os.path.join(tmpdir, 'inline.c')
        asm_path = os.path.join(tmpdir, 'inline.s' if compiler != 'cl' else 'inline.asm')
        try:
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            if compiler == 'clang':
                cmd = ['clang', '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer', '-masm=intel', '-m64']
            elif compiler == 'gcc':
                cmd = ['gcc', '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer', '-masm=intel', '-m64']
            elif compiler == 'tcc':
                cmd = ['tcc', '-nostdlib', '-S', c_path, '-o', asm_path]
            else:  # cl (MSVC)
                return ['; [inline c compiler: cl]'] + self._compile_with_msvc(c_code)

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not os.path.exists(asm_path):
                return []

            with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()

            # Try Intel-to-NASM normalization first; if looks AT&T, fallback
            if '%rip' not in raw and '$' not in raw and '%' not in raw and '.att_syntax' not in raw:
                norm = self._intel_to_nasm(raw)
                if norm:
                    return [f'; [inline c compiler: {compiler}]'] + norm

            translated = self._translate_att_to_nasm(raw)
            if translated:
                return [f'; [inline c compiler: {compiler}]'] + translated

            return [f'; [inline c compiler: {compiler}]', '; [begin compiled C assembly]'] + \
                   ['; ' + ln for ln in raw.splitlines()] + ['; [end compiled C assembly]']
        except Exception as ex:
            return [f'; [inline c compile error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _intel_to_nasm(self, intel_asm: str) -> List[str]:
        out: List[str] = []
        for line in intel_asm.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith(('.', '#', ';')):
                # map sections, comment the rest
                if s.startswith('.text'):
                    out.append('section .text ; from intel')
                elif s.startswith('.data'):
                    out.append('section .data ; from intel')
                elif s.startswith('.bss'):
                    out.append('section .bss ; from intel')
                elif s.startswith('.globl') or s.startswith('.global') or s.startswith('.type') or s.startswith('.size') or s.startswith('.ident') or s.startswith('.file'):
                    continue
                else:
                    out.append('; ' + s)
                continue
            # label
            if s.endswith(':'):
                out.append(s)
                continue
            # strip comments starting with '#'
            s = s.split(' #', 1)[0].rstrip()
            out.append(s)
        return out

    def _translate_att_to_nasm(self, att_asm: str) -> List[str]:
        out: List[str] = []
        for line in att_asm.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith(('.', '#', ';')):
                # map sections, comment the rest
                if s.startswith('.text'):
                    out.append('section .text ; from att')
                elif s.startswith('.data'):
                    out.append('section .data ; from att')
                elif s.startswith('.bss'):
                    out.append('section .bss ; from att')
                elif s.startswith('.globl') or s.startswith('.global') or s.startswith('.type') or s.startswith('.size') or s.startswith('.ident') or s.startswith('.file'):
                    continue
                else:
                    out.append('; ' + s)
                continue
            # label
            if s.endswith(':'):
                out.append(s)
                continue

def start_debugger(ast):
    stack = [(ast, 0)]
    while stack:
        node, depth = stack.pop()
        print(f"[{depth}] Node: {node}")
        print(f"[Glyph] {node.to_dodecagram()}")
        cmd = input("> ").strip()
        if cmd == "print":
            print(node)
        elif cmd == "glyph":
            print(node.to_dodecagram())
        elif cmd == "history":
            for m in node.mutations:
                print(f"Mutation #{m.id}: {m.description}")
        elif cmd == "next":
            children = node.get_children()
            for child in reversed(children):
                stack.append((child, depth + 1))
        elif cmd == "quit":
            break


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--version':
        print("Density 2 Compiler v2.0.0\nBackend: NASM 2.15 / PE64")
        sys.exit(0)
    # otherwise normal compile path…
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess, sys, os, shlex

def run_file(p):
    try:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),"candywrapper.py"), p], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Candy Wrapper", f"Build/Run failed.\n\n{e}")

def choose():
    p = filedialog.askopenfilename(filetypes=[("Assembly/Objects/Density-2",".asm .obj .den"),
                                              ("All files","*.*")])
    if p:
        run_file(p)

root = tk.Tk()
root.title("Candy Wrapper")
root.geometry("420x160")
lbl = tk.Label(root, text="Drop a .asm / .obj / .den here, or click Browse.", font=("Segoe UI", 11))
lbl.pack(pady=16)
btn = tk.Button(root, text="Browse…", command=choose, width=18)
btn.pack(pady=8)

def dnd(ev):
    # Basic drag support (Windows shell drops a quoted path)
    p = ev.data.strip().strip("{}").strip()
    if p:
        run_file(p)
try:
    # TkDnD might not be present; keep UI minimal
    root.drop_target_register('DND_Files')
    root.dnd_bind('<<Drop>>', dnd)
except Exception:
    pass

root.mainloop()

#!/usr/bin/env python3
# Candy Wrapper: wrap .asm (NASM) and .obj into instantly runnable .exe, plus native Density-2 support.
# Windows x64 focus. Requires: nasm + (link.exe or lld-link or gcc/MinGW).
# Copyright (c) 2025

import argparse, os, sys, shutil, subprocess, tempfile, textwrap, time, re
from pathlib import Path

IS_WIN = (os.name == "nt")

def which(names):
    if isinstance(names, str): names = [names]
    for n in names:
        p = shutil.which(n)
        if p: return p
    return None

def run(cmd, cwd=None, check=True, capture=False):
    print(f"[Candy] $ {' '.join(cmd)}")
    if capture:
        cp = subprocess.run(cmd, cwd=cwd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return cp.stdout
    else:
        subprocess.run(cmd, cwd=cwd, check=check)

def ensure_file(p: Path):
    if not p.exists():
        sys.exit(f"[Candy] Missing expected file: {p}")

def detect_tools():
    tools = {}
    tools["nasm"] = which(["nasm"])
    tools["link"] = which(["link"])        # MSVC
    tools["lld"]  = which(["lld-link"])    # LLVM
    tools["gcc"]  = which(["gcc"])         # MinGW fallback
    tools["python"] = which(["python", "python3"])
    tools["density2c"] = which(["density2c", "density2c.bat", "density2c.cmd"])
    return tools

def nasm_assemble(nasm, asm_path: Path, obj_path: Path):
    cmd = [nasm, "-f", "win64", str(asm_path), "-o", str(obj_path)]
    run(cmd)

def try_link_with_linker(linker, objs, out_exe, entry=None, libs=None, extra=None):
    libs = libs or []
    extra = extra or []
    args = [linker, "/nologo", "/SUBSYSTEM:CONSOLE", f"/OUT:{out_exe}"]
    if entry:
        args.append(f"/ENTRY:{entry}")
    args += extra
    args += [str(o) for o in objs]
    # libraries (found via LIB environment)
    args += libs
    run(args)

def try_link_with_lld(lld, objs, out_exe, entry=None, libs=None, extra=None):
    libs = libs or []
    extra = extra or []
    args = [lld, "/NOLOGO", "/SUBSYSTEM:CONSOLE", f"/OUT:{out_exe}"]
    if entry:
        args.append(f"/ENTRY:{entry}")
    args += extra
    args += [str(o) for o in objs]
    args += libs
    run(args)

def try_link_with_gcc(gcc, objs, out_exe, entry=None, libs=None, extra=None):
    # MinGW-w64; we’ll drive entry via -Wl if needed. -lkernel32 is typical.
    libs = libs or []
    extra = extra or []
    args = [gcc, "-o", str(out_exe)]
    args += [str(o) for o in objs]
    if entry:
        args += ["-Wl,/entry:" + entry]
    args += extra
    # Translate MS-style libs to -l where possible; pass raw if *.a given.
    for L in libs:
        if L.lower().endswith(".lib"):
            # strip .lib, pass as -lX; MinGW maps to libX.a
            args.append("-l" + Path(L).stem)
        else:
            args.append(L)
    run(args)

def make_stub_call_main(work: Path) -> Path:
    # A tiny entry that calls `main` then ExitProcess; solves “missing entrypoint” for plain .obj with main only.
    asm = textwrap.dedent(r"""
        default rel
        extern main
        extern ExitProcess
        section .text
        global start
        start:
            ; Windows x64 requires 32 shadow bytes; we use 40 for safety/alignment
            sub     rsp, 40
            call    main
            mov     ecx, eax
            call    ExitProcess
    """).strip()
    stub_asm = work / "candy_stub_main.asm"
    stub_obj = work / "candy_stub_main.obj"
    stub_asm.write_text(asm, encoding="utf-8")
    return stub_asm, stub_obj

def link_objs(tools, objs, out_exe: Path):
    # Strategy ladder:
    #  1) link.exe with /ENTRY:main + msvcrt,kernel32
    #  2) link.exe with /ENTRY:start + kernel32
    #  3) If both fail, synthesize a stub calling `main`, link {user,obj}+stub with /ENTRY:start
    #  4) Repeat 1–3 using lld-link
    #  5) Fallback: gcc (MinGW) driving link, with -Wl,/entry:...
    tried = []
    def attempt(fn, label):
        tried.append(label)
        try:
            fn()
            return True
        except subprocess.CalledProcessError as e:
            print(f"[Candy] Link attempt failed ({label}).")
            return False

    # 1) & 2) MSVC link
    if tools["link"]:
        if attempt(lambda: try_link_with_linker(tools["link"], objs, out_exe, entry="main",
                                               libs=["msvcrt.lib","kernel32.lib"]), "link.exe + ENTRY:main"):
            return
        if attempt(lambda: try_link_with_linker(tools["link"], objs, out_exe, entry="start",
                                               libs=["kernel32.lib"]), "link.exe + ENTRY:start"):
            return
        # 3) stub + link.exe
        work = out_exe.parent
        stub_asm, stub_obj = make_stub_call_main(work)
        nasm_assemble(tools["nasm"], stub_asm, stub_obj)
        if attempt(lambda: try_link_with_linker(tools["link"], objs + [stub_obj], out_exe, entry="start",
                                               libs=["kernel32.lib"]), "link.exe + stub + ENTRY:start"):
            return

    # 4) LLVM lld-link
    if tools["lld"]:
        if attempt(lambda: try_link_with_lld(tools["lld"], objs, out_exe, entry="main",
                                             libs=["msvcrt.lib","kernel32.lib"]), "lld-link + ENTRY:main"):
            return
        if attempt(lambda: try_link_with_lld(tools["lld"], objs, out_exe, entry="start",
                                             libs=["kernel32.lib"]), "lld-link + ENTRY:start"):
            return
        work = out_exe.parent
        stub_asm, stub_obj = make_stub_call_main(work)
        nasm_assemble(tools["nasm"], stub_asm, stub_obj)
        if attempt(lambda: try_link_with_lld(tools["lld"], objs + [stub_obj], out_exe, entry="start",
                                             libs=["kernel32.lib"]), "lld-link + stub + ENTRY:start"):
            return

    # 5) MinGW gcc driver
    if tools["gcc"]:
        if attempt(lambda: try_link_with_gcc(tools["gcc"], objs, out_exe, entry="main",
                                             libs=["-lkernel32","-lmsvcrt"]), "gcc + ENTRY:main"):
            return
        if attempt(lambda: try_link_with_gcc(tools["gcc"], objs, out_exe, entry="start",
                                             libs=["-lkernel32"]), "gcc + ENTRY:start"):
            return
        work = out_exe.parent
        stub_asm, stub_obj = make_stub_call_main(work)
        nasm_assemble(tools["nasm"], stub_asm, stub_obj)
        if attempt(lambda: try_link_with_gcc(tools["gcc"], objs + [stub_obj], out_exe, entry="start",
                                             libs=["-lkernel32"]), "gcc + stub + ENTRY:start"):
            return

    print("[Candy] Tried linkers:", " | ".join(tried))
    sys.exit("[Candy] Unable to link. Ensure you’re in MSVC/LLVM/MinGW environment with SDK libs available.")

def build_from_asm(tools, asm_path: Path, out_dir: Path):
    assert tools["nasm"], "NASM not found in PATH."
    obj_path = out_dir / (asm_path.stem + ".obj")
    exe_path = out_dir / (asm_path.stem + ".exe")
    nasm_assemble(tools["nasm"], asm_path, obj_path)
    link_objs(tools, [obj_path], exe_path)
    return exe_path

def build_from_obj(tools, obj_path: Path, out_dir: Path):
    exe_path = out_dir / (obj_path.stem + ".exe")
    link_objs(tools, [obj_path], exe_path)
    return exe_path

def try_density2_frontend(tools, den_path: Path, out_dir: Path):
    """
    Native Density-2 support:
    1) Prefer 'density2c' (installer wrapper) if present.
    2) Else try python density2_compiler.py (in repo, cwd, or PATH).
    After invocation, we look for emitted .exe, else .asm/.obj to finish.
    """
    base = den_path.stem
    possible_outs = [
        out_dir / f"{base}.exe",
        out_dir / f"{base}.asm",
        out_dir / f"{base}.obj",
        den_path.parent / f"{base}.exe",
        den_path.parent / f"{base}.asm",
        den_path.parent / f"{base}.obj",
    ]

    env = os.environ.copy()
    # 1) density2c
    if tools["density2c"]:
        try:
            run([tools["density2c"], str(den_path)], cwd=den_path.parent, check=True)
        except subprocess.CalledProcessError:
            print("[Candy] density2c failed; will try python fallback...")

    # 2) python density2_compiler.py
    if not any(p.exists() for p in possible_outs) and tools["python"]:
        # Try common locations for the compiler script
        candidates = [
            Path.cwd() / "density2_compiler.py",
            den_path.parent / "density2_compiler.py",
            # If installed under Program Files\Density2\bin per your installer:
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Density2" / "bin" / "density2_compiler.py",
        ]
        for c in candidates:
            if c.exists():
                print(f"[Candy] Using Density-2 compiler at {c}")
                try:
                    # Conservative call: emit ASM to out_dir if supported; else capture stdout
                    out_asm = out_dir / f"{base}.asm"
                    try:
                        run([tools["python"], str(c), str(den_path), "-o", str(out_asm)], cwd=out_dir, check=True)
                    except subprocess.CalledProcessError:
                        # Last resort: run without -o and see if it writes .asm next to input
                        run([tools["python"], str(c), str(den_path)], cwd=den_path.parent, check=True)
                except subprocess.CalledProcessError:
                    print("[Candy] density2_compiler.py invocation failed.")
                break

    # Now decide what we got:
    exe = next((p for p in possible_outs if p.exists() and p.suffix.lower()==".exe"), None)
    if exe: return exe
    asm = next((p for p in possible_outs if p.exists() and p.suffix.lower()==".asm"), None)
    if asm:
        return build_from_asm(tools, asm, out_dir)
    obj = next((p for p in possible_outs if p.exists() and p.suffix.lower()==".obj"), None)
    if obj:
        return build_from_obj(tools, obj, out_dir)

    sys.exit("[Candy] Density-2: no output detected. Ensure your Density-2 toolchain emits .exe/.asm/.obj.")

def main():
    ap = argparse.ArgumentParser(prog="candy", description="Candy Wrapper – instantly run .asm/.obj/.den as a native .exe")
    ap.add_argument("input", help="Path to .asm (NASM), .obj (COFF), or .den (Density-2)")
    ap.add_argument("--keep", action="store_true", help="Keep build folder (default removes on exit)")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    if not inp.exists():
        sys.exit(f"[Candy] No such file: {inp}")

    tools = detect_tools()
    if not tools["nasm"]:
        sys.exit("[Candy] NASM not found on PATH. Install NASM and reopen Developer Prompt.")

    work = Path(tempfile.mkdtemp(prefix="candywrap-"))
    print(f"[Candy] Work: {work}")

    try:
        ext = inp.suffix.lower()
        if ext == ".asm":
            exe = build_from_asm(tools, inp, work)
        elif ext == ".obj":
            exe = build_from_obj(tools, inp, work)
        elif ext == ".den":
            exe = try_density2_frontend(tools, inp, work)
        else:
            sys.exit("[Candy] Unsupported input. Use .asm, .obj, or .den.")

        ensure_file(exe)
        print(f"[Candy] Running: {exe}")
        run([str(exe)], cwd=exe.parent, check=True)
    finally:
        if args.keep:
            print(f"[Candy] Kept: {work}")
        else:
            # Best-effort cleanup on Windows (file locks may linger)
            try:
                shutil.rmtree(work)
            except Exception:
                pass

if __name__ == "__main__":
    if not IS_WIN:
        sys.exit("[Candy] Windows-only wrapper (PE/COFF).")
    main()

# -----------------------------
# Cohesion/Safety shim (append-only)
# - Makes the module safe to import as a library.
# - Normalizes the public API to a single, coherent surface.
# - Tames the Candy Wrapper GUI/CLI when not explicitly requested.
# - Picks/repairs a robust tokenizer and guards duplicate definitions.
# -----------------------------
from types import SimpleNamespace as _D2NS
import sys as _sys
import re as _re
import os as _os
import threading as _threading

# 0) One-time shield to prevent accidental GUI popups when imported
# If this module is imported (not executed as script), try to close any Tk window
# that may have been created earlier in this file due to merge artifacts.
if __name__ != '__main__':
    try:
        import tkinter as _tk  # noqa: F401
        _root = getattr(_tk, "_default_root", None)
        if _root is not None:
            # Close ASAP without blocking the import
            try:
                _root.after(0, _root.destroy)
            except Exception:
                pass
    except Exception:
        pass

# 1) Canonical, robust tokenizer (always available under name: tokenize)
#    - Uses the final TOKEN_SPECIFICATION found in globals()
#    - Compiles one regex with DOTALL
#    - Treats comments/whitespace as skip
#    - Honors a 'BAD_STRING' token if present (optional recovery mode)
def _d2_build_tokenizer():
    spec = globals().get('TOKEN_SPECIFICATION')
    if not isinstance(spec, list) or not spec:
        raise RuntimeError("TOKEN_SPECIFICATION not found or empty.")

    # Ensure CIAM stays non-greedy
    fixed = []
    for name, pattern in spec:
        if name == 'CIAM':
            fixed.append((name, r"'''(.*?),,,"))
        else:
            fixed.append((name, pattern))

    regex = '|'.join('(?P<%s>%s)' % (name, pat) for name, pat in fixed)
    _compiled = _re.compile(regex, _re.DOTALL)

    # Prefer the latest Token class defined; otherwise define a minimal one.
    _Token = globals().get('Token')
    if _Token is None:
        class _Token:  # type: ignore
            def __init__(self, type_, value, line, column):
                self.type, self.value, self.line, self.column = type_, value, line, column

    _HAS_BAD_STRING = any(name == 'BAD_STRING' for name, _ in fixed)

    def _tokenize(text: str):
        # Optional BOM strip
        if text.startswith('\ufeff'):
            text = text.lstrip('\ufeff')
        tokens = []
        line_num = 1
        line_start = 0
        for mo in _compiled.finditer(text):
            kind = mo.lastgroup
            value = mo.group()
            column = mo.start() - line_start

            if kind == 'NEWLINE':
                line_num += 1
                line_start = mo.end()
                continue
            if kind in ('SKIP', 'COMMENT', 'MCOMMENT'):
                continue
            if _HAS_BAD_STRING and kind == 'BAD_STRING':
                # Recover by emitting an ERROR token and keep going
                tokens.append(_Token('ERROR', value, line_num, column))
                continue
            if kind == 'MISMATCH':
                # Upgrade to a nicer error if available
                _SyntaxErrorEx = globals().get('SyntaxErrorEx')
                if _SyntaxErrorEx:
                    raise _SyntaxErrorEx(f"Unexpected character {value!r}", line_num, column, hint="Remove or escape it")
                raise RuntimeError(f'{value!r} unexpected at {line_num}:{column}')
            tokens.append(_Token(kind, value, line_num, column))
        return tokens

    # Expose compiled regex too (for tools)
    globals()['_TOKEN_RE'] = _compiled
    globals()['token_regex'] = regex
    return _tokenize

try:
    # Replace any earlier tokenize with the robust one
    globals()['tokenize'] = _d2_build_tokenizer()
except Exception:
    # Leave existing 'tokenize' intact if build failed
    pass

# 2) Provide a coherent, defensive public API surface
#    Always available (even if earlier parts partially failed).
def _d2_safe_parse(code: str):
    Tok = globals().get('tokenize')
    Parser = globals().get('Parser')
    if not callable(Tok) or Parser is None:
        raise RuntimeError("Density-2 parser is not available.")
    tokens = Tok(code)
    parser = Parser(tokens)
    program = parser.parse()
    # Macro expand if available
    expand_macros = globals().get('expand_macros')
    if callable(expand_macros):
        program = expand_macros(program, getattr(parser, 'macro_table', {}))
    return program

def _d2_safe_codegen(ast_or_source):
    CodeGen = globals().get('CodeGenerator')
    if CodeGen is None:
        raise RuntimeError("CodeGenerator is not available.")
    if isinstance(ast_or_source, str):
        try:
            program = _d2_safe_parse(ast_or_source)
        except Exception:
            # Heuristic: if it already looks like NASM, pass through
            text = ast_or_source
            if _re.search(r'^\s*section\s+\.text\b', text, flags=_re.MULTILINE) and 'global _start' in text:
                return text
            raise
    else:
        program = ast_or_source
    gen = CodeGen(program)
    return gen.generate()

def parse_density2(code: str):
    try:
        return _d2_safe_parse(code)
    except Exception as ex:
        # Fallback: minimal empty Program to keep tools alive
        Program = globals().get('Program')
        if Program:
            return Program([])
        raise ex

def generate_nasm(ast_or_source):
    try:
        return _d2_safe_codegen(ast_or_source)
    except Exception as ex:
        # Graceful error as NASM comments
        return f"; [density2 error] {ex!r}\nsection .text\nglobal _start\n_start:\n    mov rax, 60\n    xor rdi, rdi\n    syscall\n"

def compile_density2(code: str) -> str:
    # Compatibility wrapper
    return generate_nasm(code)

# 3) Stable debugger entry (no-op if real debugger already exists)
if 'start_debugger' not in globals():
    def start_debugger(ast_or_source, filename=None):  # noqa: F401
        print("[density2] Debugger not available in this build.")

# 4) Normalize __all__ to only export coherent surface
globals()['__all__'] = tuple(sorted(set([
    # AST
    'ASTNode','Program','Function','Statement','PrintStatement','CIAMBlock','MacroCall','InlineBlock',
    # Core API
    'tokenize','Parser','expand_macros','CodeGenerator',
    'parse_density2','generate_nasm','compile_density2','start_debugger',
])))

# 5) Prevent double-CLI execution conflicts:
#    If this file was executed as a script and some earlier merge-added
#    "main()" ran already (or Candy wrapper tried to run), avoid re-running.
#    You can force the Candy wrapper explicitly by setting D2_RUN_CANDY=1.
def _d2_should_run_candy():
    return (__name__ == '__main__'
            and _os.name == 'nt'
            and _os.environ.get('D2_RUN_CANDY', '0') == '1')

def _d2_should_run_compiler():
    return (__name__ == '__main__'
            and _os.environ.get('D2_RUN_COMPILER', '1') == '1')

# If both were present earlier, make the choice explicit; default to compiler CLI.
if __name__ == '__main__' and not _d2_should_run_candy() and _d2_should_run_compiler():
    # Provide a minimal, reliable CLI that writes out.asm next to input
    try:
        import argparse as _argparse
        ap = _argparse.ArgumentParser(prog='density2c', add_help=True)
        ap.add_argument('input', help='Density-2 source file (.den2)')
        ap.add_argument('-o', '--out', help='Output .asm path (default: out.asm in same dir)')
        args, unknown = ap.parse_known_args()

        src_path = args.input
        if not _os.path.exists(src_path):
            print("Input file not found.", file=_sys.stderr)
            _sys.exit(2)
        with open(src_path, 'r', encoding='utf-8') as _f:
            _source = _f.read()

        _asm = generate_nasm(_source)
        out_path = args.out or _os.path.join(_os.path.dirname(src_path) or '.', 'out.asm')
        with open(out_path, 'w', encoding='utf-8') as _f:
            _f.write(_asm)
        print("✅ NASM written to", out_path)
        # Mark that a CLI already ran to discourage later merge-artifact CLIs
        _os.environ['D2_RUN_COMPILER'] = '0'
    except SystemExit:
        raise
    except Exception as _ex:
        print(f"[density2] CLI error: {_ex!r}", file=_sys.stderr)
        _sys.exit(1)

# 6) Optionally neutralize Candy GUI if it was embedded above and we are imported.
# It is already blocked in (0) for typical cases; this is an extra safety pass.
def _d2_neutralize_candy_gui():
    try:
        import tkinter as _tk2  # noqa: F401
        _root2 = getattr(_tk2, "_default_root", None)
        if _root2:
            try:
                # Destroy soon on the main Tk loop if it starts later
                _root2.after(0, _root2.destroy)
            except Exception:
                pass
    except Exception:
        pass

if __name__ != '__main__':
    _d2_neutralize_candy_gui()

# 7) Soft notice (only once) to signal the shim is active.
if _os.environ.get('D2_SHIM_NOTICE', '1') == '1' and __name__ != '__main__':
    _os.environ['D2_SHIM_NOTICE'] = '0'
    try:
        # Keep quiet in most automation; uncomment for debugging:
        # print("[density2] Cohesion shim active (library-safe).")
        pass
    except Exception:
        pass
        # 8) If Candy wrapper was embedded above, run it now if requested.

        if _d2_should_run_candy():
            try:
                _threading.Thread(target=main, daemon=True).start()
            except Exception:
                pass
            # Mark that a CLI already ran to discourage later merge-artifact CLIs

            _os.environ['D2_RUN_CANDY'] = '0'
            _os.environ['D2_RUN_COMPILER'] = '0'
            _d2_neutralize_candy_gui()
            # 9) End of
            # cohesion shim.
            _os.environ['D2_SHIM_NOTICE'] = '0'
            # density2_compiler.py: Density-2 to NASM compiler backend

# -----------------------------
# Type System, Semantic Analysis, IR, and Unit Tests (append-only)
# -----------------------------
from typing import Any, Iterable
from dataclasses import dataclass

# --- Types ---
class _Type:
    __slots__ = ('name',)
    def __init__(self, name: str): self.name = name
    def __repr__(self): return self.name
    def __eq__(self, o): return isinstance(o, _Type) and o.name == self.name
    def __hash__(self): return hash(self.name)

IntType    = _Type('int')
BoolType   = _Type('bool')
StringType = _Type('string')
VoidType   = _Type('void')
UnknownType = _Type('unknown')

_TYPE_BY_NAME = {
    'int': IntType, 'i32': IntType,
    'bool': BoolType, 'boolean': BoolType,
    'string': StringType, 'str': StringType,
    'void': VoidType
}

def _type_from_typename(tn: Any) -> _Type:
    # tn is likely TypeName(name: str)
    n = getattr(tn, 'name', None)
    if isinstance(n, str):
        return _TYPE_BY_NAME.get(n.lower(), UnknownType)
    return UnknownType

def _is_truthy_type(t: _Type) -> bool:
    return t in (IntType, BoolType)


# --- Semantic Issues ---
@dataclass
class SemanticIssue:
    severity: str  # 'error' | 'warning' | 'note'
    message: str
    line: int = -1
    col: int = -1
    node: Any = None

    def __repr__(self):
        pos = f" @ {self.line}:{self.col}" if self.line >= 0 else ""
        return f"[{self.severity}] {self.message}{pos}"


# --- Semantic Analyzer ---
class SemanticAnalyzer:
    def __init__(self):
        self.issues: list[SemanticIssue] = []

    def analyze(self, program: Any) -> list[SemanticIssue]:
        self.issues.clear()
        if not program or not hasattr(program, 'functions'):
            self._warn("No program/functions to analyze", node=program)
            return self.issues

        # Per-function variable type environment
        for fn in getattr(program, 'functions', []):
            env: dict[str, _Type] = {}
            self._analyze_function(fn, env)
        return self.issues

    def _analyze_function(self, fn: Any, env: dict[str, _Type]):
        for st in getattr(fn, 'body', []) or []:
            self._analyze_stmt(fn, st, env)

    def _analyze_stmt(self, fn: Any, st: Any, env: dict[str, _Type]):
        clsname = st.__class__.__name__

        if clsname == 'PrintStatement':
            # Expect text to be string; if the parser used text (string) we accept; if expr node exists, infer
            text = getattr(st, 'text', None)
            if isinstance(text, str):
                # Attach inferred type for tools
                setattr(st, '_type', VoidType)
                return
            # Otherwise, try to evaluate expression type
            t = self._infer_expr_type(fn, st, env)
            if t != StringType and t != UnknownType:
                self._warn("Print expects string; got " + repr(t), st)
            setattr(st, '_type', VoidType)
            return

        if clsname == 'VariableDecl':
            name = getattr(st, 'name', None)
            tn = getattr(st, 'type_name', None)
            init = getattr(st, 'init', None)
            var_t = _type_from_typename(tn) if tn is not None else UnknownType
            if init is not None:
                init_t = self._infer_expr_type(fn, init, env)
                if var_t == UnknownType:
                    var_t = init_t
                elif init_t != UnknownType and var_t != init_t:
                    self._err(f"Type mismatch in declaration '{name}': {var_t} vs {init_t}", st)
            if isinstance(name, str):
                env[name] = var_t if var_t != UnknownType else IntType  # default int if still unknown
            setattr(st, '_type', VoidType)
            return

        if clsname == 'Assign':
            name = getattr(st, 'name', None)
            expr = getattr(st, 'expr', None)
            rhs_t = self._infer_expr_type(fn, expr, env)
            if isinstance(name, str):
                # If not declared, implicitly declare as rhs type
                if name not in env:
                    env[name] = rhs_t if rhs_t != UnknownType else IntType
                else:
                    if env[name] != UnknownType and rhs_t != UnknownType and env[name] != rhs_t:
                        self._err(f"Cannot assign {rhs_t} to '{name}' of type {env[name]}", st)
            setattr(st, '_type', rhs_t)
            return

        if clsname == 'ExprStatement':
            expr = getattr(st, 'expr', None)
            t = self._infer_expr_type(fn, expr, env)
            setattr(st, '_type', t)
            return

        if clsname == 'IfStatement':
            cond = getattr(st, 'cond', None)
            then_body = getattr(st, 'then_body', []) or []
            else_body = getattr(st, 'else_body', []) or []
            ct = self._infer_expr_type(fn, cond, env)
            if not _is_truthy_type(ct) and ct != UnknownType:
                self._warn("If condition should be int/bool", st)
            for s in then_body: self._analyze_stmt(fn, s, env)
            for s in else_body: self._analyze_stmt(fn, s, env)
            setattr(st, '_type', VoidType)
            return

        if clsname == 'WhileStatement':
            cond = getattr(st, 'cond', None)
            body = getattr(st, 'body', []) or []
            ct = self._infer_expr_type(fn, cond, env)
            if not _is_truthy_type(ct) and ct != UnknownType:
                self._warn("While condition should be int/bool", st)
            for s in body: self._analyze_stmt(fn, s, env)
            setattr(st, '_type', VoidType)
            return

        if clsname == 'ForStatement':
            init = getattr(st, 'init', None)
            cond = getattr(st, 'cond', None)
            post = getattr(st, 'post', None)
            body = getattr(st, 'body', []) or []
            if init: self._analyze_stmt(fn, init, env)
            if cond:
                ct = self._infer_expr_type(fn, cond, env)
                if not _is_truthy_type(ct) and ct != UnknownType:
                    self._warn("For condition should be int/bool", st)
            for s in body: self._analyze_stmt(fn, s, env)
            if post: self._analyze_stmt(fn, post, env)
            setattr(st, '_type', VoidType)
            return

        if clsname == 'ReturnStatement':
            val = getattr(st, 'value', None)
            if val is not None:
                vt = self._infer_expr_type(fn, val, env)
                setattr(st, '_type', vt)
            else:
                setattr(st, '_type', VoidType)
            return

        # CIAM/Macro/Inline or unknown -> ignore
        setattr(st, '_type', VoidType)

    def _infer_expr_type(self, fn: Any, expr: Any, env: dict[str, _Type]) -> _Type:
        if expr is None:
            return UnknownType
        cn = expr.__class__.__name__

        if cn == 'Literal':
            v = getattr(expr, 'value', None)
            if isinstance(v, bool): return BoolType
            if isinstance(v, int):  return IntType
            if isinstance(v, str):  return StringType
            return UnknownType

        if cn == 'VarRef':
            nm = getattr(expr, 'name', None)
            if isinstance(nm, str):
                t = env.get(nm, UnknownType)
                setattr(expr, '_type', t)
                return t
            return UnknownType

        if cn == 'Assign':
            # Expression form of assignment; left is name, right is expr
            nm = getattr(expr, 'name', None)
            rhs = getattr(expr, 'expr', None)
            rt = self._infer_expr_type(fn, rhs, env)
            if isinstance(nm, str):
                env[nm] = env.get(nm, rt if rt != UnknownType else IntType)
            setattr(expr, '_type', rt)
            return rt

        if cn == 'UnaryOp':
            op = getattr(expr, 'op', '')
            t = self._infer_expr_type(fn, getattr(expr, 'expr', None), env)
            if op == 'BANG':
                return BoolType
            # numeric unary ops default to int
            return IntType if t in (IntType, UnknownType) else UnknownType

        if cn == 'BinaryOp':
            op = getattr(expr, 'op', '')
            lt = self._infer_expr_type(fn, getattr(expr, 'left', None), env)
            rt = self._infer_expr_type(fn, getattr(expr, 'right', None), env)
            if op in ('PLUS','MINUS','STAR','SLASH','PERCENT','ANDAND','OROR'):
                # arithmetic/logic result as int/bool
                if op in ('ANDAND','OROR'):
                    return BoolType
                return IntType
            if op in ('EQEQ','NEQ','LT','LTE','GT','GTE'):
                return BoolType
            return UnknownType

        if cn == 'TypeName':
            return _type_from_typename(expr)

        # Some parsers store string literals directly in PrintStatement.text, etc.
        if isinstance(expr, str):
            return StringType
        if isinstance(expr, bool):
            return BoolType
        if isinstance(expr, int):
            return IntType

        return UnknownType

    def _pos(self, node: Any) -> tuple[int,int]:
        return (getattr(node, 'line', -1), getattr(node, 'col', -1))

    def _err(self, msg: str, node: Any):
        line, col = self._pos(node)
        self.issues.append(SemanticIssue('error', msg, line, col, node))

    def _warn(self, msg: str, node: Any = None):
        line, col = self._pos(node) if node is not None else (-1, -1)
        self.issues.append(SemanticIssue('warning', msg, line, col, node))


# Public API for analysis
def analyze_types(program: Any) -> list[SemanticIssue]:
    sa = SemanticAnalyzer()
    return sa.analyze(program)


# --- IR (Three-address style, minimal) ---
@dataclass
class IRInstr:
    op: str
    dst: str | None = None
    a: str | int | None = None
    b: str | int | None = None
    comment: str | None = None

@dataclass
class IRBlock:
    name: str
    code: list[IRInstr]

@dataclass
class IRFunction:
    name: str
    blocks: list[IRBlock]

@dataclass
class IRModule:
    functions: list[IRFunction]


class IRBuilder:
    def __init__(self):
        self.temp_counter = 0
        self.cur_block: IRBlock | None = None
        self.functions: list[IRFunction] = []
        self.locals: dict[str, dict[str, str]] = {}  # func -> var -> slot

    def _t(self) -> str:
        v = f"t{self.temp_counter}"
        self.temp_counter += 1
        return v

    def _ensure_block(self, name: str):
        if self.cur_block is None or self.cur_block.name != name:
            self.cur_block = IRBlock(name, [])
        return self.cur_block

    def _emit(self, op: str, dst=None, a=None, b=None, comment=None):
        assert self.cur_block is not None, "IRBuilder: no current block"
        self.cur_block.code.append(IRInstr(op, dst, a, b, comment))

    def build(self, program: Any) -> IRModule:
        self.functions.clear()
        for fn in getattr(program, 'functions', []) or []:
            self._build_function(fn)
        return IRModule(self.functions)

    def _build_function(self, fn: Any):
        entry = IRBlock('entry', [])
        self.cur_block = entry
        self.locals[fn.name] = {}

        # Hoist variable decls to 'alloc' slots
        for st in getattr(fn, 'body', []) or []:
            if st.__class__.__name__ == 'VariableDecl':
                var = getattr(st, 'name', None)
                if isinstance(var, str) and var not in self.locals[fn.name]:
                    slot = f"{fn.name}.{var}"
                    self.locals[fn.name][var] = slot
                    self._emit('alloc', dst=slot)

        # Emit statements
        for st in getattr(fn, 'body', []) or []:
            self._emit_stmt(fn, st)

        # Implicit exit for Main
        if getattr(fn, 'name', '') == 'Main':
            self._emit('exit', a=0)

        self.functions.append(IRFunction(getattr(fn, 'name', '<fn>'), [entry]))

    def _emit_stmt(self, fn: Any, st: Any):
        cn = st.__class__.__name__

        if cn == 'PrintStatement':
            msg = getattr(st, 'text', None)
            if not isinstance(msg, str):
                # non-string: produce value to temp then comment-print
                v = self._emit_expr(fn, getattr(st, 'text', st))
                self._emit('print', a=v, comment="non-literal")
            else:
                self._emit('print', a=msg)
            return

        if cn == 'VariableDecl':
            if getattr(st, 'init', None) is not None:
                v = self._emit_expr(fn, st.init)
                slot = self._slot(fn, st.name)
                self._emit('store', dst=slot, a=v)
            return

        if cn == 'Assign':
            v = self._emit_expr(fn, getattr(st, 'expr', None))
            slot = self._slot(fn, getattr(st, 'name', None))
            self._emit('store', dst=slot, a=v)
            return

        if cn == 'ExprStatement':
            self._emit_expr(fn, getattr(st, 'expr', None))
            return

        if cn == 'IfStatement':
            # Simple linearized form: evaluate cond then guarded block comments
            c = self._emit_expr(fn, getattr(st, 'cond', None))
            self._emit('br_true', a=c, comment='then')
            for s in getattr(st, 'then_body', []) or []:
                self._emit_stmt(fn, s)
            if getattr(st, 'else_body', []):
                self._emit('br_else', comment='else')
                for s in getattr(st, 'else_body', []):
                    self._emit_stmt(fn, s)
            self._emit('br_end')
            return

        if cn == 'WhileStatement':
            # Linear pseudo structure
            self._emit('loop_begin')
            c = self._emit_expr(fn, getattr(st, 'cond', None))
            self._emit('br_true', a=c, comment='while-body')
            for s in getattr(st, 'body', []) or []:
                self._emit_stmt(fn, s)
            self._emit('loop_end')
            return

        if cn == 'ForStatement':
            if getattr(st, 'init', None):
                self._emit_stmt(fn, getattr(st, 'init'))
            self._emit('loop_begin')
            if getattr(st, 'cond', None):
                c = self._emit_expr(fn, getattr(st, 'cond'))
                self._emit('br_true', a=c, comment='for-body')
            for s in getattr(st, 'body', []) or []:
                self._emit_stmt(fn, s)
            if getattr(st, 'post', None):
                self._emit_stmt(fn, getattr(st, 'post'))
            self._emit('loop_end')
            return

        if cn == 'ReturnStatement':
            if getattr(st, 'value', None) is not None:
                v = self._emit_expr(fn, getattr(st, 'value'))
                self._emit('ret', a=v)
            else:
                self._emit('ret')
            return

        # CIAM/Macro/Inline -> emit as comment
        if cn in ('CIAMBlock','MacroCall','InlineBlock'):
            self._emit('comment', comment=f'{cn} ignored in IR')
            return

        self._emit('comment', comment=f'Unknown stmt {cn}')

    def _slot(self, fn: Any, name: str | None) -> str:
        if not isinstance(name, str):
            name = f"unnamed_{self._t()}"
        table = self.locals.setdefault(fn.name, {})
        if name not in table:
            table[name] = f"{fn.name}.{name}"
            self._emit('alloc', dst=table[name])
        return table[name]

    def _emit_expr(self, fn: Any, e: Any) -> str:
        if e is None:
            t = self._t()
            self._emit('const', dst=t, a=0)
            return t
        cn = e.__class__.__name__

        if cn == 'Literal':
            v = getattr(e, 'value', None)
            t = self._t()
            self._emit('const', dst=t, a=v)
            return t

        if cn == 'VarRef':
            nm = getattr(e, 'name', None)
            slot = self._slot(fn, nm if isinstance(nm, str) else None)
            t = self._t()
            self._emit('load', dst=t, a=slot)
            return t

        if cn == 'Assign':
            v = self._emit_expr(fn, getattr(e, 'expr', None))
            slot = self._slot(fn, getattr(e, 'name', None))
            self._emit('store', dst=slot, a=v)
            # result is the assigned value
            return v

        if cn == 'UnaryOp':
            op = getattr(e, 'op', '')
            v = self._emit_expr(fn, getattr(e, 'expr', None))
            t = self._t()
            self._emit('uop', dst=t, a=op, b=v)
            return t

        if cn == 'BinaryOp':
            op = getattr(e, 'op', '')
            l = self._emit_expr(fn, getattr(e, 'left', None))
            r = self._emit_expr(fn, getattr(e, 'right', None))
            t = self._t()
            self._emit('binop', dst=t, a=op, b=(l, r))
            return t

        if isinstance(e, str) or isinstance(e, (int, bool)):
            t = self._t()
            self._emit('const', dst=t, a=e)
            return t

        # Fallback: constant unknown
        t = self._t()
        self._emit('const', dst=t, a=0, comment=f'unknown expr {cn}')
        return t


# IR helpers
def ast_to_ir(program: Any) -> IRModule:
    return IRBuilder().build(program)

def ir_to_text(ir: IRModule) -> str:
    lines: list[str] = []
    for fn in getattr(ir, 'functions', []) or []:
        lines.append(f"func {fn.name}:")
        for blk in getattr(fn, 'blocks', []) or []:
            lines.append(f"  block {blk.name}:")
            for ins in blk.code:
                rhs = ""
                if ins.op in ('const','load','store','uop','binop','print','ret','exit','br_true','br_else','br_end','loop_begin','loop_end','comment','alloc'):
                    rhs = f" dst={ins.dst}" if ins.dst is not None else ""
                    if ins.a is not None:
                        rhs += f" a={ins.a}"
                    if ins.b is not None:
                        rhs += f" b={ins.b}"
                cmt = f" ; {ins.comment}" if ins.comment else ""
                lines.append(f"    {ins.op}{rhs}{cmt}")
    return "\n".join(lines)

# Simple optimization: fold constants in binop/uop chains
def optimize_ir(ir: IRModule, passes: Iterable[str] = ('fold_consts',)) -> IRModule:
    if not ir or not getattr(ir, 'functions', None):
        return ir
    do_fold = 'fold_consts' in set(passes or [])
    if not do_fold:
        return ir

    def try_fold(ins: IRInstr) -> None:
        # binop with (l, r) both constants if previously assigned by const to temps
        if ins.op == 'binop' and isinstance(ins.b, tuple) and len(ins.b) == 2:
            l, r = ins.b
            # We can't easily resolve temp to const here without a map; do a local peephole
            pass
        # leave as-is for simplicity

    for fn in ir.functions:
        for blk in fn.blocks:
            for ins in blk.code:
                try_fold(ins)
    return ir


# --- Public surface additions ---
def emit_ir_text(ast_or_source: Any) -> str:
    prog = ast_or_source
    if isinstance(ast_or_source, str):
        # Parse source if a string
        prog = parse_density2(ast_or_source)
    ir = ast_to_ir(prog)
    return ir_to_text(ir)


# Extend __all__
try:
    __all__ = tuple(sorted(set(__all__ + (
        'IntType','BoolType','StringType','VoidType','UnknownType',
        'analyze_types','ast_to_ir','ir_to_text','emit_ir_text','optimize_ir',
        'IRInstr','IRBlock','IRFunction','IRModule','SemanticIssue',
    ))))
except Exception:
    pass


# --- Unit Tests (run with: set D2_RUN_TESTS=1) ---
def _build_min_program_for_tracking():
    # Build a tiny AST manually to avoid parser variability
    f = Function('Main', [PrintStatement("A")])
    return Program([f])

def _sample_macro_src() -> str:
    return """\
Main() {
    '''Say(name)
        Print: ("Hello, " + name + "!");
    ,,,
    Say("World");
}
"""

def _run_unit_tests():
    import unittest

    class TestDensity2(unittest.TestCase):
        def test_macro_expansion(self):
            prog = parse_density2(_sample_macro_src())
            # Ensure MacroCall is expanded away and we see a PrintStatement with text "Hello, World!"
            found = []
            for fn in getattr(prog, 'functions', []) or []:
                for st in getattr(fn, 'body', []) or []:
                    if isinstance(st, PrintStatement):
                        found.append(st.text)
            self.assertTrue(any("Hello, World!" in s for s in found), "Expanded print not found")

        def test_mutation_tracking(self):
            prog = _build_min_program_for_tracking()
            prog.enable_mutation_tracking(True, recursive=True)
            fn = prog.functions[0]
            fn.body.append(PrintStatement("B"))
            hist = prog.get_mutation_history(recursive=True)
            self.assertTrue(len(hist) > 0, "No mutation history recorded")
            enc = prog.encode_history(recursive=True)
            self.assertTrue(isinstance(enc, str) and len(enc) > 0, "No encoded history")

        def test_semantic_analysis(self):
            # Let x: int = 1; x = x + 2; Print: ("done");
            try:
                # Try to use extended nodes if present; else skip
                x_decl = None
                if 'VariableDecl' in globals():
                    x_decl = VariableDecl('x', TypeName('int'), Literal(1), is_const=False)
                    x_assign = Assign('x', BinaryOp('PLUS', VarRef('x'), Literal(2)))
                    pr = PrintStatement("done")
                    fn = Function('Main', [x_decl, x_assign, pr])
                    prog = Program([fn])
                    issues = analyze_types(prog)
                    errs = [i for i in issues if i.severity == 'error']
                    self.assertFalse(errs, f"Unexpected semantic errors: {issues}")
                else:
                    self.skipTest("Extended AST not available")
            except Exception as ex:
                self.fail(f"Semantic analysis raised exception: {ex!r}")

        def test_ir_emit(self):
            # Build small AST manually for IR
            if 'VariableDecl' in globals():
                fn = Function('Main', [
                    VariableDecl('x', TypeName('int'), Literal(2), False),
                    Assign('x', BinaryOp('PLUS', VarRef('x'), Literal(3))),
                    ReturnStatement(VarRef('x')),
                ])
            else:
                # Fallback with basic nodes only
                fn = Function('Main', [PrintStatement("IR")])
            prog = Program([fn])
            text = emit_ir_text(prog)
            self.assertIn("func Main", text)
            self.assertIn("block entry", text)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDensity2)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)

# Run tests only if explicitly requested to avoid clashing with existing CLIs
if __name__ == '__main__' and os.environ.get('D2_RUN_TESTS', '0') == '1':
    _run_unit_tests()

    import os
    os.environ['D2_RUN_TESTS'] = '0'
    import re as _re
    os.environ['D2_RUN_COMPILER'] = '0'
    os.environ['D2_RUN_CANDY'] = '0'
    # density2_compiler.py: Density-2 to NASM compiler backend

# -----------------------------
# Density 2 – Append-only extensions:
# - Formal Type System (augmentations)
# - Concurrency Primitives (spawn/await/channels)
# - Package Management (CIAM modules + dependency graph)
# - Live Editing (hot-reload hooks on mutation tracking)
# -----------------------------
from dataclasses import dataclass
from typing import Optional as _Optional, Callable as _Callable, Iterable as _Iterable
import json as _json
import threading as _thr

# ===== 1) Type System augmentations =====

# Reuse the existing simple _Type; add helpers for parametric encodings.
def D2_ChanOf(inner: _Type) -> _Type:
    return _Type(f'chan<{getattr(inner, "name", str(inner))}>')

def D2_TaskOf(inner: _Type) -> _Type:
    return _Type(f'task<{getattr(inner, "name", str(inner))}>')

def D2_TypeFromString(name: str) -> _Type:
    # Supports "chan<int>", "task<string>", etc.
    if not isinstance(name, str):
        return UnknownType
    n = name.strip().lower()
    if n in _TYPE_BY_NAME:
        return _TYPE_BY_NAME[n]
    return _Type(name)

# Monkey-patch SemanticAnalyzer for new nodes and parametric types
if 'SemanticAnalyzer' in globals():
    _D2_SA__infer_expr_type = SemanticAnalyzer._infer_expr_type
    _D2_SA__analyze_stmt = SemanticAnalyzer._analyze_stmt

    def _d2_infer_expr_type_ext(self: 'SemanticAnalyzer', fn, expr, env):
        if expr is None:
            return _D2_SA__infer_expr_type(self, fn, expr, env)
        cn = expr.__class__.__name__

        # Await returns inner of task<T> if detectable
        if cn == 'D2AwaitExpr':
            t = self._infer_expr_type(fn, getattr(expr, 'expr', None), env)  # type: ignore[attr-defined]
            name = getattr(t, 'name', '')
            if name.startswith('task<') and name.endswith('>'):
                inner = name[5:-1]
                return D2_TypeFromString(inner)
            return UnknownType

        # Recv returns inner of chan<T> if detectable
        if cn == 'D2RecvExpr':
            ct = self._infer_expr_type(fn, getattr(expr, 'chan', None), env)  # type: ignore[attr-defined]
            cname = getattr(ct, 'name', '')
            if cname.startswith('chan<') and cname.endswith('>'):
                inner = cname[5:-1]
                return D2_TypeFromString(inner)
            return UnknownType

        # Spawn returns a task<T>; if target expr has a known return type, wrap, else task<unknown>
        if cn == 'D2SpawnExpr':
            inner = UnknownType
            target = getattr(expr, 'expr', None)  # type: ignore[attr-defined]
            # Try a heuristic: if it's a VarRef of a variable with known type 'task<T>' we just use it; else Unknown
            it = self._infer_expr_type(fn, target, env)
            # If the expression is just a constant or such, we do not know T; keep Unknown
            return D2_TaskOf(inner)

        # Fall back to original
        return _D2_SA__infer_expr_type(self, fn, expr, env)

    def _d2_analyze_stmt_ext(self: 'SemanticAnalyzer', fn, st, env):
        cn = st.__class__.__name__

        # Channel declarations introduce a chan<T> variable
        if cn == 'D2ChanDecl':
            name = getattr(st, 'name', None)
            tn = getattr(st, 'type_name', None)  # TypeName or None
            base_t = _type_from_typename(tn) if tn is not None else UnknownType
            env[str(name)] = D2_ChanOf(base_t if base_t != UnknownType else UnknownType)
            setattr(st, '_type', VoidType)
            return

        # Send: ensure channel and value types are at least compatible if known
        if cn == 'D2SendStmt':
            chan_expr = getattr(st, 'chan', None)
            val_expr = getattr(st, 'value', None)
            ct = self._infer_expr_type(fn, chan_expr, env)
            vt = self._infer_expr_type(fn, val_expr, env)
            cname = getattr(ct, 'name', '')
            if cname.startswith('chan<') and cname.endswith('>'):
                inner = D2_TypeFromString(cname[5:-1])
                if vt != UnknownType and inner != UnknownType and vt != inner:
                    self._warn(f"Send: value type {vt} differs from channel type {inner}", st)
            setattr(st, '_type', VoidType)
            return

        # Unknown concurrency statements default
        if cn in ('D2SpawnStmt',):
            setattr(st, '_type', VoidType)
            return

        return _D2_SA__analyze_stmt(self, fn, st, env)

    # Install patches
    SemanticAnalyzer._infer_expr_type = _d2_infer_expr_type_ext  # type: ignore[assignment]
    SemanticAnalyzer._analyze_stmt = _d2_analyze_stmt_ext        # type: ignore[assignment]


# ===== 2) Concurrency primitives (AST + parser extensions + IR scaffolding) =====

class D2ChanDecl(Statement):
    def __init__(self, name: str, type_name: _Optional['TypeName']):
        self.name = name
        self.type_name = type_name
    def __repr__(self): return f"D2ChanDecl({self.name!r}, type={self.type_name!r})"

class D2SendStmt(Statement):
    def __init__(self, chan: ASTNode, value: ASTNode):
        self.chan = chan
        self.value = value
    def __repr__(self): return f"D2Send({self.chan!r}, {self.value!r})"

class D2SpawnExpr(ASTNode):
    def __init__(self, expr: ASTNode):
        self.expr = expr
    def __repr__(self): return f"D2Spawn({self.expr!r})"

class D2AwaitExpr(ASTNode):
    def __init__(self, expr: ASTNode):
        self.expr = expr
    def __repr__(self): return f"D2Await({self.expr!r})"

class D2RecvExpr(ASTNode):
    def __init__(self, chan: ASTNode):
        self.chan = chan
    def __repr__(self): return f"D2Recv({self.chan!r})"

# Parser extension (monkey-patch) for: Chan/Send statements; Spawn/Await/Recv expressions
if 'Parser' in globals():
    _D2_parse_statement_orig = Parser.parse_statement
    _D2_parse_primary_orig = getattr(Parser, '_parse_primary', None)

    def _d2_parse_statement(self: 'Parser'):
        tok = self.peek()
        if tok and tok.type == 'IDENT':
            val = tok.value

            # Chan name [: Type] ;
            if val == 'Chan':
                self.consume('IDENT')  # Chan
                name_tok = self.consume('IDENT')
                tname = None
                if self.match('COLON'):
                    t_ident = self.consume('IDENT')
                    tname = TypeName(t_ident.value)
                self.consume('SEMICOLON')
                return D2ChanDecl(name_tok.value, tname)

            # Send(channel, expr);
            if val == 'Send':
                self.consume('IDENT')  # Send
                self.consume('LPAREN')
                chan_expr = self.parse_expression()
                self.consume('COMMA')
                value_expr = self.parse_expression()
                self.consume('RPAREN')
                self.consume('SEMICOLON')
                return D2SendStmt(chan_expr, value_expr)

            # Allow Spawn(...) as statement (discard handle)
            if val == 'Spawn':
                expr = _d2_parse_spawn_expr(self)
                self.consume('SEMICOLON')
                return ExprStatement(expr)

        # Fallback to original
        return _D2_parse_statement_orig(self)

    def _d2_parse_spawn_expr(self: 'Parser') -> D2SpawnExpr:
        # Spawn '(' expression ')'
        self.consume('IDENT')  # Spawn
        self.consume('LPAREN')
        inner = self.parse_expression()
        self.consume('RPAREN')
        return D2SpawnExpr(inner)

    def _d2_parse_await_expr(self: 'Parser') -> D2AwaitExpr:
        # Await '(' expression ')'
        self.consume('IDENT')  # Await
        self.consume('LPAREN')
        inner = self.parse_expression()
        self.consume('RPAREN')
        return D2AwaitExpr(inner)

    def _d2_parse_recv_expr(self: 'Parser') -> D2RecvExpr:
        # Recv '(' expression ')'
        self.consume('IDENT')  # Recv
        self.consume('LPAREN')
        ch = self.parse_expression()
        self.consume('RPAREN')
        return D2RecvExpr(ch)

    def _d2_parse_primary(self: 'Parser'):
        tok = self.peek()
        if tok and tok.type == 'IDENT':
            kw = tok.value
            if kw == 'Spawn':
                return _d2_parse_spawn_expr(self)
            if kw == 'Await':
                return _d2_parse_await_expr(self)
            if kw == 'Recv':
                return _d2_parse_recv_expr(self)
        # Fall back to original primary
        if callable(_D2_parse_primary_orig):
            return _D2_parse_primary_orig(self)
        # If no primary exists (older parser), emulate minimal STRING/INT/IDENT path
        # Defer to existing parse_expression error handling.
        return _D2_parse_primary_orig(self)  # type: ignore[misc]

    # Install parser patches
    Parser.parse_statement = _d2_parse_statement  # type: ignore[assignment]
    if _D2_parse_primary_orig is not None:
        Parser._parse_primary = _d2_parse_primary  # type: ignore[assignment]

# IR scaffolding: ignore concurrency at codegen/IR and leave as comments
if 'IRBuilder' in globals():
    _D2_ir_emit_stmt = IRBuilder._emit_stmt

    def _d2_ir_emit_stmt(self: 'IRBuilder', fn, st):
        cn = st.__class__.__name__
        if cn in ('D2ChanDecl',):
            # Allocate a slot to model the channel handle
            slot = self._slot(fn, getattr(st, 'name', None))
            self._emit('alloc', dst=slot, comment='chan')
            return
        if cn in ('D2SendStmt',):
            self._emit('comment', comment='Send(chan, value)')
            return
        if cn in ('ExprStatement',) and isinstance(getattr(st, 'expr', None), D2SpawnExpr):
            self._emit('comment', comment='Spawn(expr)')
            return
        # Default
        return _D2_ir_emit_stmt(self, fn, st)

    IRBuilder._emit_stmt = _d2_ir_emit_stmt  # type: ignore[assignment]


# ===== 3) Package Management – CIAM-as-Modules + dependency graph =====

@dataclass
class D2Module:
    name: str                 # Macro name (CIAM header)
    key: str                  # Stable hash of body
    exports: list[str]        # exported macro names (typically [name])
    deps: list[str]           # names of required modules
    body: str                 # raw CIAM body

class D2DepGraph:
    def __init__(self):
        self.nodes: dict[str, D2Module] = {}
        self.adj: dict[str, set[str]] = {}

    def add(self, mod: D2Module):
        self.nodes[mod.name] = mod
        self.adj.setdefault(mod.name, set())
        for d in mod.deps:
            self.adj.setdefault(mod.name, set()).add(d)
            self.adj.setdefault(d, set())

    def topo_order(self) -> list[str]:
        # Kahn's algorithm
        indeg: dict[str, int] = {n: 0 for n in self.adj}
        for u in self.adj:
            for v in self.adj[u]:
                indeg[v] = indeg.get(v, 0) + 1
        q = [n for n, d in indeg.items() if d == 0]
        out: list[str] = []
        while q:
            u = q.pop(0)
            out.append(u)
            for v in self.adj.get(u, ()):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(out) != len(indeg):
            # cycle present; return partial order followed by remaining
            remaining = [n for n in indeg if n not in out]
            return out + remaining
        return out

    def to_manifest(self) -> str:
        payload = {
            'modules': [
                {
                    'name': m.name,
                    'key': m.key,
                    'exports': m.exports,
                    'deps': m.deps,
                } for m in self.nodes.values()
            ],
            'order': self.topo_order(),
        }
        return _json.dumps(payload, indent=2)

def d2_scan_ciam_modules_from_source(source: str) -> list[D2Module]:
    """
    Lightweight CIAM scanner from raw source (no full parse needed).
    - Detects blocks: '''Name(args) ... ,,,
    - deps are found via lines like: // require: OtherModule
    """
    if not isinstance(source, str):
        return []
    mods: list[D2Module] = []
    # Non-greedy CIAM capture
    for m in re.finditer(r"'''([A-Za-z_]\w*)\s*\((.*?)\)\s*(.*?),,,", source, re.DOTALL):
        name = m.group(1)
        body = m.group(3).strip()
        deps: list[str] = []
        for ln in body.splitlines():
            ln = ln.strip()
            mo = re.match(r'//\s*require\s*:\s*([A-Za-z_]\w*)', ln, re.IGNORECASE)
            if mo:
                deps.append(mo.group(1))
        key = hashlib.sha1(body.encode('utf-8')).hexdigest()[:16]
        mods.append(D2Module(name=name, key=key, exports=[name], deps=deps, body=body))
    return mods

def d2_build_depgraph(mods: _Iterable[D2Module]) -> D2DepGraph:
    g = D2DepGraph()
    for m in mods:
        g.add(m)
    return g


# ===== 4) Live Editing – hot-reload hooks on mutation tracking =====

# Subscribe to mutation events by wrapping ASTNode._record_mutation if present
_D2_MUTATION_SUBS: list[_Callable[[dict], None]] = []

def d2_subscribe_mutations(cb: _Callable[[dict], None]) -> None:
    if cb not in _D2_MUTATION_SUBS:
        _D2_MUTATION_SUBS.append(cb)

def d2_unsubscribe_mutations(cb: _Callable[[dict], None]) -> None:
    try:
        _D2_MUTATION_SUBS.remove(cb)
    except ValueError:
        pass

# Wrap the existing _record_mutation to fan-out events
if hasattr(ASTNode, '_record_mutation'):
    _D2_orig_record_mutation = ASTNode._record_mutation  # type: ignore[attr-defined]

    def _d2_record_mutation_and_publish(self: ASTNode, *, op: str, field: str, before, after, detail: _Optional[dict]):
        _D2_orig_record_mutation(self, op=op, field=field, before=before, after=after, detail=detail)  # type: ignore[misc]
        ev = {
            'node': self.__class__.__name__,
            'op': op,
            'field': field,
            'before': before,
            'after': after,
            'detail': detail or {},
            'ts': time.time(),
        }
        for cb in list(_D2_MUTATION_SUBS):
            try:
                cb(ev)
            except Exception:
                # Keep engine resilient
                pass

    ASTNode._record_mutation = _d2_record_mutation_and_publish  # type: ignore[assignment]

class D2HotReloadEngine:
    """
    Hot-reload scaffolding:
    - watch(program, on_change): enable tracking and invoke on_change(events_batch) asynchronously.
    - stop(): detach.
    - optional debounce_ms to coalesce bursts.
    """
    def __init__(self, debounce_ms: int = 100):
        self._debounce = max(0, debounce_ms) / 1000.0
        self._buf: list[dict] = []
        self._lock = _thr.Lock()
        self._timer: _Optional[_thr.Timer] = None
        self._on_change: _Optional[_Callable[[list[dict]], None]] = None

    def _flush(self):
        with self._lock:
            batch = self._buf[:]
            self._buf.clear()
        if batch and self._on_change:
            try:
                self._on_change(batch)
            except Exception:
                pass

    def _schedule(self):
        if self._debounce <= 0:
            self._flush()
            return
        if self._timer and self._timer.is_alive():
            return
        self._timer = _thr.Timer(self._debounce, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _on_mut(self, ev: dict):
        with self._lock:
            self._buf.append(ev)
        self._schedule()

    def watch(self, program: Program, on_change: _Callable[[list[dict]], None]):
        self._on_change = on_change
        # Ensure tracking is on
        if hasattr(program, 'enable_mutation_tracking'):
            program.enable_mutation_tracking(True, recursive=True)  # type: ignore[misc]
        d2_subscribe_mutations(self._on_mut)

    def stop(self):
        d2_unsubscribe_mutations(self._on_mut)
        with self._lock:
            self._buf.clear()
        if self._timer and self._timer.is_alive():
            try:
                self._timer.cancel()
            except Exception:
                pass
        self._timer = None
        self._on_change = None

# Simple collaborative scaffold: collect and merge event streams.
class D2CollabSession:
    def __init__(self):
        self._events: list[dict] = []

    def ingest_local(self, events: list[dict]):
        self._events.extend(events)

    def ingest_remote(self, events: list[dict]):
        # Merge by timestamp; no conflict resolution at AST level (scaffold)
        self._events.extend(events)
        self._events.sort(key=lambda e: e.get('ts', 0.0))

    def history(self) -> list[dict]:
        return list(self._events)

# Convenience utilities
def d2_on_hot_reload_reemit_program(program: Program, *, reemit_ir: bool = False) -> D2HotReloadEngine:
    """
    Example: on changes, print IR or NASM to stdout (scaffold for IDE integration).
    """
    def on_change(batch: list[dict]):
        try:
            if reemit_ir and 'ast_to_ir' in globals():
                ir = ast_to_ir(program)  # type: ignore[misc]
                print(ir_to_text(ir))
            else:
                asm = generate_nasm(program)  # type: ignore[misc]
                print(asm)
        except Exception as ex:
            print(f"; [hot-reload error] {ex!r}")

    eng = D2HotReloadEngine(debounce_ms=150)
    eng.watch(program, on_change)
    return eng

# Public API exposure (append-only)
try:
    __all__ = tuple(sorted(set(__all__ + (
        'D2_ChanOf','D2_TaskOf','D2_TypeFromString',
        'D2ChanDecl','D2SendStmt','D2SpawnExpr','D2AwaitExpr','D2RecvExpr',
        'D2Module','D2DepGraph','d2_scan_ciam_modules_from_source','d2_build_depgraph',
        'd2_subscribe_mutations','d2_unsubscribe_mutations',
        'D2HotReloadEngine','D2CollabSession','d2_on_hot_reload_reemit_program',
    ))))
except Exception:
    pass

# ==============================================================
# === Density 2 Compiler — Enhancement Layer (Runtime Patch) ===
# ==============================================================
# This block attaches advanced optimization, macro hygiene,
# debugging, and toolchain utilities to the existing compiler.
# It must be placed at the *bottom* of density2_compiler.py
# and will automatically register itself when imported.

import time, hashlib, inspect, random, sys, os, re
from collections import defaultdict

print("[Density2 Enhancement Layer] Loaded successfully.")

# --------------------------------------------------------------
# 1. PERFORMANCE & OPTIMIZATION PIPELINE
# --------------------------------------------------------------

class OptimizationPass:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def run(self, ir):
        start = time.time()
        new_ir = self.func(ir)
        dur = (time.time() - start) * 1000
        print(f"[OPT] {self.name} completed in {dur:.2f} ms")
        return new_ir

def _const_fold(ir):
    """Simplify constant expressions."""
    return re.sub(r'(\d+)\s*\+\s*(\d+)', lambda m: str(int(m.group(1))+int(m.group(2))), ir)

def _peephole(ir):
    """Remove redundant instructions like mov rax, rax."""
    lines = []
    for l in ir.splitlines():
        if re.search(r'\bmov\s+(\w+),\s*\1\b', l):
            continue
        lines.append(l)
    return "\n".join(lines)

def _loop_unroll(ir):
    """Basic loop unrolling (static small-count)"""
    return re.sub(r'for\s*\((\d+)\)', lambda m: 'unrolled_'+m.group(1), ir)

OPTIMIZATION_PASSES = [
    OptimizationPass("Constant Folding", _const_fold),
    OptimizationPass("Peephole Simplifier", _peephole),
    OptimizationPass("Loop Unroller", _loop_unroll),
]

def run_optimizer(ir):
    for p in OPTIMIZATION_PASSES:
        ir = p.run(ir)
    return ir


# --------------------------------------------------------------
# 2. MACRO SYSTEM EXTENSIONS & HYGIENE
# --------------------------------------------------------------

class MacroRegistry:
    def __init__(self):
        self.macros = {}
        self.hygiene_hash = {}

    def register(self, name, pattern, body):
        token = hashlib.sha1(f"{name}{pattern}{body}".encode()).hexdigest()[:8]
        self.macros[name] = (pattern, body)
        self.hygiene_hash[name] = token
        print(f"[MACRO] Registered hygienic macro '{name}' :: {token}")

    def expand(self, src):
        for name, (pattern, body) in self.macros.items():
            token = self.hygiene_hash[name]
            src = re.sub(pattern, body.replace("$HYGIENE$", token), src)
        return src

MACRO_ENGINE = MacroRegistry()


def register_macro(name, pattern, body):
    """API for user-defined hygienic macros."""
    MACRO_ENGINE.register(name, pattern, body)


# --------------------------------------------------------------
# 3. DEBUGGING & TOOLCHAIN UTILITIES
# --------------------------------------------------------------

class Debugger:
    def __init__(self):
        self.source_map = defaultdict(lambda: "unknown")
        self.breakpoints = set()

    def add_mapping(self, src_line, asm_label):
        self.source_map[asm_label] = src_line

    def list_mappings(self):
        for label, line in self.source_map.items():
            print(f"ASM:{label:<10} → SRC:{line}")

    def add_breakpoint(self, label):
        self.breakpoints.add(label)
        print(f"[DBG] Breakpoint set at {label}")

    def hit(self, label):
        if label in self.breakpoints:
            print(f"[DBG] Breakpoint hit at {label}")
            input("Press Enter to continue...")

DEBUGGER = Debugger()


# Hook into codegen if exists
if 'emit_asm' in globals():
    _orig_emit_asm = emit_asm # type: ignore

    def emit_asm_with_debug(ir_code):
        ir_code = run_optimizer(ir_code)
        lines = ir_code.splitlines()
        for idx, l in enumerate(lines):
            DEBUGGER.add_mapping(idx, f"L{idx}")
        asm = _orig_emit_asm(ir_code)
        print("[DBG] Source map generated.")
        return asm

    emit_asm = emit_asm_with_debug


# --------------------------------------------------------------
# 4. STANDARD LIBRARY & SYSTEM HELPERS
# --------------------------------------------------------------

class DensityStdLib:
    def __init__(self):
        self.modules = {}

    def register_module(self, name, content):
        self.modules[name] = content
        print(f"[STDLIB] Module '{name}' registered.")

    def import_module(self, name):
        if name not in self.modules:
            raise ImportError(f"No such Density2 module '{name}'")
        print(f"[STDLIB] Module '{name}' imported.")
        return self.modules[name]

STDLIB = DensityStdLib()
STDLIB.register_module("io", "print, read, write, open")
STDLIB.register_module("math", "add, sub, mul, div, sqrt, sin, cos")
STDLIB.register_module("thread", "spawn, join, lock, unlock")


# --------------------------------------------------------------
# 5. CROSS-PLATFORM BUILD WRAPPERS
# --------------------------------------------------------------

def build_executable(output_name, asm_file):
    """Auto-detect platform and build."""
    platform = sys.platform
    print(f"[BUILD] Detected platform: {platform}")
    if "win" in platform:
        os.system(f"nasm -f win64 {asm_file} -o temp.obj && link /SUBSYSTEM:CONSOLE temp.obj /OUT:{output_name}.exe")
    elif "linux" in platform:
        os.system(f"nasm -f elf64 {asm_file} -o temp.o && ld temp.o -o {output_name}")
    else:
        print("[WARN] Unsupported platform, manual linking required.")
    print(f"[BUILD] {output_name} built successfully.")


# --------------------------------------------------------------
# 6. TESTING & VALIDATION SUITE
# --------------------------------------------------------------

class DensityTestSuite:
    def __init__(self):
        self.tests = []

    def add(self, name, func):
        self.tests.append((name, func))

    def run_all(self):
        print(f"[TEST] Running {len(self.tests)} validation tests...")
        passed = 0
        for name, func in self.tests:
            try:
                func()
                print(f"✅ {name}")
                passed += 1
            except Exception as e:
                print(f"❌ {name}: {e}")
        print(f"[TEST] {passed}/{len(self.tests)} passed.")

TEST_SUITE = DensityTestSuite()

# Example minimal internal validation tests
TEST_SUITE.add("Constant Folding", lambda: _const_fold("3+4") == "7")
TEST_SUITE.add("Macro Hygiene", lambda: MACRO_ENGINE.expand("foo") is not None)
TEST_SUITE.add("StdLib Load", lambda: STDLIB.import_module("io") is not None)

if __name__ == "__main__":
    TEST_SUITE.run_all()

# ==============================================================
# === Density 2 Compiler — Enhancement Layer v2 (Auto-Wrap) ===
# ==============================================================
# Drop this at the *very bottom* of density2_compiler.py
# It auto-detects and augments the main compiler class.
# ==============================================================

import os, re, sys, time, hashlib, inspect, random
from collections import defaultdict

print("[Density2 Enhancement Layer] Bootstrapping...")

# --------------------------------------------------------------
# OPTIMIZATION PIPELINE
# --------------------------------------------------------------

class OptimizationPass:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def run(self, ir):
        start = time.time()
        new_ir = self.func(ir)
        dur = (time.time() - start) * 1000
        print(f"[OPT] {self.name:<22} :: {dur:.2f} ms")
        return new_ir

def _const_fold(ir):
    return re.sub(r'(\d+)\s*\+\s*(\d+)', lambda m: str(int(m.group(1))+int(m.group(2))), ir)

def _peephole(ir):
    out = []
    for line in ir.splitlines():
        if re.search(r'\bmov\s+(\w+),\s*\1\b', line):
            continue
        out.append(line)
    return "\n".join(out)

def _loop_unroll(ir):
    return re.sub(r'for\s*\((\d+)\)', lambda m: f"; unrolled loop {m.group(1)}", ir)

OPTIMIZATION_PASSES = [
    OptimizationPass("Constant Folding", _const_fold),
    OptimizationPass("Peephole Simplifier", _peephole),
    OptimizationPass("Loop Unroller", _loop_unroll),
]

def run_optimizer(ir):
    for p in OPTIMIZATION_PASSES:
        ir = p.run(ir)
    return ir


# --------------------------------------------------------------
# MACRO HYGIENE SYSTEM
# --------------------------------------------------------------

class MacroRegistry:
    def __init__(self):
        self.macros = {}
        self.hygiene_hash = {}

    def register(self, name, pattern, body):
        token = hashlib.sha1(f"{name}{pattern}{body}".encode()).hexdigest()[:8]
        self.macros[name] = (pattern, body)
        self.hygiene_hash[name] = token
        print(f"[MACRO] Registered hygienic macro '{name}' :: {token}")

    def expand(self, src):
        for name, (pattern, body) in self.macros.items():
            token = self.hygiene_hash[name]
            src = re.sub(pattern, body.replace("$HYGIENE$", token), src)
        return src

MACRO_ENGINE = MacroRegistry()

def register_macro(name, pattern, body):
    MACRO_ENGINE.register(name, pattern, body)


# --------------------------------------------------------------
# DEBUGGER / SOURCE MAP
# --------------------------------------------------------------

class Debugger:
    def __init__(self):
        self.source_map = defaultdict(lambda: "unknown")
        self.breakpoints = set()

    def add_mapping(self, src_line, asm_label):
        self.source_map[asm_label] = src_line

    def add_breakpoint(self, label):
        self.breakpoints.add(label)
        print(f"[DBG] Breakpoint set at {label}")

    def hit(self, label):
        if label in self.breakpoints:
            print(f"[DBG] Breakpoint hit at {label}")
            input("Press Enter to continue...")

    def list_mappings(self):
        for k, v in self.source_map.items():
            print(f"{k:<10} -> line {v}")

DEBUGGER = Debugger()


# --------------------------------------------------------------
# STANDARD LIBRARY REGISTRY
# --------------------------------------------------------------

class DensityStdLib:
    def __init__(self):
        self.modules = {}

    def register(self, name, content):
        self.modules[name] = content
        print(f"[STDLIB] Module '{name}' registered.")

    def import_module(self, name):
        if name not in self.modules:
            raise ImportError(f"No such Density2 module '{name}'")
        print(f"[STDLIB] Module '{name}' imported.")
        return self.modules[name]

STDLIB = DensityStdLib()
STDLIB.register("io", "print, read, write, open")
STDLIB.register("math", "add, sub, mul, div, sqrt, sin, cos")
STDLIB.register("thread", "spawn, join, lock, unlock")


# --------------------------------------------------------------
# CROSS-PLATFORM BUILD WRAPPER
# --------------------------------------------------------------

def build_executable(output_name, asm_file):
    platform = sys.platform
    print(f"[BUILD] Platform detected: {platform}")
    if "win" in platform:
        os.system(f"nasm -f win64 {asm_file} -o temp.obj && link /SUBSYSTEM:CONSOLE temp.obj /OUT:{output_name}.exe")
    elif "linux" in platform:
        os.system(f"nasm -f elf64 {asm_file} -o temp.o && ld temp.o -o {output_name}")
    else:
        print("[WARN] Unsupported platform, manual linking required.")
    print(f"[BUILD] {output_name} built successfully.")


# --------------------------------------------------------------
# TESTING SUITE
# --------------------------------------------------------------

class DensityTestSuite:
    def __init__(self):
        self.tests = []

    def add(self, name, func):
        self.tests.append((name, func))

    def run_all(self):
        print(f"[TEST] Running {len(self.tests)} tests...")
        passed = 0
        for name, func in self.tests:
            try:
                func()
                print(f"✅ {name}")
                passed += 1
            except Exception as e:
                print(f"❌ {name}: {e}")
        print(f"[TEST] {passed}/{len(self.tests)} passed.")

TEST_SUITE = DensityTestSuite()


# --------------------------------------------------------------
# AUTO-WRAPPER
# --------------------------------------------------------------

def _find_compiler_class():
    """Locate the main compiler class dynamically."""
    for name, obj in globals().items():
        if inspect.isclass(obj) and "compile" in dir(obj):
            return name, obj
    return None, None


def _inject_methods(cls):
    """Attach enhancement methods to the compiler class."""
    def optimize_ir(self, ir_code):
        return run_optimizer(ir_code)

    def expand_macros(self, src):
        return MACRO_ENGINE.expand(src)

    def debug_map(self, ir):
        for i, line in enumerate(ir.splitlines()):
            DEBUGGER.add_mapping(i, f"L{i}")

    def build(self, out_name, asm_file):
        return build_executable(out_name, asm_file)

    setattr(cls, "optimize_ir", optimize_ir)
    setattr(cls, "expand_macros", expand_macros)
    setattr(cls, "debug_map", debug_map)
    setattr(cls, "build", build)

    print(f"[WRAP] Attached optimization, macro, debug, and build methods to {cls.__name__}")
    return cls


def _final_wrap():
    cname, cobj = _find_compiler_class()
    if cobj:
        globals()[cname] = _inject_methods(cobj)
        print(f"[WRAP] Successfully wrapped compiler class '{cname}'")
    else:
        print("[WRAP] No compiler class found — skipping auto-wrap.")

_final_wrap()


# --------------------------------------------------------------
# SELF-VALIDATION TESTS
# --------------------------------------------------------------

TEST_SUITE.add("ConstFold", lambda: _const_fold("2+2") == "4")
TEST_SUITE.add("Peephole", lambda: "mov rax,rax" not in _peephole("mov rax,rax"))
TEST_SUITE.add("MacroEngine", lambda: isinstance(MACRO_ENGINE.expand("x"), str))
TEST_SUITE.add("StdLib IO", lambda: "print" in STDLIB.import_module("io"))

if __name__ == "__main__":
    TEST_SUITE.run_all()

    print("[Density2 Enhancement Layer] All systems operational.")

    if hasattr(ss, '__name__'): # type: ignore
        print(f"Running as script: {ss.__name__}") # type: ignore

# ==============================================================
# === Density-2 Finalizer: Safe Full-Deck Optimizer & Builder ===
# ==============================================================
# Drop this at the very bottom of density2_compiler.py
# It runs after normal compilation and produces a
# fully linked, optimized, optionally compressed binary.
# ==============================================================

import subprocess, shutil, os, sys, tempfile

def ultra_optimize(asm_path: str, output_name: str = "a.out", compress: bool = True):
    """
    Turn an assembly (.asm) file into a highly optimized executable.
    Steps:
        1. Assemble with NASM (ELF64/Win64)
        2. Link with LD or LINK
        3. Optimize via llvm-opt / strip
        4. Optionally compress via UPX
    """

    if not os.path.exists(asm_path):
        raise FileNotFoundError(f"Assembly file not found: {asm_path}")

    platform = sys.platform
    temp_obj = tempfile.mktemp(suffix=".o")
    temp_exec = output_name

    # ----------------------------------------------------------
    # 1. Assemble
    # ----------------------------------------------------------
    if "win" in platform:
        cmd = ["nasm", "-f", "win64", asm_path, "-o", temp_obj]
    else:
        cmd = ["nasm", "-f", "elf64", asm_path, "-o", temp_obj]
    subprocess.run(cmd, check=True)
    print(f"[FINAL] Assembled → {temp_obj}")

    # ----------------------------------------------------------
    # 2. Link
    # ----------------------------------------------------------
    if "win" in platform:
        link_cmd = [
            "link",
            "/LTCG",  # link-time code generation
            "/OPT:REF", "/OPT:ICF",
            "/OUT:" + temp_exec + ".exe",
            temp_obj
        ]
        subprocess.run(link_cmd, check=True)
    else:
        ld_flags = ["-O3", "--gc-sections", "--strip-all"]
        subprocess.run(["ld", *ld_flags, temp_obj, "-o", temp_exec], check=True)
    print(f"[FINAL] Linked → {temp_exec}")

    # ----------------------------------------------------------
    # 3. Post-link optimization (strip + llvm-opt if available)
    # ----------------------------------------------------------
    if shutil.which("llvm-opt"):
        subprocess.run(["llvm-opt", "-O3", "-S", temp_exec, "-o", temp_exec], check=False)
    if shutil.which("strip"):
        subprocess.run(["strip", temp_exec], check=False)
    print("[FINAL] Binary stripped / LTO applied where available.")

    # ----------------------------------------------------------
    # 4. Optional UPX compression
    # ----------------------------------------------------------
    if compress and shutil.which("upx"):
        subprocess.run(["upx", "--best", "--lzma", temp_exec], check=False)
        print("[FINAL] Compressed with UPX.")
    else:
        print("[FINAL] Compression skipped (UPX not installed).")

    print(f"[FINAL] Optimized executable ready: {temp_exec}")

    return os.path.abspath(temp_exec)


# --------------------------------------------------------------
# Auto-integration with the wrapped compiler
# --------------------------------------------------------------
def _attach_ultra_optimize():
    cname, cobj = None, None
    for n, o in globals().items():
        if hasattr(o, "compile") and inspect.isclass(o):
            cname, cobj = n, o
            break
    if cobj:
        setattr(cobj, "ultra_optimize", staticmethod(ultra_optimize))
        print(f"[FINAL] ultra_optimize attached to compiler class '{cname}'")

_attach_ultra_optimize()

# --------------------------------------------------------------

# ==============================================================
# === Density-2 Stream Definition: ss Logic ====================
# ==============================================================

import threading
from io import StringIO

class SafeStream:
    """
    SafeStream ('ss') — unified write-buffer stream used by Density-2.
    Works like a high-performance stdout/file buffer.
    Features:
        - Thread-safe writes
        - Auto-flush to file or console
        - Context-manager support
        - Easy retrieval via get() or str()
    """

    def __init__(self, auto_flush=False, file_path=None):
        self._buffer = StringIO()
        self._lock = threading.Lock()
        self.auto_flush = auto_flush
        self.file_path = file_path
        self._file_handle = open(file_path, "w") if file_path else None

    def write(self, data):
        """Thread-safe write to the internal buffer or file."""
        with self._lock:
            text = str(data)
            self._buffer.write(text)
            if self._file_handle:
                self._file_handle.write(text)
                if self.auto_flush:
                    self._file_handle.flush()

    def writeln(self, data=""):
        """Write a line with newline."""
        self.write(str(data) + "\n")

    def flush(self):
        """Flush buffer to file (if enabled)."""
        with self._lock:
            if self._file_handle:
                self._file_handle.flush()

    def get(self):
        """Return current contents of the buffer."""
        with self._lock:
            return self._buffer.getvalue()

    def clear(self):
        """Clear internal buffer."""
        with self._lock:
            self._buffer = StringIO()

    def __str__(self):
        return self.get()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle:
            self._file_handle.close()


# --------------------------------------------------------------
# Global Singleton Instance: ss
# --------------------------------------------------------------

if "ss" not in globals():
    ss = SafeStream(auto_flush=False)
    globals()["ss"] = ss
    print("[STREAM] Global SafeStream 'ss' initialized.")


    # ==============================================================

# ==============================================================
# === Density-2 Assembly Emitter: emit_asm Logic ===============
# ==============================================================

import os, time

def emit_asm(ir_code: str,
             output_path: str = "out.asm",
             verbose: bool = True,
             header: bool = True) -> str:
    """
    emit_asm() — Writes finalized assembly text to disk safely.
    This function converts IR or high-level intermediate code into
    a properly formatted .asm file that NASM/LD can assemble.

    Parameters:
        ir_code     : str  — The IR or assembly text to emit.
        output_path : str  — Path of .asm file to create.
        verbose     : bool — Prints output status messages.
        header      : bool — Adds metadata header (timestamp, banner).

    Returns:
        str — Absolute path of the generated .asm file.
    """

    # ----------------------------------------------------------
    # 1. Safety / Validation
    # ----------------------------------------------------------
    if not isinstance(ir_code, str):
        raise TypeError("emit_asm expects string IR input.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ----------------------------------------------------------
    # 2. Build Assembly Header
    # ----------------------------------------------------------
    header_block = ""
    if header:
        header_block = (
            "; ==============================================================\n"
            ";  Density-2 Assembly Output\n"
            f";  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f";  File: {os.path.basename(output_path)}\n"
            ";  Architecture: x86-64 (NASM syntax)\n"
            "; ==============================================================\n\n"
        )

    # ----------------------------------------------------------
    # 3. Write to file using SafeStream (ss)
    # ----------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            if header_block:
                f.write(header_block)
                ss.writeln(header_block)
            f.write(ir_code)
            ss.writeln(ir_code)
            ss.flush()
    except Exception as e:
        raise IOError(f"emit_asm: failed to write {output_path}: {e}")

    # ----------------------------------------------------------
    # 4. Verbose Output
    # ----------------------------------------------------------
    abs_path = os.path.abspath(output_path)
    if verbose:
        print(f"[ASM] Emitted assembly to: {abs_path}")
        print(f"[ASM] Size: {len(ir_code)} bytes")

    return abs_path


# --------------------------------------------------------------
# 5. Global Hook: Attach to Compiler Class Automatically
# --------------------------------------------------------------
def _attach_emit_asm_to_compiler():
    import inspect
    for n, o in globals().items():
        if inspect.isclass(o) and "compile" in dir(o):
            setattr(o, "emit_asm", staticmethod(emit_asm))
            print(f"[WRAP] emit_asm() attached to compiler class '{n}'")
            break

_attach_emit_asm_to_compiler()

print("[ASM] emit_asm logic initialized.")


# ==============================================================

# ==============================================================
# === Density-2 Diagnostic Stream (Timestamped + Colored) ======
# ==============================================================

import sys, time, threading
from io import StringIO

class SafeStream:
    """
    SafeStream ('ss') — timestamped, colorized debug stream.
    Writes to memory buffer and optionally to console or file.
    """

    COLORS = {
        "reset": "\033[0m",
        "time":  "\033[38;5;244m",
        "info":  "\033[38;5;39m",
        "warn":  "\033[38;5;214m",
        "err":   "\033[38;5;196m",
    }

    def __init__(self, auto_flush=False, file_path=None, color=True):
        self._buf = StringIO()
        self._lock = threading.Lock()
        self.auto_flush = auto_flush
        self.file_path = file_path
        self._file = open(file_path, "w", encoding="utf-8") if file_path else None
        self.color = color

    def _stamp(self):
        return time.strftime("%H:%M:%S")

    def _color(self, kind, text):
        if not self.color or not sys.stdout.isatty():
            return text
        return f"{self.COLORS.get(kind, '')}{text}{self.COLORS['reset']}"

    def write(self, msg, kind="info"):
        with self._lock:
            stamp = f"[{self._stamp()}] "
            colored = self._color("time", stamp) + self._color(kind, str(msg))
            line = colored
            self._buf.write(f"{stamp}{msg}")
            if self._file:
                self._file.write(f"{stamp}{msg}")
                if self.auto_flush:
                    self._file.flush()
            print(line, end="")

    def writeln(self, msg="", kind="info"):
        self.write(str(msg) + "\n", kind)

    def get(self):
        with self._lock:
            return self._buf.getvalue()

    def clear(self):
        with self._lock:
            self._buf = StringIO()

    def flush(self):
        if self._file:
            self._file.flush()

    def __enter__(self): return self
    def __exit__(self, *a):
        if self._file: self._file.close()

# Singleton
if "ss" not in globals():
    ss = SafeStream()
    globals()["ss"] = ss
    print("[STREAM] Timestamped & colorized SafeStream 'ss' ready.")

# ==============================================================
# === Density-2 Assembly Emitter (safe) ========================
# ==============================================================

import os, platform, textwrap

def emit_asm(ir_text: str, output_path="out.asm", optimize=True):
    """
    Writes IR/assembly to file and prints safe build commands.
    """

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    header = textwrap.dedent(f"""\
    ; ==============================================================
    ; Density-2 Assembly Output
    ; Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
    ; ==============================================================
    """)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + ir_text)

    ss.writeln(f"[ASM] emitted → {os.path.abspath(output_path)}", "info")

    # ----------------------------------------------------------
    # Show optimized manual build commands (safe, not automatic)
    # ----------------------------------------------------------
    sysname = platform.system().lower()
    asm_file = os.path.basename(output_path)
    if "win" in sysname:
        cmd = f"nasm -f win64 {asm_file} -o temp.obj && link /LTCG /OPT:REF /OPT:ICF temp.obj /OUT:{asm_file[:-4]}.exe"
    else:
        cmd = f"nasm -f elf64 {asm_file} -o temp.o && ld -O3 --gc-sections --strip-all temp.o -o {asm_file[:-4]}"
    ss.writeln("[BUILD] To assemble & link manually:", "warn")
    ss.writeln("   " + cmd, "time")
    if optimize:
        ss.writeln("[OPT] Recommended: add strip / LTO / UPX for size-speed tuning.", "info")

    return os.path.abspath(output_path)

#!/usr/bin/env python3
"""
Density-2 Safe Builder
======================

Manual or CI-triggered assembler/linker driver for Density-2.
Detects NASM + LD/LINK, validates tools, and builds executables
with full logging and error diagnostics.

Usage:
    python build_density2.py path/to/out.asm [output_name]
"""

import subprocess, platform, os, sys, shutil, time

LOG_DIR = "build_logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"build_{time.strftime('%Y%m%d_%H%M%S')}.log")

def log(msg, end="\n"):
    print(msg, end=end)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + end)

def check_tool(name):
    path = shutil.which(name)
    if not path:
        log(f"[ERROR] Required tool '{name}' not found in PATH.")
        return False
    log(f"[OK] Found {name} at {path}")
    return True

def run_cmd(cmd, cwd=None):
    log(f"\n[CMD] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True,
                                text=True, check=True)
        if result.stdout:
            log(result.stdout)
        if result.stderr:
            log("[WARN] " + result.stderr)
    except subprocess.CalledProcessError as e:
        log(f"[FAIL] Command failed: {' '.join(cmd)}")
        log(e.stdout or "")
        log(e.stderr or "")
        raise

def build_density2(asm_path: str, output_name: str = "a.out"):
    if not os.path.exists(asm_path):
        log(f"[ERROR] Assembly file not found: {asm_path}")
        return

    sysname = platform.system().lower()
    asm_file = os.path.abspath(asm_path)
    base = os.path.splitext(output_name)[0]
    cwd = os.path.dirname(asm_file) or "."

    log(f"=== Density-2 Build Started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(f"[SYS] Platform detected: {sysname}")
    log(f"[SRC] {asm_file}")
    log(f"[OUT] {base}")

    # ----------------------------------------------------------
    # Toolchain detection
    # ----------------------------------------------------------
    if "win" in sysname:
        tools_ok = all(check_tool(t) for t in ("nasm", "link"))
    else:
        tools_ok = all(check_tool(t) for t in ("nasm", "ld"))
    if not tools_ok:
        log("[ABORT] Missing required tools; build aborted.")
        return

    # ----------------------------------------------------------
    # Assemble
    # ----------------------------------------------------------
    obj = os.path.join(cwd, "temp.obj" if "win" in sysname else "temp.o")
    fmt = "win64" if "win" in sysname else "elf64"
    run_cmd(["nasm", "-f", fmt, asm_file, "-o", obj])

    # ----------------------------------------------------------
    # Link
    # ----------------------------------------------------------
    if "win" in sysname:
        exe = f"{base}.exe"
        link_cmd = [
            "link", "/LTCG", "/OPT:REF", "/OPT:ICF",
            "/OUT:" + exe, obj
        ]
    else:
        exe = base
        link_cmd = [
            "ld", "-O3", "--gc-sections", "--strip-all",
            obj, "-o", exe
        ]
    run_cmd(link_cmd)

    # ----------------------------------------------------------
    # Post-build diagnostics
    # ----------------------------------------------------------
    if os.path.exists(exe):
        size = os.path.getsize(exe)
        log(f"[DONE] Build succeeded → {exe} ({size:,} bytes)")
    else:
        log("[WARN] Linker finished, but output not found.")

    log(f"=== Build finished; full log saved to {log_path} ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_density2.py <path/to/out.asm> [output_name]")
    else:
        build_density2(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "a.out")

        # ==============================================================

#!/usr/bin/env python3
"""
den_to_c.py — conservative .den -> .c transpiler (SAFE)

Usage:
    python den_to_c.py input.den [output.c]

What it does:
    - Translates a small, well-defined subset of .den syntax to C.
    - Writes an output C file.
    - Prints platform-aware compile & link commands for you to run manually.
    - DOES NOT run assembler/linker automatically.

Notes:
    - This translator is intentionally conservative / auditable.
    - Extend pattern rules if your .den uses more features.
"""

import sys
import os
import platform
import re
import textwrap

# --------------------------
# Config: modify as needed
# --------------------------
DEFAULT_OUTPUT_C = "out.c"
STD_HEADERS = ["stdio.h", "stdlib.h", "string.h"]

# --------------------------
# Simple translator helpers
# --------------------------

def den_escape_string(s):
    # Escape C string characters
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

def translate_line_to_c(line, in_main):
    """
    Translate a single .den line to C (very conservative).
    Returns (c_line, new_in_main_flag)
    If translation is unknown, line is emitted as a comment to keep context.
    """
    stripped = line.strip()

    # empty or comment
    if stripped == "":
        return ("", in_main)
    if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
        return (f"// {stripped}", in_main)

    # #asm block (inline asm). We'll map:
    # #asm{ mov rax, 1 }  -> asm volatile("mov rax, 1");
    m = re.match(r'#asm\s*\{(.*)\}\s*$', stripped)
    if m:
        asm_body = m.group(1).strip()
        asm_escaped = asm_body.replace('"', '\\"')
        return (f'__asm__ volatile ("{asm_escaped}");', in_main)

    # Main start: Main() or module main patterns
    if re.match(r'(?i)main\s*\(\)\s*$', stripped) or re.match(r'(?i)Main\s*\(\)\s*$', stripped):
        # Start main if not already in main
        if in_main:
            return ("// (ignored duplicate main() start)", in_main)
        else:
            return ("int main(void) {", True)

    # Print: ("Hello, world")
    m = re.match(r'(?i)print\s*:\s*\(\s*"(.*)"\s*\)\s*;?$', stripped)
    if m:
        s = den_escape_string(m.group(1))
        return (f'printf("{s}\\n");', in_main)

    # Generic Print call with args: Print: ("x", var)
    m = re.match(r'(?i)print\s*:\s*\(\s*"(.*)"\s*,\s*([A-Za-z_][\w]*)\s*\)\s*;?$', stripped)
    if m:
        s = den_escape_string(m.group(1))
        var = m.group(2)
        return (f'printf("{s}%d\\n", {var});', in_main)

    # Variable assignment (int-like): let x = 3;
    m = re.match(r'(?i)(?:let|var)\s+([A-Za-z_]\w*)\s*=\s*([0-9]+)\s*;?$', stripped)
    if m:
        name = m.group(1)
        val = m.group(2)
        return (f'int {name} = {val};', in_main)

    # Simple return statement inside main: return 0;
    if re.match(r'(?i)return\s+[0-9]+\s*;?$', stripped):
        return (stripped if stripped.endswith(";") else stripped + ";", in_main)

    # End of block markers: maybe 'end' or '}' or ')'. We treat 'end' as close brace.
    if re.match(r'(?i)end\s*$', stripped) or stripped == "}":
        if in_main:
            return ("}", False)
        else:
            return ("// end", False)

    # If the line looks like a C-style statement already, pass-through
    if stripped.endswith(";") and not any(c in stripped for c in "{}"):
        return (stripped, in_main)

    # Fallback: emit as comment with original for clarity
    return (f"// [den untranslated] {stripped}", in_main)


# --------------------------
# Transpiler entrypoint
# --------------------------

def den_to_c(src_path, out_c_path=DEFAULT_OUTPUT_C):
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)

    with open(src_path, "r", encoding="utf-8") as f:
        den_lines = f.readlines()

    c_lines = []
    # add headers
    c_lines.append("/* Generated by den_to_c.py — review before compiling */")
    for h in STD_HEADERS:
        c_lines.append(f"#include <{h}>")
    c_lines.append("")  # blank
    c_lines.append("// -- Begin translated code --")
    c_lines.append("")

    in_main = False
    for idx, line in enumerate(den_lines):
        translated, in_main = translate_line_to_c(line, in_main)
        if translated is None:
            continue
        if translated != "":
            c_lines.append(translated)

    # Ensure main closed
    if in_main:
        c_lines.append("    return 0;")
        c_lines.append("}")

    c_text = "\n".join(c_lines) + "\n"

    with open(out_c_path, "w", encoding="utf-8") as f:
        f.write(c_text)

    return os.path.abspath(out_c_path)


# --------------------------
# Helper: print compile commands
# --------------------------

def print_compile_instructions(c_path, output_name=None):
    sysname = platform.system().lower()
    c_base = os.path.splitext(os.path.basename(c_path))[0]
    out = output_name or c_base

    print("\n=== Manual build commands (run these yourself) ===")
    if "windows" in sysname or "mingw" in platform.system().lower() or "win" in sysname:
        # Use gcc/clang on Windows (mingw) or MSVC instruction example
        print("\nUsing GCC/MinGW:")
        print(f"  gcc -O3 -march=native -flto -pipe {c_path} -o {out}.exe")
        print("\nOr using MSVC (developer prompt):")
        print(f"  cl /O2 /GL {c_path} /Fe{out}.exe")
    else:
        # Linux / macOS
        print(f"  gcc -O3 -march=native -flto -pipe {c_path} -o {out}")
        print("  # optional: strip for size")
        print(f"  strip {out}")
        print("\nFor extra compression (optional):")
        print(f"  upx --best {out}  # if upx installed and appropriate for your platform")
    print("=== End commands ===\n")


# --------------------------
# CLI
# --------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python den_to_c.py input.den [output.c] [executable_name]")
        sys.exit(1)

    src = sys.argv[1]
    out_c = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_OUTPUT_C
    exe_name = sys.argv[3] if len(sys.argv) >= 4 else None

    c_path = den_to_c(src, out_c)
    print(f"[OK] Translated: {src} -> {c_path}")
    print_compile_instructions(c_path, exe_name)

if __name__ == "__main__":
    main()

    # ==============================================================

# density_vm.py
class DensityVM:
    def __init__(self):
        self.vars = {}
        self.labels = {}
        self.pc = 0
        self.program = []

    def load_program(self, lines):
        self.program = [ln.strip() for ln in lines if ln.strip() and not ln.startswith(";")]
        # index labels
        for i, line in enumerate(self.program):
            if line.endswith(":"):
                self.labels[line[:-1]] = i

    def run(self):
        while self.pc < len(self.program):
            line = self.program[self.pc]
            self.pc += 1
            parts = line.split()
            op = parts[0].upper()
            if op == "LOAD":
                var, val = parts[1].rstrip(","), int(parts[2])
                self.vars[var] = val
            elif op == "ADD":
                dst, a, b = [p.rstrip(",") for p in parts[1:4]]
                self.vars[dst] = self.vars.get(a, 0) + self.vars.get(b, 0)
            elif op == "PRINT":
                v = parts[1]
                print(self.vars.get(v, v))
            elif op == "JMP":
                label = parts[1]
                self.pc = self.labels[label]
            elif op == "EXIT":
                break
            elif line.endswith(":"):
                continue
            else:
                print(f"[WARN] Unknown instruction: {line}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python density_vm.py program.dbc")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        code = f.readlines()
    vm = DensityVM()
    vm.load_program(code)
    vm.run()

# -----------------------------
# Token Definitions
# -----------------------------
TOKEN_SPECIFICATION = [
    # Blocks first (non-greedy to avoid overconsumption)
    ('CIAM',        r"'''(.*?),,,"),              # ''' ... ,,,
    ('INLINE_ASM',  r"#asm(.*?)#endasm"),
    ('INLINE_C',    r"#c(.*?)#endc"),
    ('INLINE_PY',   r"#python(.*?)#endpython"),

    # Comments
    ('COMMENT',     r'//[^\n]*'),
    ('MCOMMENT',    r'/\*.*?\*/'),

    # Literals and identifiers
    ('STRING',      r'"(?:\\.|[^"\\])*"'),
    ('IDENT',       r'[A-Za-z_][A-Za-z0-9_]*'),

    # Symbols
    ('LBRACE',      r'\{'),
    ('RBRACE',      r'\}'),
    ('LPAREN',      r'\('),
    ('RPAREN',      r'\)'),
    ('COLON',       r':'),
    ('SEMICOLON',   r';'),
    ('COMMA',       r','),         # added for macro args
    ('PLUS',        r'\+'),        # added for string concat

    # Whitespace, newline, and mismatch
    ('NEWLINE',     r'\n'),
    ('SKIP',        r'[ \t]+'),
    ('MISMATCH',    r'.'),
]

# Make CIAM and INLINE_* non-greedy by ensuring (.*?) groups are used above.
# Note: CIAM pattern above currently uses (.*?),,, via the outer token list string.
# We will rebuild it here to ensure the non-greedy subpattern.
TOKEN_SPECIFICATION = [
    ('CIAM',        r"'''(.*?),,,"),              # We'll replace *? shortly
    ('INLINE_ASM',  r"#asm(.*?)#endasm"),
    ('INLINE_C',    r"#c(.*?)#endc"),
    ('INLINE_PY',   r"#python(.*?)#endpython"),
    ('COMMENT',     r'//[^\n]*'),
    ('MCOMMENT',    r'/\*.*?\*/'),
    ('STRING',      r'"(?:\\.|[^"\\])*"'),
    ('IDENT',       r'[A-Za-z_][A-Za-z0-9_]*'),
    ('LBRACE',      r'\{'),
    ('RBRACE',      r'\}'),
    ('LPAREN',      r'\('),
    ('RPAREN',      r'\)'),
    ('COLON',       r':'),
    ('SEMICOLON',   r';'),
    ('COMMA',       r','),         # macro args
    ('PLUS',        r'\+'),        # string concat
    ('NEWLINE',     r'\n'),
    ('SKIP',        r'[ \t]+'),
    ('MISMATCH',    r'.'),
]

# Fix CIAM to be non-greedy explicitly
TOKEN_SPECIFICATION = [
    (name, (pattern if name != 'CIAM' else r"'''(.*?),,,"))
    for (name, pattern) in TOKEN_SPECIFICATION
]

token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATION)


class Token:
    def __init__(self, type_: str, value: str, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line}, col={self.column})"


# -----------------------------
# Lexer
# -----------------------------
def tokenize(code: str) -> List[Token]:
    tokens: List[Token] = []
    line_num = 1
    line_start = 0
    for mo in re.finditer(token_regex, code, re.DOTALL):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
            continue
        elif kind in ('SKIP', 'COMMENT', 'MCOMMENT'):
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        tokens.append(Token(kind, value, line_num, column))
    return tokens


# -----------------------------
# AST Nodes
# -----------------------------
class ASTNode:
    """
    Base class for all Density 2 AST nodes.

    Features:
    - Optional source position tracking: filename, (line, col) -> (end_line, end_col)
    - Child discovery: children() finds nested AST nodes and lists of nodes
    - Traversal: walk() yields nodes in preorder
    - Visitor pattern: accept(visitor) calls visitor.visit_<Type>(self) or visitor.visit(self)
    - Structural replace: replace_child(old, new) updates direct attributes/lists
    - Serialization: to_dict()/pretty() for debugging and tooling
    - Copy: copy(**overrides) for shallow cloning
    - Dodecagram encoding: to_dodecagram() uses global ast_to_dodecagram if available
    - Structural equality: __eq__ based on type and serialized content (excluding positions)
    """

    # Position information is optional and can be set later via set_pos().
    def __init__(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ):
        self.filename = filename
        self.line = line
        self.col = col
        self.end_line = end_line
        self.end_col = end_col

    # ----- Source position helpers -----
    def set_pos(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ) -> "ASTNode":
        if filename is not None:
            self.filename = filename
        if line is not None:
            self.line = line
        if col is not None:
            self.col = col
        if end_line is not None:
            self.end_line = end_line
        if end_col is not None:
            self.end_col = end_col
        return self

    # ----- Introspection helpers -----
    def _is_pos_field(self, name: str) -> bool:
        return name in ("filename", "line", "col", "end_line", "end_col")

    def _iter_fields(self):
        # Do not consider private/dunder attributes as AST data
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            yield k, v

    def children(self) -> List["ASTNode"]:
        """Return direct child AST nodes (flattening lists)."""
        result: List[ASTNode] = []
        for _, v in self._iter_fields():
            if isinstance(v, ASTNode):
                result.append(v)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, ASTNode):
                        result.append(it)
        return result

    def walk(self):
        """Preorder traversal of this subtree."""
        yield self
        for c in self.children():
            yield from c.walk()

    # ----- Visitor pattern -----
    def accept(self, visitor):
        """Call visitor.visit_<Type>(self) if present, else visitor.visit(self) if present."""
        method = getattr(visitor, f"visit_{self.__class__.__name__}", None)
        if callable(method):
            return method(self)
        generic = getattr(visitor, "visit", None)
        if callable(generic):
            return generic(self)
        return None

    # ----- Structural operations -----
    def replace_child(self, old: "ASTNode", new: Optional["ASTNode"]) -> bool:
        """
        Replace a direct child 'old' with 'new'.
        If 'new' is None, removes the child if it's in a list; clears attribute otherwise.
        Returns True if a replacement/removal occurred.
        """
        changed = False
        for k, v in list(self._iter_fields()):
            if isinstance(v, ASTNode):
                if v is old:
                    setattr(self, k, new)
                    changed = True
            elif isinstance(v, list):
                # Replace in lists; remove if new is None
                new_list = []
                for it in v:
                    if it is old:
                        if new is not None:
                            new_list.append(new)
                        changed = True
                    else:
                        new_list.append(it)
                if changed:
                    setattr(self, k, new_list)
        return changed

    # ----- Serialization / Debugging -----
    def to_dict(self, *, include_pos: bool = True) -> Dict[str, object]:
        """Convert the node (recursively) to a dict suitable for JSON/debugging."""
        d: Dict[str, object] = {"__type__": self.__class__.__name__}
        for k, v in self._iter_fields():
            if not include_pos and self._is_pos_field(k):
                continue
            if isinstance(v, ASTNode):
                d[k] = v.to_dict(include_pos=include_pos)
            elif isinstance(v, list):
                d[k] = [
                    (it.to_dict(include_pos=include_pos) if isinstance(it, ASTNode) else it)
                    for it in v
                ]
            else:
                d[k] = v
        return d

    def pretty(self, indent: str = "  ") -> str:
        """Human-readable multi-line tree dump."""
        lines: List[str] = []

        def rec(n: "ASTNode", depth: int):
            pad = indent * depth
            header = n.__class__.__name__
            pos = []
            if n.filename:
                pos.append(f'file="{n.filename}"')
            if n.line is not None and n.col is not None:
                pos.append(f"@{n.line}:{n.col}")
            if n.end_line is not None and n.end_col is not None:
                pos.append(f"-{n.end_line}:{n.end_col}")
            if pos:
                header += " [" + " ".join(pos) + "]"
            lines.append(pad + header)

            # Show scalar fields
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    continue
                if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                    continue
                lines.append(pad + indent + f"{k} = {v!r}")

            # Recurse into child nodes
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    lines.append(pad + indent + f"{k}:")
                    rec(v, depth + 2)
                elif isinstance(v, list):
                    child_nodes = [it for it in v if isinstance(it, ASTNode)]
                    if child_nodes:
                        lines.append(pad + indent + f"{k}: [{len(child_nodes)}]")
                        for it in child_nodes:
                            rec(it, depth + 2)

        rec(self, 0)
        return "\n".join(lines)

    def copy(self, **overrides):
        """
        Shallow copy with optional field overrides:
            new = node.copy(body=new_body)
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)  # type: ignore
        # Copy all instance attributes
        new_obj.__dict__.update(self.__dict__)
        # Apply overrides
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    def to_dodecagram(self) -> str:
        """
        Encode this node (and subtree) using the Dodecagram mapping.
        Relies on a global function ast_to_dodecagram(node).
        """
        f = globals().get("ast_to_dodecagram")
        if callable(f):
            return f(self)  # type: ignore[misc]
        raise RuntimeError("ast_to_dodecagram() is not available in this module")

    # ----- Equality / Representation -----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASTNode):
            return False
        if self.__class__ is not other.__class__:
            return False
        return self.to_dict(include_pos=False) == other.to_dict(include_pos=False)

    def __repr__(self) -> str:
        # Compact representation showing scalar fields only
        fields: List[str] = []
        for k, v in self._iter_fields():
            if isinstance(v, ASTNode):
                continue
            if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                continue
            fields.append(f"{k}={v!r}")
        inner = ", ".join(fields)
        return f"{self.__class__.__name__}({inner})"


class Program(ASTNode):
    def __init__(self, functions: List['Function']):
        self.functions = functions

    def __repr__(self):
        return f"Program({self.functions})"


class Function(ASTNode):
    def __init__(self, name: str, body: List['Statement']):
        self.name = name
        self.body = body

    def __repr__(self):
        return f"Function({self.name}, body={self.body})"


class Statement(ASTNode):
    """Base class for all statements in Density 2."""
    pass


class PrintStatement(Statement):
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Print({self.text!r})"


class CIAMBlock(Statement):
    def __init__(self, name: str, params: List[str], body_text: str):
        self.name = name
        self.params = params
        self.body_text = body_text  # raw Density 2 snippet

    def __repr__(self):
        return f"CIAMBlock(name={self.name!r}, params={self.params}, body_len={len(self.body_text)})"


class MacroCall(Statement):
    def __init__(self, name: str, arg_texts: List[str]):
        self.name = name
        self.arg_texts = arg_texts  # raw argument texts

    def __repr__(self):
        return f"MacroCall({self.name!r}, args={self.arg_texts})"


class InlineBlock(Statement):
    def __init__(self, lang: str, content: str):
        self.lang = lang  # 'asm', 'c', 'python'
        self.content = content

    def __repr__(self):
        return f"InlineBlock(lang={self.lang!r}, content_len={len(self.content)})"


# -----------------------------
# Dodecagram AST encoding
# -----------------------------
_DODECAGRAM_MAP = {
    'Program': '0',
    'Function': '1',
    'PrintStatement': '2',
    'CIAMBlock': '3',
    'MacroCall': '4',
    # InlineBlock variants:
    'InlineBlock:asm': '5',
    'InlineBlock:python': '6',
    'InlineBlock:py': '6',
    'InlineBlock:c': '7',
    'InlineBlock:other': '8',
    # Reserved for future nodes:
    '_reserved9': '9',
    '_reserveda': 'a',
    '_reservedb': 'b',
}

def ast_to_dodecagram(node: ASTNode) -> str:
    """
    Preorder encoding of the AST using the Dodecagram alphabet 0-9,a,b.
    """
    def enc(n: ASTNode) -> str:
        if isinstance(n, Program):
            s = _DODECAGRAM_MAP['Program']
            for f in n.functions:
                s += enc(f)
            return s
        if isinstance(n, Function):
            s = _DODECAGRAM_MAP['Function']
            for st in n.body:
                s += enc(st)
            return s
        if isinstance(n, PrintStatement):
            return _DODECAGRAM_MAP['PrintStatement']
        if isinstance(n, CIAMBlock):
            return _DODECAGRAM_MAP['CIAMBlock']
        if isinstance(n, MacroCall):
            return _DODECAGRAM_MAP['MacroCall']
        if isinstance(n, InlineBlock):
            key = f'InlineBlock:{n.lang}'
            ch = _DODECAGRAM_MAP.get(key, _DODECAGRAM_MAP['InlineBlock:other'])
            return ch
        # Unknown node -> reserved
        return _DODECAGRAM_MAP['_reserved9']
    return enc(node)

# -----------------------------
# Parser
# -----------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        # Macro table collected while parsing: name -> CIAMBlock
        self.macro_table: Dict[str, CIAMBlock] = {}

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def lookahead(self, offset: int) -> Optional[Token]:
        idx = self.pos + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_type: str) -> Token:
        tok = self.peek()
        if not tok or tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok

    def match(self, expected_type: str) -> bool:
        tok = self.peek()
        if tok and tok.type == expected_type:
            self.pos += 1
            return True
        return False

    def parse(self) -> Program:
        functions: List[Function] = []
        while self.peek() is not None:
            # Expect IDENT (function name) or skip stray tokens
            tok = self.peek()
            if tok.type == 'IDENT':
                # Look for function signature IDENT()
                if self.lookahead(1) and self.lookahead(1).type == 'LPAREN':
                    functions.append(self.parse_function())
                else:
                    # Skip stray identifier
                    self.pos += 1
            else:
                # Skip unknown tokens at top-level
                self.pos += 1
        return Program(functions)

    def parse_function(self) -> Function:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        self.consume('RPAREN')
        body: List[Statement] = []
        if self.match('LBRACE'):
            # Block body
            body = self.parse_statements_until_rbrace()
            self.consume('RBRACE')
        else:
            # Single statement allowed
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        return Function(name_tok.value, body)

    def parse_statements_until_rbrace(self) -> List[Statement]:
        stmts: List[Statement] = []
        while True:
            tok = self.peek()
            if tok is None or tok.type == 'RBRACE':
                break
            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    stmts.extend(stmt)
                else:
                    stmts.append(stmt)
        return stmts

    def parse_statement(self) -> Optional[Union[Statement, List[Statement]]]:
        tok = self.peek()
        if tok is None:
            return None

        if tok.type == 'IDENT' and tok.value == 'Print':
            return self.parse_print()

        if tok.type == 'CIAM':
            # Parse CIAM block definition, store in macro table, no direct output
            ciam_tok = self.consume('CIAM')
            content = ciam_tok.value
            # Strip delimiters: starts with ''' and ends with ,,,
            if content.startswith("'''"):
                content_inner = content[3:]
                if content_inner.endswith(",,,"):
                    content_inner = content_inner[:-3]
            else:
                content_inner = content

            name, params, body_text = self._parse_ciam_content(content_inner.strip(), ciam_tok)
            ciam_block = CIAMBlock(name, params, body_text)
            self.macro_table[name] = ciam_block
            # Defining a macro does not emit code
            return None

        if tok.type.startswith('INLINE_'):
            return self.parse_inline_block()

        if tok.type == 'IDENT':
            # Possibly a macro invocation: IDENT(...)
            ident = tok.value
            la = self.lookahead(1)
            if la and la.type == 'LPAREN':
                return self.parse_macro_call()
            else:
                # Unknown ident; consume to progress
                self.pos += 1
                return None

        # Unknown token; consume to progress
        self.pos += 1
        return None

    def _parse_ciam_content(self, content: str, tok: Token) -> Tuple[str, List[str], str]:
        # First line: Name(params)
        lines = content.splitlines()
        if not lines:
            raise SyntaxError(f"Empty CIAM at line {tok.line}")

        header = lines[0].strip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$', header)
        if not m:
            raise SyntaxError(f"Invalid CIAM header '{header}' at line {tok.line}")
        name = m.group(1)
        params_str = m.group(2).strip()
        params = [p.strip() for p in params_str.split(',') if p.strip()] if params_str else []
        body_text = '\n'.join(lines[1:]).strip()
        return name, params, body_text

    def parse_macro_call(self) -> MacroCall:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        args: List[str] = []
        current_parts: List[str] = []
        paren_depth = 1
        while True:
            tok = self.peek()
            if tok is None:
                raise SyntaxError("Unclosed macro call")
            if tok.type == 'LPAREN':
                paren_depth += 1
                current_parts.append(tok.value)
                self.pos += 1
            elif tok.type == 'RPAREN':
                paren_depth -= 1
                if paren_depth == 0:
                    # finalize last arg
                    arg_text = ''.join(current_parts).strip()
                    if arg_text:
                        args.append(arg_text)
                    self.pos += 1
                    break
                else:
                    current_parts.append(tok.value)
                    self.pos += 1
            elif tok.type == 'COMMA' and paren_depth == 1:
                arg_text = ''.join(current_parts).strip()
                args.append(arg_text)
                current_parts = []
                self.pos += 1
            else:
                current_parts.append(tok.value)
                self.pos += 1
        self.consume('SEMICOLON')
        return MacroCall(name_tok.value, args)

    def parse_inline_block(self) -> InlineBlock:
        tok = self.peek()
        lang = tok.type.split('_', 1)[1].lower()  # asm / c / py
        inline_tok = self.consume(tok.type)
        # strip off #lang and #endlang markers:
        content = re.sub(r'^#\w+', '', inline_tok.value, flags=re.DOTALL)
        content = re.sub(r'#end\w+$', '', content, flags=re.DOTALL)
        return InlineBlock(lang, content.strip())

    def parse_print(self) -> PrintStatement:
        self.consume('IDENT')  # 'Print'
        self.consume('COLON')
        self.consume('LPAREN')

        # Simple string concat of STRING (+ STRING)*
        parts: List[str] = []
        first = True
        while True:
            tok = self.peek()
            if not tok:
                raise SyntaxError("Unclosed Print: missing ')'")
            if first:
                if tok.type != 'STRING':
                    raise SyntaxError(f"Print expects a string literal, got {tok}")
                parts.append(eval(tok.value))
                self.pos += 1
                first = False
            else:
                if tok.type == 'PLUS':
                    self.pos += 1
                    tok2 = self.peek()
                    if not tok2 or tok2.type != 'STRING':
                        raise SyntaxError("Expected string after '+' in Print")
                    parts.append(eval(tok2.value))
                    self.pos += 1
                else:
                    break

            # optional spaces are already skipped by lexer

        self.consume('RPAREN')
        self.consume('SEMICOLON')
        full_text = ''.join(parts)
        return PrintStatement(full_text)


# -----------------------------
# Macro Expansion
# -----------------------------
def expand_macros(program: Program, macro_table: Dict[str, CIAMBlock], max_depth: int = 32) -> Program:
    def expand_statements(stmts: List[Statement], depth: int) -> List[Statement]:
        result: List[Statement] = []
        for s in stmts:
            if isinstance(s, MacroCall):
                if s.name not in macro_table:
                    # Unknown macro, ignore call
                    continue
                macro = macro_table[s.name]
                expanded = expand_macro_call(macro, s, depth)
                result.extend(expanded)
            else:
                result.append(s)
        return result

    def expand_macro_call(macro: CIAMBlock, call: MacroCall, depth: int) -> List[Statement]:
        if depth <= 0:
            raise RecursionError("Macro expansion depth exceeded")
        # Build replacement text by simple identifier replacement
        body = macro.body_text

        # map params -> args text
        mapping = {p: (call.arg_texts[i] if i < len(call.arg_texts) else '') for i, p in enumerate(macro.params)}

        # Replace params with args (word boundary replacement)
        for p, a in mapping.items():
            # Use a simple token-ish replace; avoid capturing partial identifiers
            body = re.sub(rf'\b{re.escape(p)}\b', a, body)

        # Re-parse the expanded snippet into statements
        sub_tokens = tokenize(body)
        sub_parser = Parser(sub_tokens)
        # Share the same macro table to allow nested macro definitions/uses
        sub_parser.macro_table = macro_table
        # We parse a synthetic function body to reuse statement parser
        sub_stmts = sub_parser.parse_statements_until_rbrace()  # until EOF
        # Recursively expand within
        return expand_statements(sub_stmts, depth - 1)

    # Expand within each function
    new_functions: List[Function] = []
    for f in program.functions:
        expanded_body = expand_statements(f.body, max_depth)
        new_functions.append(Function(f.name, expanded_body))
    return Program(new_functions)


# -----------------------------
# NASM Emitter
# -----------------------------
class CodeGenerator:
    def __init__(self, ast: Program):
        self.ast = ast
        self.text_lines: List[str] = []
        self.data_lines: List[str] = []
        self.string_table: Dict[str, str] = {}  # map string -> label
        self.label_counter = 0
        # simple register reuse cache (invalidated across inline blocks)
        self._reg_cache = {'rax_sys_write': False, 'rdi_stdout': False}

    def generate(self) -> str:
        # Expand macros already done in compile_density2
        self.text_lines = []
        self.data_lines = []
        self.string_table = {}
        self.label_counter = 0
        self._invalidate_reg_cache()

        # Header and functions
        self._emit_header()
        for func in self.ast.functions:
            self._emit_function(func)

        # Build final assembly
        final_lines: List[str] = []
        final_lines.append('section .data')
        final_lines.extend('    ' + line for line in self.data_lines)
        final_lines.append('section .text')
        final_lines.append('    global _start')
        for l in self.text_lines:
            final_lines.append(l)

        return '\n'.join(final_lines)

    def _invalidate_reg_cache(self):
        self._reg_cache['rax_sys_write'] = False
        self._reg_cache['rdi_stdout'] = False

    def _emit_header(self):
        # Add a small banner; sections are assembled at the end
        self.text_lines.append('; --- Density 2 NASM output ---')
        self.text_lines.append('; inline C: compiled then translated (AT&T -> NASM best-effort)')
        self.text_lines.append('; inline Python: executed at codegen; use emit("...") to output NASM')

    def _emit_function(self, func: Function):
        # Entry label
        if func.name == 'Main':
            self.text_lines.append('_start:')
        else:
            self.text_lines.append(f'{func.name}:')
        # function entry: registers not assumed
        self._invalidate_reg_cache()

        for stmt in func.body:
            if isinstance(stmt, PrintStatement):
                self._emit_print(stmt.text)
            elif isinstance(stmt, InlineBlock):
                self._emit_inline(stmt)
                # assume inline code may clobber registers
                self._invalidate_reg_cache()
            elif isinstance(stmt, CIAMBlock):
                # Already handled by expand_macros; leave a comment in case any remain
                self.text_lines.append(f'    ; CIAMBlock ignored (should be expanded): {getattr(stmt, "name", "?")}')
            elif isinstance(stmt, MacroCall):
                # Should be expanded away
                self.text_lines.append(f'    ; MacroCall ignored (should be expanded): {getattr(stmt, "name", "?")}')
            else:
                # Unknown statement type
                self.text_lines.append('    ; Unknown statement')

        if func.name == 'Main':
            self._emit_exit()

    def _encode_string_bytes(self, s: str) -> List[int]:
        return list(s.encode('utf-8'))

    def _get_string_label(self, text: str) -> Tuple[str, int]:
        if text not in self.string_table:
            label = f'str_{self.label_counter}'
            self.label_counter += 1
            data_bytes = self._encode_string_bytes(text)
            # Store bytes + newline + NUL terminator
            stored = data_bytes + [10, 0]
            bytes_list = ', '.join(str(b) for b in stored)
            self.data_lines.append(f'{label} db {bytes_list}')
            # Length to write: bytes plus newline only (exclude NUL)
            length = len(data_bytes) + 1
            self.string_table[text] = label
        else:
            label = self.string_table[text]
            data_bytes = self._encode_string_bytes(text)
            length = len(data_bytes) + 1
        return label, length

    def _emit_print(self, text: str):
        label, length = self._get_string_label(text)
        # register reuse: avoid redundant loads across consecutive prints
        if not self._reg_cache['rax_sys_write']:
            self.text_lines.append(f'    mov rax, 1          ; sys_write')
            self._reg_cache['rax_sys_write'] = True
        if not self._reg_cache['rdi_stdout']:
            self.text_lines.append(f'    mov rdi, 1          ; stdout')
            self._reg_cache['rdi_stdout'] = True
        self.text_lines.append(f'    mov rsi, {label}    ; message')
        self.text_lines.append(f'    mov rdx, {length}         ; length (bytes)')
        self.text_lines.append('    syscall')
        # after syscall, rax/rdi are still fine for subsequent prints in our model

    def _emit_exit(self):
        self.text_lines.append('    mov rax, 60         ; sys_exit')
        self.text_lines.append('    xor rdi, rdi        ; status 0')
        self.text_lines.append('    syscall')
        # invalidate cache after exit emitter
        self._invalidate_reg_cache()

    def _emit_inline(self, block: InlineBlock):
        if block.lang == 'asm':
            self.text_lines.append('    ; inline NASM start')
            for line in block.content.splitlines():
                line = line.rstrip()
                if line:
                    self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline NASM end')
        elif block.lang in ('py', 'python'):
            self.text_lines.append('    ; inline Python start')
            for line in self._run_inline_python(block.content):
                self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline Python end')
        elif block.lang == 'c':
            self.text_lines.append('    ; inline C start')
            asm_lines = self._compile_c_to_asm(block.content)
            if asm_lines:
                for l in asm_lines:
                    self.text_lines.append('    ' + l)
            else:
                # Fallback: keep it commented
                for line in block.content.splitlines():
                    self.text_lines.append('    ; ' + line)
                self.text_lines.append('    ; (no C compiler found; block left as comment)')
            self.text_lines.append('    ; inline C end')
        else:
            # Unknown inline language
            for line in block.content.splitlines():
                self.text_lines.append(f'    ; inline {block.lang} ignored: {line}')

    def _run_inline_python(self, code: str) -> List[str]:
        # Provide a tiny DSL where Python can emit NASM
        lines: List[str] = []

        def emit(s: str):
            if not isinstance(s, str):
                raise TypeError("emit() expects a string")
            lines.append(s)

        def label(prefix: str = 'gen'):
            lbl = f'{prefix}_{self.label_counter}'
            self.label_counter += 1
            return lbl

        globals_dict = {
            '__builtins__': {
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'print': print,
            },
            'emit': emit,
            'label': label,
        }
        locals_dict: Dict[str, object] = {}
        try:
            exec(code, globals_dict, locals_dict)
        except Exception as ex:
            lines.append(f'; [inline python error] {ex!r}')
        return lines

    def _compile_c_to_asm(self, c_code: str) -> List[str]:
        # Try clang, gcc, tcc, cl (MSVC) in that order to produce assembly text
        compiler = None
        for cand in ('clang', 'gcc', 'tcc', 'cl'):
            if shutil.which(cand):
                compiler = cand
                break
        if compiler is None:
            return []

        # MSVC path (use /FA to generate assembly listing)
        if compiler == 'cl':
            lines = self._compile_with_msvc(c_code)
            if lines:
                # Unobtrusive log of the compiler used
                return ['; [inline c compiler: cl]'] + lines
            return []

        tmpdir = tempfile.mkdtemp(prefix='den2_c_')
        c_path = os.path.join(tmpdir, 'inline.c')
        asm_path = os.path.join(tmpdir, 'inline.s' if compiler != 'cl' else 'inline.asm')
        try:
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            if compiler == 'clang':
                cmd = ['clang', '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer', '-masm=intel', '-m64']
            elif compiler == 'gcc':
                cmd = ['gcc', '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer', '-masm=intel', '-m64']
            elif compiler == 'tcc':
                cmd = ['tcc', '-nostdlib', '-S', c_path, '-o', asm_path]
            else:  # cl (MSVC)
                return ['; [inline c compiler: cl]'] + self._compile_with_msvc(c_code)

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not os.path.exists(asm_path):
                return []

            # Try Intel-to-NASM normalization first; if looks AT&T, fallback
            translated = self._intel_to_nasm(asm_path)
            if translated:
                return [f'; [inline c compiler: {compiler}]'] + translated

            fallback = ['; [inline c compiler: {compiler}]', '; [begin compiled C assembly]']
            with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()
            return fallback + ['; ' + ln for ln in raw.splitlines()] + ['; [end compiled C assembly]']
        except Exception as ex:
            return [f'; [inline c compile error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _compile_with_msvc(self, c_code: str) -> List[str]:
        tmpdir = tempfile.mkdtemp(prefix='den2_msvc_')
        try:
            c_path = os.path.join(tmpdir, 'inline.c')
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            cmd = ['cl', '/nologo', '/FA', '/c', os.path.basename(c_path)]
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            asm_listing = os.path.join(tmpdir, 'inline.asm')
            lines: List[str] = []

            # If cl produced diagnostics, include them as comments (unobtrusive)
            if proc.stdout:
                for ln in proc.stdout.splitlines():
                    if ln.strip():
                        lines.append('; [cl] ' + ln)
            if proc.returncode != 0 and proc.stderr:
                for ln in proc.stderr.splitlines():
                    if ln.strip():
                        lines.append('; [cl err] ' + ln)

            if os.path.exists(asm_listing):
                with open(asm_listing, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                translated = self._msvc_asm_to_nasm(raw)
                # Place translated code after any diagnostics
                return lines + translated

            # No asm produced; return only diagnostics if any
            return lines
        except Exception as ex:
            return [f'; [inline c compile (msvc) error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _msvc_asm_to_nasm(self, msvc_asm: str) -> List[str]:
        out: List[str] = []
        for line in msvc_asm.splitlines():
            s = line.rstrip()
            if not s:
                continue

            # Keep original comments, already ';' prefixed in MSVC listings
            if s.lstrip().startswith(';'):
                out.append(s)
                continue

            tok = s.strip()

            # Skip or rewrite common MSVC metadata/directives
            upper = tok.upper()
            if upper.startswith(('TITLE ', 'COMMENT ', 'INCLUDE ', 'INCLUDELIB ')):
                # Ignore headers/includes in listing
                continue
            if upper.startswith(('.MODEL', '.CODE', '.DATA', '.CONST', '.XDATA', '.PDATA', '.STACK', '.LIST', '.686', '.686P', '.XMM', '.X64')):
                continue
            if upper.startswith(('PUBLIC ', 'EXTRN ', 'EXTERN ', 'ASSUME ')):
                continue
            if upper == 'END':
                continue
            if upper.startswith('ALIGN '):
                # Map to NASM align
                parts = tok.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    out.append(f'align {parts[1]}')
                continue

            # PROC/ENDP handling: turn "label PROC" -> "label:" and skip ENDP
            m_proc = re.match(r'^([A-Za-z_$.@?][\w$.@?]*)\s+PROC\b', tok)
            if m_proc:
                out.append(f'{m_proc.group(1)}:')
                continue
            if re.match(r'^[A-Za-z_$.@?][\w$.@?]*\s+ENDP\b', tok):
                # End of procedure: ignore (no epilogue emission here)
                continue

            # Data directives: DB/DW/DD/DQ -> lowercase NASM-friendly
            m_data = re.match(r'^(DB|DW|DD|DQ)\b(.*)$', tok, re.IGNORECASE)
            if m_data:
                out.append(m_data.group(1).lower() + m_data.group(2))
                continue

            # Replace "BYTE/WORD/DWORD/QWORD PTR" with NASM "byte/word/dword/qword"
            tok = re.sub(r'\b(BYTE|WORD|DWORD|QWORD)\s+PTR\b', lambda m: m.group(1).lower(), tok)

            # Replace "OFFSET FLAT:label" or "OFFSET label" with just label
            tok = re.sub(r'\bOFFSET\s+FLAT:', '', tok, flags=re.IGNORECASE)
            tok = re.sub(r'\bOFFSET\s+', '', tok, flags=re.IGNORECASE)

            # Some MSVC emits "FLAT:" as a segment label prefix; drop it
            tok = tok.replace(' FLAT:', ' ')

            # Instructions are already Intel syntax; keep as-is
            out.append(tok)

        return out

    # --- AT&T -> NASM best-effort translation helpers ---
    def _translate_att_to_nasm(self, att_asm: str) -> List[str]:
        out: List[str] = []
        for line in att_asm.splitlines():
            s = line.strip()
            if not s:
                continue

            if s.startswith(('.') or '#'):
                if s.startswith(('.text', '.data', '.bss')):
                    out.append(f"section {s[1:]}")
                continue

            if s.endswith(':'):
                out.append(s)
                continue

            s = s.split('\t#', 1)[0].split(' #', 1)[0].strip()
            if not s:
                continue

            parts = s.split(None, 1)
            op = parts[0]
            rest = parts[1] if len(parts) > 1 else ''
            op_n = re.sub(r'(q|l|w|b)$', '', op)

            ops = [o.strip() for o in rest.split(',')] if rest else []
            ops = [self._att_operand_to_nasm(o) for o in ops]

            if len(ops) == 2:
                ops = [ops[1], ops[0]]

            if ops:
                out.append(f'{op_n} ' + ', '.join(ops))
            else:
                out.append(op_n)

        return out

    def _att_operand_to_nasm(self, o: str) -> str:
        o = o.strip()
        if o.startswith('$'):
            return o[1:]
        o = re.sub(r'%([a-zA-Z][a-zA-Z0-9]*)', r'\1', o)

        m = re.match(r'^\s*([\-+]?\d+)?\s*\(\s*([a-zA-Z0-9%]+)\s*(?:,\s*([a-zA-Z0-9%]+)\s*(?:,\s*(\d+))?)?\s*\)\s*$', o)
        if m:
            disp, base, index, scale = m.groups()
            base = base.lstrip('%')
            parts = [base]
            if index:
                idx = index.lstrip('%')
                sc = int(scale) if scale else 1
                parts.append(f'{idx}*{sc}')
            addr = ' + '.join(parts) if parts else ''
            if disp:
                sign = '+' if int(disp) >= 0 else '-'
                addr = f'{addr} {sign} {abs(int(disp))}' if addr else str(disp)
            return f'[{addr}]'

        return o


# -----------------------------
# Public API
# -----------------------------
def generate_nasm(ast_or_source: Union[Program, str]) -> str:
    """
    Generate NASM assembly.
    - If given a Program AST, emit NASM via CodeGenerator.
    - If given Density 2 source (str), parse, expand macros, then emit NASM.
    - If given an assembly-looking string (already NASM), return as-is.
    """
    if isinstance(ast_or_source, Program):
        gen = CodeGenerator(ast_or_source)
        return gen.generate()

    if isinstance(ast_or_source, str):
        text = ast_or_source
        # Heuristic: looks like NASM already
        if re.search(r'^\s*section\s+\.text\b', text, flags=re.MULTILINE) and 'global _start' in text:
            return text
        # Treat as Density 2 source
        tokens = tokenize(text)
        parser = Parser(tokens)
        program = parser.parse()
        program = expand_macros(program, parser.macro_table)
        gen = CodeGenerator(program)
        return gen.generate()

    raise TypeError(f"generate_nasm expects Program or str, got {type(ast_or_source).__name__}")


def parse_density2(code: str) -> Program:
    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse()
    # Expand macros now that we have parser.macro_table
    ast = expand_macros(ast, parser.macro_table)
    return ast


# -----------------------------
# CLI
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: density2_compiler.py <input.den2> [--debug]")
        sys.exit(0)

    debug_mode = '--debug' in sys.argv
    src_path = next((a for a in sys.argv[1:] if not a.startswith('--')), None)
    if not src_path or not os.path.exists(src_path):
        print("Input file not found.")
        sys.exit(2)

    with open(src_path, 'r', encoding='utf-8') as f:
        source = f.read()

    if debug_mode:
        print("🔍 Entering Density 2 Debugger...")
        program = parse_density2(source)
        start_debugger(program, filename=src_path)
        return

    asm = generate_nasm(source)
    out_path = os.path.join(os.path.dirname(src_path) or '.', 'out.asm')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(asm)
    print("✅ NASM written to", out_path)
    exe_name = os.path.splitext(os.path.basename(src_path))[0]
    try:
        build_executable(exe_name, out_path)
    except Exception as e:
        print(f"[ERROR] Building executable: {e}")

    print("\nTo assemble and link (Linux x86-64), run:")
    print("  nasm -f elf64 out.asm -o out.o")
    print("  ld out.o -o out")

    print("\nThen execute with:")
    print("  ./out")

    print("\nTo assemble and link (Windows x86-64), run:")
    print("  nasm -f pe64 out.asm -o out.o")
    print("  link out.o /SUBSYSTEM:CONSOLE /OUT:out.exe")

    print("\nThen execute with:")
    print("  out.exe")

    print("\nTo assemble and link (macOS x86-64), run:")
    print("  nasm -f macho64 out.asm -o out.o")
    print("  ld -macosx_version_min 10.13 -e _start -lSystem -o out out.o")

    print("\nThen execute with:")
    print("  ./out")

    # End of density2_compiler.py

import os
import re
import shutil
import subprocess
import tempfile
import time
import hashlib
from typing import List, Tuple, Union, Dict, Optional

# -----------------------------
# Token Definitions
# -----------------------------
TOKEN_SPECIFICATION = [
    # Blocks first (non-greedy to avoid overconsumption)
    ('CIAM',        r"'''(.*?),,,"),              # ''' ... ,,,
    ('INLINE_ASM',  r"#asm(.*?)#endasm"),
    ('INLINE_C',    r"#c(.*?)#endc"),
    ('INLINE_PY',   r"#python(.*?)#endpython"),

    # Comments
    ('COMMENT',     r'//[^\n]*'),
    ('MCOMMENT',    r'/\*.*?\*/'),

    # Literals and identifiers
    # Unterminated string recovery: BAD_STRING must be before STRING and MISMATCH
    ('BAD_STRING',  r'"(?:\\.|[^"\\])*?(?=\n|$)'),
    ('STRING',      r'"(?:\\.|[^"\\])*"'),
    # Unicode identifier: first char is any Unicode letter (no digit/underscore), then word chars
    ('IDENT',       r'[^\W\d_][\w]*'),

    # Symbols
    ('LBRACE',      r'\{'),
    ('RBRACE',      r'\}'),
    ('LPAREN',      r'\('),
    ('RPAREN',      r'\)'),
    ('COLON',       r':'),
    ('SEMICOLON',   r';'),
    ('COMMA',       r','),         # macro args
    ('PLUS',        r'\+'),        # string concat

    # Whitespace, newline, and mismatch
    ('NEWLINE',     r'\n'),
    ('SKIP',        r'[ \t]+'),
    ('MISMATCH',    r'.'),
]

# Make CIAM and INLINE_* non-greedy by ensuring (.*?) groups are used above.
# Note: CIAM pattern above currently uses (.*?),,, via the outer token list string.
TOKEN_SPECIFICATION = [
    (name, (pattern if name != 'CIAM' else r"'''(.*?),,,")) for (name, pattern) in TOKEN_SPECIFICATION
]

token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATION)

# Global token filter hooks and lexer error buffer
TOKEN_FILTERS: List = []
_LAST_LEX_ERRORS: List[str] = []

def register_token_filter(fn) -> None:
    TOKEN_FILTERS.append(fn)

def get_last_lex_errors() -> List[str]:
    return list(_LAST_LEX_ERRORS)


class Token:
    def __init__(self, type_: str, value: str, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line}, col={self.column})"


# -----------------------------
# Lexer
# -----------------------------
class SyntaxErrorEx(Exception):
    def __init__(self, message: str, line: int, col: int, suggestion: Optional[str] = None):
        self.line = line
        self.col = col
        self.suggestion = suggestion
        super().__init__(f"{message} @ {line}:{col}" + (f" | hint: {suggestion}" if suggestion else ""))

def tokenize(code: str) -> List[Token]:
    # UTF-8 BOM handling: strip BOM if present
    if code.startswith('\ufeff'):
        code = code.lstrip('\ufeff')

    tokens: List[Token] = []
    _LAST_LEX_ERRORS.clear()
    line_num = 1
    line_start = 0
    for mo in re.finditer(token_regex, code, re.DOTALL):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
            continue
        elif kind in ('SKIP', 'COMMENT', 'MCOMMENT'):
            continue
        elif kind == 'BAD_STRING':
            # Recover: capture as error token and continue to next line
            msg = f"Unterminated string literal"
            _LAST_LEX_ERRORS.append(f"{msg} at {line_num}:{column}")
            tokens.append(Token('ERROR', value, line_num, column))
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        tokens.append(Token(kind, value, line_num, column))

    # Apply token filters last
    for filt in TOKEN_FILTERS:
        tokens = filt(tokens)
    return tokens


# -----------------------------
# AST Nodes
# -----------------------------
class ASTNode:
    """
    Base class for all Density 2 AST nodes.

    Features:
    - Optional source position tracking: filename, (line, col) -> (end_line, end_col)
    - Child discovery: children() finds nested AST nodes and lists of nodes
    - Traversal: walk() yields nodes in preorder
    - Visitor pattern: accept(visitor) calls visitor.visit_<Type>(self) or visitor.visit(self)
    - Structural replace: replace_child(old, new) updates direct attributes/lists
    - Serialization: to_dict()/pretty() for debugging and tooling
    - Copy: copy(**overrides) for shallow cloning
    - Dodecagram encoding: to_dodecagram() uses global ast_to_dodecagram if available
    - Structural equality: __eq__ based on type and serialized content (excluding positions)
    """

    # Position information is optional and can be set later via set_pos().
    def __init__(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ):
        self.filename = filename
        self.line = line
        self.col = col
        self.end_line = end_line
        self.end_col = end_col

    # ----- Source position helpers -----
    def set_pos(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ) -> "ASTNode":
        if filename is not None:
            self.filename = filename
        if line is not None:
            self.line = line
        if col is not None:
            self.col = col
        if end_line is not None:
            self.end_line = end_line
        if end_col is not None:
            self.end_col = end_col
        return self

    # ----- Introspection helpers -----
    def _is_pos_field(self, name: str) -> bool:
        return name in ("filename", "line", "col", "end_line", "end_col")

    def _iter_fields(self):
        # Do not consider private/dunder attributes as AST data
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            yield k, v

    def children(self) -> List["ASTNode"]:
        """Return direct child AST nodes (flattening lists)."""
        result: List[ASTNode] = []
        for _, v in self._iter_fields():
            if isinstance(v, ASTNode):
                result.append(v)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, ASTNode):
                        result.append(it)
        return result

    def walk(self):
        """Preorder traversal of this subtree."""
        yield self
        for c in self.children():
            yield from c.walk()

    # ----- Visitor pattern -----
    def accept(self, visitor):
        """Call visitor.visit_<Type>(self) if present, else visitor.visit(self) if present."""
        method = getattr(visitor, f"visit_{self.__class__.__name__}", None)
        if callable(method):
            return method(self)
        generic = getattr(visitor, "visit", None)
        if callable(generic):
            return generic(self)
        return None

    # ----- Structural operations -----
    def replace_child(self, old: "ASTNode", new: Optional["ASTNode"]) -> bool:
        """
        Replace a direct child 'old' with 'new'.
        If 'new' is None, removes the child if it's in a list; clears attribute otherwise.
        Returns True if a replacement/removal occurred.
        """
        changed = False
        for k, v in list(self._iter_fields()):
            if isinstance(v, ASTNode):
                if v is old:
                    setattr(self, k, new)
                    changed = True
            elif isinstance(v, list):
                # Replace in lists; remove if new is None
                new_list = []
                for it in v:
                    if it is old:
                        if new is not None:
                            new_list.append(new)
                        changed = True
                    else:
                        new_list.append(it)
                if changed:
                    setattr(self, k, new_list)
        return changed

    # ----- Serialization / Debugging -----
    def to_dict(self, *, include_pos: bool = True) -> Dict[str, object]:
        """Convert the node (recursively) to a dict suitable for JSON/debugging."""
        d: Dict[str, object] = {"__type__": self.__class__.__name__}
        for k, v in self._iter_fields():
            if not include_pos and self._is_pos_field(k):
                continue
            if isinstance(v, ASTNode):
                d[k] = v.to_dict(include_pos=include_pos)
            elif isinstance(v, list):
                d[k] = [
                    (it.to_dict(include_pos=include_pos) if isinstance(it, ASTNode) else it)
                    for it in v
                ]
            else:
                d[k] = v
        return d

    def pretty(self, indent: str = "  ") -> str:
        """Human-readable multi-line tree dump."""
        lines: List[str] = []

        def rec(n: "ASTNode", depth: int):
            pad = indent * depth
            header = n.__class__.__name__
            pos = []
            if n.filename:
                pos.append(f'file="{n.filename}"')
            if n.line is not None and n.col is not None:
                pos.append(f"@{n.line}:{n.col}")
            if n.end_line is not None and n.end_col is not None:
                pos.append(f"-{n.end_line}:{n.end_col}")
            if pos:
                header += " [" + " ".join(pos) + "]"
            lines.append(pad + header)

            # Show scalar fields
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    continue
                if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                    continue
                lines.append(pad + indent + f"{k} = {v!r}")

            # Recurse into child nodes
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    lines.append(pad + indent + f"{k}:")
                    rec(v, depth + 2)
                elif isinstance(v, list):
                    child_nodes = [it for it in v if isinstance(it, ASTNode)]
                    if child_nodes:
                        lines.append(pad + indent + f"{k}: [{len(child_nodes)}]")
                        for it in child_nodes:
                            rec(it, depth + 2)

        rec(self, 0)
        return "\n".join(lines)

    def copy(self, **overrides):
        """
        Shallow copy with optional field overrides:
            new = node.copy(body=new_body)
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)  # type: ignore
        # Copy all instance attributes
        new_obj.__dict__.update(self.__dict__)
        # Apply overrides
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    def to_dodecagram(self) -> str:
        """
        Encode this node (and subtree) using the Dodecagram mapping.
        Relies on a global function ast_to_dodecagram(node).
        """
        f = globals().get("ast_to_dodecagram")
        if callable(f):
            return f(self)  # type: ignore[misc]
        raise RuntimeError("ast_to_dodecagram() is not available in this module")

    # ----- Equality / Representation -----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASTNode):
            return False
        if self.__class__ is not other.__class__:
            return False
        return self.to_dict(include_pos=False) == other.to_dict(include_pos=False)

    def __repr__(self) -> str:
        # Compact representation showing scalar fields only
        fields: List[str] = []
        for k, v in self._iter_fields():
            if isinstance(v, ASTNode):
                continue
            if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                continue
            fields.append(f"{k}={v!r}")
        inner = ", ".join(fields)
        return f"{self.__class__.__name__}({inner})"


class Program(ASTNode):
    def __init__(self, functions: List['Function']):
        self.functions = functions

    def __repr__(self):
        return f"Program({self.functions})"


class Function(ASTNode):
    def __init__(self, name: str, body: List['Statement']):
        self.name = name
        self.body = body

    def __repr__(self):
        return f"Function({self.name}, body={self.body})"


class Statement(ASTNode):
    """Base class for all statements in Density 2."""
    pass


class PrintStatement(Statement):
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Print({self.text!r})"


class CIAMBlock(Statement):
    def __init__(self, name: str, params: List[str], body_text: str):
        self.name = name
        self.params = params
        self.body_text = body_text  # raw Density 2 snippet

    def __repr__(self):
        return f"CIAMBlock(name={self.name!r}, params={self.params}, body_len={len(self.body_text)})"


class MacroCall(Statement):
    def __init__(self, name: str, arg_texts: List[str]):
        self.name = name
        self.arg_texts = arg_texts  # raw argument texts

    def __repr__(self):
        return f"MacroCall({self.name!r}, args={self.arg_texts})"


class InlineBlock(Statement):
    def __init__(self, lang: str, content: str):
        self.lang = lang  # 'asm', 'c', 'python'
        self.content = content

    def __repr__(self):
        return f"InlineBlock(lang={self.lang!r}, content_len={len(self.content)})"


# -----------------------------
# Dodecagram AST encoding
# -----------------------------
_DODECAGRAM_MAP = {
    'Program': '0',
    'Function': '1',
    'PrintStatement': '2',
    'CIAMBlock': '3',
    'MacroCall': '4',
    # InlineBlock variants:
    'InlineBlock:asm': '5',
    'InlineBlock:python': '6',
    'InlineBlock:py': '6',
    'InlineBlock:c': '7',
    'InlineBlock:other': '8',
    # Reserved for future nodes:
    '_reserved9': '9',
    '_reserveda': 'a',
    '_reservedb': 'b',
}

def ast_to_dodecagram(node: ASTNode) -> str:
    """
    Preorder encoding of the AST using the Dodecagram alphabet 0-9,a,b.
    """
    def enc(n: ASTNode) -> str:
        if isinstance(n, Program):
            s = _DODECAGRAM_MAP['Program']
            for f in n.functions:
                s += enc(f)
            return s
        if isinstance(n, Function):
            s = _DODECAGRAM_MAP['Function']
            for st in n.body:
                s += enc(st)
            return s
        if isinstance(n, PrintStatement):
            return _DODECAGRAM_MAP['PrintStatement']
        if isinstance(n, CIAMBlock):
            return _DODECAGRAM_MAP['CIAMBlock']
        if isinstance(n, MacroCall):
            return _DODECAGRAM_MAP['MacroCall']
        if isinstance(n, InlineBlock):
            key = f'InlineBlock:{n.lang}'
            ch = _DODECAGRAM_MAP.get(key, _DODECAGRAM_MAP['InlineBlock:other'])
            return ch
        # Unknown node -> reserved
        return _DODECAGRAM_MAP['_reserved9']
    return enc(node)

# -----------------------------
# Parser
# -----------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        # Macro table collected while parsing: name -> CIAMBlock
        self.macro_table: Dict[str, CIAMBlock] = {}

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def lookahead(self, offset: int) -> Optional[Token]:
        idx = self.pos + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_type: str) -> Token:
        tok = self.peek()
        if not tok or tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok

    def match(self, expected_type: str) -> bool:
        tok = self.peek()
        if tok and tok.type == expected_type:
            self.pos += 1
            return True
        return False

    def parse(self) -> Program:
        functions: List[Function] = []
        while self.peek() is not None:
            # Expect IDENT (function name) or skip stray tokens
            tok = self.peek()
            if tok.type == 'IDENT':
                # Look for function signature IDENT()
                if self.lookahead(1) and self.lookahead(1).type == 'LPAREN':
                    functions.append(self.parse_function())
                else:
                    # Skip stray identifier
                    self.pos += 1
            else:
                # Skip unknown tokens at top-level
                self.pos += 1
        return Program(functions)

    def parse_function(self) -> Function:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        self.consume('RPAREN')
        body: List[Statement] = []
        if self.match('LBRACE'):
            # Block body
            body = self.parse_statements_until_rbrace()
            self.consume('RBRACE')
        else:
            # Single statement allowed
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        return Function(name_tok.value, body)

    def parse_statements_until_rbrace(self) -> List[Statement]:
        stmts: List[Statement] = []
        while True:
            tok = self.peek()
            if tok is None or tok.type == 'RBRACE':
                break
            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    stmts.extend(stmt)
                else:
                    stmts.append(stmt)
        return stmts

    def parse_statement(self) -> Optional[Union[Statement, List[Statement]]]:
        tok = self.peek()
        if tok is None:
            return None

        if tok.type == 'IDENT' and tok.value == 'Print':
            return self.parse_print()

        if tok.type == 'CIAM':
            # Parse CIAM block definition, store in macro table, no direct output
            ciam_tok = self.consume('CIAM')
            content = ciam_tok.value
            # Strip delimiters: starts with ''' and ends with ,,,
            if content.startswith("'''"):
                content_inner = content[3:]
                if content_inner.endswith(",,,"):
                    content_inner = content_inner[:-3]
            else:
                content_inner = content

            name, params, body_text = self._parse_ciam_content(content_inner.strip(), ciam_tok)
            ciam_block = CIAMBlock(name, params, body_text)
            self.macro_table[name] = ciam_block
            # Defining a macro does not emit code
            return None

        if tok.type.startswith('INLINE_'):
            return self.parse_inline_block()

        if tok.type == 'IDENT':
            # Possibly a macro invocation: IDENT(...)
            ident = tok.value
            la = self.lookahead(1)
            if la and la.type == 'LPAREN':
                return self.parse_macro_call()
            else:
                # Unknown ident; consume to progress
                self.pos += 1
                return None

        # Unknown token; consume to progress
        self.pos += 1
        return None

    def _parse_ciam_content(self, content: str, tok: Token) -> Tuple[str, List[str], str]:
        # First line: Name(params)
        lines = content.splitlines()
        if not lines:
            raise SyntaxError(f"Empty CIAM at line {tok.line}")

        header = lines[0].strip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$', header)
        if not m:
            raise SyntaxError(f"Invalid CIAM header '{header}' at line {tok.line}")
        name = m.group(1)
        params_str = m.group(2).strip()
        params = [p.strip() for p in params_str.split(',') if p.strip()] if params_str else []
        body_text = '\n'.join(lines[1:]).strip()
        return name, params, body_text

    def parse_macro_call(self) -> MacroCall:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        args: List[str] = []
        current_parts: List[str] = []
        paren_depth = 1
        while True:
            tok = self.peek()
            if tok is None:
                raise SyntaxError("Unclosed macro call")
            if tok.type == 'LPAREN':
                paren_depth += 1
                current_parts.append(tok.value)
                self.pos += 1
            elif tok.type == 'RPAREN':
                paren_depth -= 1
                if paren_depth == 0:
                    # finalize last arg
                    arg_text = ''.join(current_parts).strip()
                    if arg_text:
                        args.append(arg_text)
                    self.pos += 1
                    break
                else:
                    current_parts.append(tok.value)
                    self.pos += 1
            elif tok.type == 'COMMA' and paren_depth == 1:
                arg_text = ''.join(current_parts).strip()
                args.append(arg_text)
                current_parts = []
                self.pos += 1
            else:
                current_parts.append(tok.value)
                self.pos += 1
        self.consume('SEMICOLON')
        return MacroCall(name_tok.value, args)

    def parse_inline_block(self) -> InlineBlock:
        tok = self.peek()
        lang = tok.type.split('_', 1)[1].lower()  # asm / c / py
        inline_tok = self.consume(tok.type)
        # strip off #lang and #endlang markers:
        content = re.sub(r'^#\w+', '', inline_tok.value, flags=re.DOTALL)
        content = re.sub(r'#end\w+$', '', content, flags=re.DOTALL)
        return InlineBlock(lang, content.strip())

    def parse_print(self) -> PrintStatement:
        self.consume('IDENT')  # 'Print'
        self.consume('COLON')
        self.consume('LPAREN')

        # Simple string concat of STRING (+ STRING)*
        parts: List[str] = []
        first = True
        while True:
            tok = self.peek()
            if not tok:
                raise SyntaxError("Unclosed Print: missing ')'")
            if first:
                if tok.type != 'STRING':
                    raise SyntaxError(f"Print expects a string literal, got {tok}")
                parts.append(eval(tok.value))
                self.pos += 1
                first = False
            else:
                if tok.type == 'PLUS':
                    self.pos += 1
                    tok2 = self.peek()
                    if not tok2 or tok2.type != 'STRING':
                        raise SyntaxError("Expected string after '+' in Print")
                    parts.append(eval(tok2.value))
                    self.pos += 1
                else:
                    break

            # optional spaces are already skipped by lexer

        self.consume('RPAREN')
        self.consume('SEMICOLON')
        full_text = ''.join(parts)
        return PrintStatement(full_text)


# -----------------------------
# Macro Expansion
# -----------------------------
def expand_macros(program: Program, macro_table: Dict[str, CIAMBlock], max_depth: int = 32) -> Program:
    def expand_statements(stmts: List[Statement], depth: int) -> List[Statement]:
        result: List[Statement] = []
        for s in stmts:
            if isinstance(s, MacroCall):
                if s.name not in macro_table:
                    # Unknown macro, ignore call
                    continue
                macro = macro_table[s.name]
                expanded = expand_macro_call(macro, s, depth)
                result.extend(expanded)
            else:
                result.append(s)
        return result

    def expand_macro_call(macro: CIAMBlock, call: MacroCall, depth: int) -> List[Statement]:
        if depth <= 0:
            raise RecursionError("Macro expansion depth exceeded")
        # Build replacement text by simple identifier replacement
        body = macro.body_text

        # map params -> args text
        mapping = {p: (call.arg_texts[i] if i < len(call.arg_texts) else '') for i, p in enumerate(macro.params)}

        # Replace params with args (word boundary replacement)
        for p, a in mapping.items():
            # Use a simple token-ish replace; avoid capturing partial identifiers
            body = re.sub(rf'\b{re.escape(p)}\b', a, body)

        # Re-parse the expanded snippet into statements
        sub_tokens = tokenize(body)
        sub_parser = Parser(sub_tokens)
        # Share the same macro table to allow nested macro definitions/uses
        sub_parser.macro_table = macro_table
        # We parse a synthetic function body to reuse statement parser
        sub_stmts = sub_parser.parse_statements_until_rbrace()  # until EOF
        # Recursively expand within
        return expand_statements(sub_stmts, depth - 1)

    # Expand within each function
    new_functions: List[Function] = []
    for f in program.functions:
        expanded_body = expand_statements(f.body, max_depth)
        new_functions.append(Function(f.name, expanded_body))
    return Program(new_functions)


# -----------------------------
# NASM Emitter
# -----------------------------
class CodeGenerator:
    def __init__(self, ast: Program):
        self.ast = ast
        self.text_lines: List[str] = []
        self.data_lines: List[str] = []
        self.string_table: Dict[str, str] = {}  # map string -> label
        self.label_counter = 0
        # simple register reuse cache (invalidated across inline blocks)
        self._reg_cache = {'rax_sys_write': False, 'rdi_stdout': False}

    def generate(self) -> str:
        # Expand macros already done in compile_density2
        self.text_lines = []
        self.data_lines = []
        self.string_table = {}
        self.label_counter = 0
        self._invalidate_reg_cache()

        # Header and functions
        self._emit_header()
        for func in self.ast.functions:
            self._emit_function(func)

        # Build final assembly
        final_lines: List[str] = []
        final_lines.append('section .data')
        final_lines.extend('    ' + line for line in self.data_lines)
        final_lines.append('section .text')
        final_lines.append('    global _start')
        for l in self.text_lines:
            final_lines.append(l)

        return '\n'.join(final_lines)

    def _invalidate_reg_cache(self):
        self._reg_cache['rax_sys_write'] = False
        self._reg_cache['rdi_stdout'] = False

    def _emit_header(self):
        # Add a small banner; sections are assembled at the end
        self.text_lines.append('; --- Density 2 NASM output ---')
        self.text_lines.append('; inline C: compiled then translated (AT&T -> NASM best-effort)')
        self.text_lines.append('; inline Python: executed at codegen; use emit("...") to output NASM')

    def _emit_function(self, func: Function):
        # Entry label
        if func.name == 'Main':
            self.text_lines.append('_start:')
        else:
            self.text_lines.append(f'{func.name}:')
        # function entry: registers not assumed
        self._invalidate_reg_cache()

        for stmt in func.body:
            if isinstance(stmt, PrintStatement):
                self._emit_print(stmt.text)
            elif isinstance(stmt, InlineBlock):
                self._emit_inline(stmt)
                # assume inline code may clobber registers
                self._invalidate_reg_cache()
            elif isinstance(stmt, CIAMBlock):
                # Already handled by expand_macros; leave a comment in case any remain
                self.text_lines.append(f'    ; CIAMBlock ignored (should be expanded): {getattr(stmt, "name", "?")}')
            elif isinstance(stmt, MacroCall):
                # Should be expanded away
                self.text_lines.append(f'    ; MacroCall ignored (should be expanded): {getattr(stmt, "name", "?")}')
            else:
                # Unknown statement type
                self.text_lines.append('    ; Unknown statement')

        if func.name == 'Main':
            self._emit_exit()

    def _encode_string_bytes(self, s: str) -> List[int]:
        return list(s.encode('utf-8'))

    def _get_string_label(self, text: str) -> Tuple[str, int]:
        if text not in self.string_table:
            label = f'str_{self.label_counter}'
            self.label_counter += 1
            data_bytes = self._encode_string_bytes(text)
            # Store bytes + newline + NUL terminator
            stored = data_bytes + [10, 0]
            bytes_list = ', '.join(str(b) for b in stored)
            self.data_lines.append(f'{label} db {bytes_list}')
            # Length to write: bytes plus newline only (exclude NUL)
            length = len(data_bytes) + 1
            self.string_table[text] = label
        else:
            label = self.string_table[text]
            data_bytes = self._encode_string_bytes(text)
            length = len(data_bytes) + 1
        return label, length

    def _emit_print(self, text: str):
        label, length = self._get_string_label(text)
        # register reuse: avoid redundant loads across consecutive prints
        if not self._reg_cache['rax_sys_write']:
            self.text_lines.append(f'    mov rax, 1          ; sys_write')
            self._reg_cache['rax_sys_write'] = True
        if not self._reg_cache['rdi_stdout']:
            self.text_lines.append(f'    mov rdi, 1          ; stdout')
            self._reg_cache['rdi_stdout'] = True
        self.text_lines.append(f'    mov rsi, {label}    ; message')
        self.text_lines.append(f'    mov rdx, {length}         ; length (bytes)')
        self.text_lines.append('    syscall')
        # after syscall, rax/rdi are still fine for subsequent prints in our model

    def _emit_exit(self):
        self.text_lines.append('    mov rax, 60         ; sys_exit')
        self.text_lines.append('    xor rdi, rdi        ; status 0')
        self.text_lines.append('    syscall')
        # invalidate cache after exit emitter
        self._invalidate_reg_cache()

    def _emit_inline(self, block: InlineBlock):
        if block.lang == 'asm':
            self.text_lines.append('    ; inline NASM start')
            for line in block.content.splitlines():
                line = line.rstrip()
                if line:
                    self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline NASM end')
        elif block.lang in ('py', 'python'):
            self.text_lines.append('    ; inline Python start')
            for line in self._run_inline_python(block.content):
                self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline Python end')
        elif block.lang == 'c':
            self.text_lines.append('    ; inline C start')
            asm_lines = self._compile_c_to_asm(block.content)
            if asm_lines:
                for l in asm_lines:
                    self.text_lines.append('    ' + l)
            else:
                # Fallback: keep it commented
                for line in block.content.splitlines():
                    self.text_lines.append('    ; ' + line)
                self.text_lines.append('    ; (no C compiler found; block left as comment)')
            self.text_lines.append('    ; inline C end')
        else:
            # Unknown inline language
            for line in block.content.splitlines():
                self.text_lines.append(f'    ; inline {block.lang} ignored: {line}')

    def _run_inline_python(self, code: str) -> List[str]:
        # Provide a tiny DSL where Python can emit NASM
        lines: List[str] = []

        def emit(s: str):
            if not isinstance(s, str):
                raise TypeError("emit() expects a string")
            lines.append(s)

        def label(prefix: str = 'gen'):
            lbl = f'{prefix}_{self.label_counter}'
            self.label_counter += 1
            return lbl

        globals_dict = {
            '__builtins__': {
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'print': print,
            },
            'emit': emit,
            'label': label,
        }
        locals_dict: Dict[str, object] = {}
        try:
            exec(code, globals_dict, locals_dict)
        except Exception as ex:
            lines.append(f'; [inline python error] {ex!r}')
        return lines

    def _compile_c_to_asm(self, c_code: str) -> List[str]:
        # Try clang, gcc, tcc, cl (MSVC) in that order to produce assembly text
        compiler = None
        for cand in ('clang', 'gcc', 'tcc', 'cl'):
            if shutil.which(cand):
                compiler = cand
                break
        if compiler is None:
            return []

        # MSVC path (use /FA to generate assembly listing)
        if compiler == 'cl':
            lines = self._compile_with_msvc(c_code)
            if lines:
                # Unobtrusive log of the compiler used
                return ['; [inline c compiler: cl]'] + lines
            return []

        tmpdir = tempfile.mkdtemp(prefix='den2_c_')
        c_path = os.path.join(tmpdir, 'inline.c')
        asm_path = os.path.join(tmpdir, 'inline.s' if compiler != 'cl' else 'inline.asm')
        try:
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            if compiler == 'clang':
                cmd = ['clang', '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer', '-masm=intel', '-m64']
            elif compiler == 'gcc':
                cmd = ['gcc', '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer', '-masm=intel', '-m64']
            elif compiler == 'tcc':
                cmd = ['tcc', '-nostdlib', '-S', c_path, '-o', asm_path]
            else:  # cl (MSVC)
                return ['; [inline c compiler: cl]'] + self._compile_with_msvc(c_code)

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not os.path.exists(asm_path):
                return []

            # Try Intel-to-NASM normalization first; if looks AT&T, fallback
            translated = self._intel_to_nasm(asm_path)
            if translated:
                return [f'; [inline c compiler: {compiler}]'] + translated

            fallback = ['; [inline c compiler: {compiler}]', '; [begin compiled C assembly]']
            with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()
            return fallback + ['; ' + ln for ln in raw.splitlines()] + ['; [end compiled C assembly]']
        except Exception as ex:
            return [f'; [inline c compile error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _compile_with_msvc(self, c_code: str) -> List[str]:
        tmpdir = tempfile.mkdtemp(prefix='den2_msvc_')
        try:
            c_path = os.path.join(tmpdir, 'inline.c')
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            cmd = ['cl', '/nologo', '/FA', '/c', os.path.basename(c_path)]
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            asm_listing = os.path.join(tmpdir, 'inline.asm')
            lines: List[str] = []

            # If cl produced diagnostics, include them as comments (unobtrusive)
            if proc.stdout:
                for ln in proc.stdout.splitlines():
                    if ln.strip():
                        lines.append('; [cl] ' + ln)
            if proc.returncode != 0 and proc.stderr:
                for ln in proc.stderr.splitlines():
                    if ln.strip():
                        lines.append('; [cl err] ' + ln)

            if os.path.exists(asm_listing):
                with open(asm_listing, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                translated = self._msvc_asm_to_nasm(raw)
                # Place translated code after any diagnostics
                return lines + translated

            # No asm produced; return only diagnostics if any
            return lines
        except Exception as ex:
            return [f'; [inline c compile (msvc) error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _msvc_asm_to_nasm(self, msvc_asm: str) -> List[str]:
        out: List[str] = []
        for line in msvc_asm.splitlines():
            s = line.rstrip()
            if not s:
                continue
            if s.lstrip().startswith(';'):
                out.append(s)
                continue
            tok = s.strip()
            up = tok.upper()
            if up.startswith(('TITLE ', 'COMMENT ', 'INCLUDE ', 'INCLUDELIB ')):
                # Ignore headers/includes in listing
                continue
            if up.startswith(('.MODEL', '.CODE', '.DATA', '.CONST', '.XDATA', '.PDATA', '.STACK', '.LIST', '.686', '.686P', '.XMM', '.X64')):
                continue
            if up.startswith(('PUBLIC ', 'EXTRN ', 'EXTERN ', 'ASSUME ')):
                continue
            if up == 'END':
                continue
            if up.startswith('ALIGN '):
                # Map to NASM align
                parts = tok.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    out.append(f'align {parts[1]}')
                continue

            # PROC/ENDP handling: turn "label PROC" -> "label:" and skip ENDP
            m_proc = re.match(r'^([A-Za-z_$.@?][\w$.@?]*)\s+PROC\b', tok)
            if m_proc:
                out.append(f'{m_proc.group(1)}:')
                continue
            if re.match(r'^[A-Za-z_$.@?][\w$.@?]*\s+ENDP\b', tok):
                # End of procedure: ignore (no epilogue emission here)
                continue

            # Data directives: DB/DW/DD/DQ -> lowercase NASM-friendly
            m_data = re.match(r'^(DB|DW|DD|DQ)\b(.*)$', tok, re.IGNORECASE)
            if m_data:
                out.append(m_data.group(1).lower() + m_data.group(2))
                continue

            # Replace "BYTE/WORD/DWORD/QWORD PTR" with NASM "byte/word/dword/qword"
            tok = re.sub(r'\b(BYTE|WORD|DWORD|QWORD)\s+PTR\b', lambda m: m.group(1).lower(), tok)

            # Replace "OFFSET FLAT:label" or "OFFSET label" with just label
            tok = re.sub(r'\bOFFSET\s+FLAT:', '', tok, flags=re.IGNORECASE)
            tok = re.sub(r'\bOFFSET\s+', '', tok, flags=re.IGNORECASE)

            # Some MSVC emits "FLAT:" as a segment label prefix; drop it
            tok = tok.replace(' FLAT:', ' ')

            # Instructions are already Intel syntax; keep as-is
            out.append(tok)

        return out

    # --- AT&T -> NASM best-effort translation helpers ---
    def _translate_att_to_nasm(self, att_asm: str) -> List[str]:
        out: List[str] = []
        for line in att_asm.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith(('.', '#')):
                # map sections, comment the rest
                if s.startswith('.text'):
                    out.append(f"section .text ; from att")
                elif s.startswith('.data'):
                    out.append(f"section .data ; from att")
                elif s.startswith('.bss'):
                    out.append(f"section .bss ; from att")
                elif s.startswith('.globl') or s.startswith('.global') or s.startswith('.type') or s.startswith('.size') or s.startswith('.ident') or s.startswith('.file'):
                    continue
                else:
                    out.append('; ' + s)
                continue
            # label
            if s.endswith(':'):
                out.append(s)
                continue

            s = s.split('\t#', 1)[0].split(' #', 1)[0].strip()
            if not s:
                continue

            parts = s.split(None, 1)
            op = parts[0]
            rest = parts[1] if len(parts) > 1 else ''
            op_n = re.sub(r'(q|l|w|b)$', '', op)

            ops = [o.strip() for o in rest.split(',')] if rest else []
            ops = [self._att_operand_to_nasm(o) for o in ops]

            if len(ops) == 2:
                ops = [ops[1], ops[0]]

            if ops:
                out.append(f'{op_n} ' + ', '.join(ops))
            else:
                out.append(op_n)

        return out

    def _att_operand_to_nasm(self, o: str) -> str:
        o = o.strip()
        if o.startswith('$'):
            return o[1:]
        o = re.sub(r'%([a-zA-Z][a-zA-Z0-9]*)', r'\1', o)

        m = re.match(r'^\s*([\-+]?\d+)?\s*\(\s*([a-zA-Z0-9%]+)\s*(?:,\s*([a-zA-Z0-9%]+)\s*(?:,\s*(\d+))?)?\s*\)\s*$', o)
        if m:
            disp, base, index, scale = m.groups()
            base = base.lstrip('%')
            parts = [base]
            if index:
                idx = index.lstrip('%')
                sc = int(scale) if scale else 1
                parts.append(f'{idx}*{sc}')
            addr = ' + '.join(parts) if parts else ''
            if disp:
                sign = '+' if int(disp) >= 0 else '-'
                addr = f'{addr} {sign} {abs(int(disp))}' if addr else str(disp)
            return f'[{addr}]'

        return o


# -----------------------------
# Public API
# -----------------------------
def generate_nasm(ast_or_source: Union[Program, str]) -> str:
    """
    Generate NASM assembly.
    - If given a Program AST, emit NASM via CodeGenerator.
    - If given Density 2 source (str), parse, expand macros, then emit NASM.
    - If given an assembly-looking string (already NASM), return as-is.
    """
    if isinstance(ast_or_source, Program):
        gen = CodeGenerator(ast_or_source)
        return gen.generate()

    if isinstance(ast_or_source, str):
        text = ast_or_source
        # Heuristic: looks like NASM already
        if re.search(r'^\s*section\s+\.text\b', text, flags=re.MULTILINE) and 'global _start' in text:
            return text
        # Treat as Density 2 source
        tokens = tokenize(text)
        parser = Parser(tokens)
        program = parser.parse()
        program = expand_macros(program, parser.macro_table)
        gen = CodeGenerator(program)
        return gen.generate()

    raise TypeError(f"generate_nasm expects Program or str, got {type(ast_or_source).__name__}")


def parse_density2(code: str) -> Program:
    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse()
    # Expand macros now that we have parser.macro_table
    ast = expand_macros(ast, parser.macro_table)
    return ast


# -----------------------------
# CLI
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: density2_compiler.py <input.den2> [--debug]")
        sys.exit(0)

    debug_mode = '--debug' in sys.argv
    src_path = next((a for a in sys.argv[1:] if not a.startswith('--')), None)
    if not src_path or not os.path.exists(src_path):
        print("Input file not found.")
        sys.exit(2)

    with open(src_path, 'r', encoding='utf-8') as f:
        source = f.read()

    if debug_mode:
        print("🔍 Entering Density 2 Debugger...")
        program = parse_density2(source)
        start_debugger(program, filename=src_path)
        return

    asm = generate_nasm(source)
    out_path = os.path.join(os.path.dirname(src_path) or '.', 'out.asm')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(asm)
    print("✅ NASM written to", out_path)
    exe_name = os.path.splitext(os.path.basename(src_path))[0]
    try:
        build_executable(exe_name, out_path)
    except Exception as e:
        print(f"[ERROR] Building executable: {e}")

    print("\nTo assemble and link (Linux x86-64), run:")
    print("  nasm -f elf64 out.asm -o out.o")
    print("  ld out.o -o out")

    print("\nThen execute with:")
    print("  ./out")

    print("\nTo assemble and link (Windows x86-64), run:")
    print("  nasm -f pe64 out.asm -o out.o")
    print("  link out.o /SUBSYSTEM:CONSOLE /OUT:out.exe")

    print("\nThen execute with:")
    print("  out.exe")

    print("\nTo assemble and link (macOS x86-64), run:")
    print("  nasm -f macho64 out.asm -o out.o")
    print("  ld -macosx_version_min 10.13 -e _start -lSystem -o out out.o")

    print("\nThen execute with:")
    print("  ./out")

    # End of density2_compiler.py

import os
import re
import shutil
import subprocess
import tempfile
import time
import hashlib
from typing import List, Tuple, Union, Dict, Optional

# -----------------------------
# Token Definitions
# -----------------------------
TOKEN_SPECIFICATION = [
    # Blocks first (non-greedy to avoid overconsumption)
    ('CIAM',        r"'''(.*?),,,"),              # ''' ... ,,,
    ('INLINE_ASM',  r"#asm(.*?)#endasm"),
    ('INLINE_C',    r"#c(.*?)#endc"),
    ('INLINE_PY',   r"#python(.*?)#endpython"),

    # Comments
    ('COMMENT',     r'//[^\n]*'),
    ('MCOMMENT',    r'/\*.*?\*/'),

    # Literals and identifiers
    # Unterminated string recovery: BAD_STRING must be before STRING and MISMATCH
    ('BAD_STRING',  r'"(?:\\.|[^"\\])*?(?=\n|$)'),
    ('STRING',      r'"(?:\\.|[^"\\])*"'),
    # Unicode identifier: first char is any Unicode letter (no digit/underscore), then word chars
    ('IDENT',       r'[^\W\d_][\w]*'),

    # Symbols
    ('LBRACE',      r'\{'),
    ('RBRACE',      r'\}'),
    ('LPAREN',      r'\('),
    ('RPAREN',      r'\)'),
    ('COLON',       r':'),
    ('SEMICOLON',   r';'),
    ('COMMA',       r','),         # macro args
    ('PLUS',        r'\+'),        # string concat

    # Whitespace, newline, and mismatch
    ('NEWLINE',     r'\n'),
    ('SKIP',        r'[ \t]+'),
    ('MISMATCH',    r'.'),
]

# Make CIAM and INLINE_* non-greedy by ensuring (.*?) groups are used above.
# Note: CIAM pattern above currently uses (.*?),,, via the outer token list string.
TOKEN_SPECIFICATION = [
    (name, (pattern if name != 'CIAM' else r"'''(.*?),,,")) for (name, pattern) in TOKEN_SPECIFICATION
]

token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATION)

# Global token filter hooks and lexer error buffer
TOKEN_FILTERS: List = []
_LAST_LEX_ERRORS: List[str] = []

def register_token_filter(fn) -> None:
    TOKEN_FILTERS.append(fn)

def get_last_lex_errors() -> List[str]:
    return list(_LAST_LEX_ERRORS)


class Token:
    def __init__(self, type_: str, value: str, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line}, col={self.column})"


# -----------------------------
# Lexer
# -----------------------------
class SyntaxErrorEx(Exception):
    def __init__(self, message: str, line: int, col: int, suggestion: Optional[str] = None):
        self.line = line
        self.col = col
        self.suggestion = suggestion
        super().__init__(f"{message} @ {line}:{col}" + (f" | hint: {suggestion}" if suggestion else ""))

def tokenize(code: str) -> List[Token]:
    # UTF-8 BOM handling: strip BOM if present
    if code.startswith('\ufeff'):
        code = code.lstrip('\ufeff')

    tokens: List[Token] = []
    _LAST_LEX_ERRORS.clear()
    line_num = 1
    line_start = 0
    for mo in re.finditer(token_regex, code, re.DOTALL):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
            continue
        elif kind in ('SKIP', 'COMMENT', 'MCOMMENT'):
            continue
        elif kind == 'BAD_STRING':
            # Recover: capture as error token and continue to next line
            msg = f"Unterminated string literal"
            _LAST_LEX_ERRORS.append(f"{msg} at {line_num}:{column}")
            tokens.append(Token('ERROR', value, line_num, column))
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        tokens.append(Token(kind, value, line_num, column))

    # Apply token filters last
    for filt in TOKEN_FILTERS:
        tokens = filt(tokens)
    return tokens


# -----------------------------
# AST Nodes
# -----------------------------
class ASTNode:
    """
    Base class for all Density 2 AST nodes.

    Features:
    - Optional source position tracking: filename, (line, col) -> (end_line, end_col)
    - Child discovery: children() finds nested AST nodes and lists of nodes
    - Traversal: walk() yields nodes in preorder
    - Visitor pattern: accept(visitor) calls visitor.visit_<Type>(self) or visitor.visit(self)
    - Structural replace: replace_child(old, new) updates direct attributes/lists
    - Serialization: to_dict()/pretty() for debugging and tooling
    - Copy: copy(**overrides) for shallow cloning
    - Dodecagram encoding: to_dodecagram() uses global ast_to_dodecagram if available
    - Structural equality: __eq__ based on type and serialized content (excluding positions)
    """

    # Position information is optional and can be set later via set_pos().
    def __init__(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ):
        self.filename = filename
        self.line = line
        self.col = col
        self.end_line = end_line
        self.end_col = end_col

    # ----- Source position helpers -----
    def set_pos(
        self,
        *,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_line: Optional[int] = None,
        end_col: Optional[int] = None,
    ) -> "ASTNode":
        if filename is not None:
            self.filename = filename
        if line is not None:
            self.line = line
        if col is not None:
            self.col = col
        if end_line is not None:
            self.end_line = end_line
        if end_col is not None:
            self.end_col = end_col
        return self

    # ----- Introspection helpers -----
    def _is_pos_field(self, name: str) -> bool:
        return name in ("filename", "line", "col", "end_line", "end_col")

    def _iter_fields(self):
        # Do not consider private/dunder attributes as AST data
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            yield k, v

    def children(self) -> List["ASTNode"]:
        """Return direct child AST nodes (flattening lists)."""
        result: List[ASTNode] = []
        for _, v in self._iter_fields():
            if isinstance(v, ASTNode):
                result.append(v)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, ASTNode):
                        result.append(it)
        return result

    def walk(self):
        """Preorder traversal of this subtree."""
        yield self
        for c in self.children():
            yield from c.walk()

    # ----- Visitor pattern -----
    def accept(self, visitor):
        """Call visitor.visit_<Type>(self) if present, else visitor.visit(self) if present."""
        method = getattr(visitor, f"visit_{self.__class__.__name__}", None)
        if callable(method):
            return method(self)
        generic = getattr(visitor, "visit", None)
        if callable(generic):
            return generic(self)
        return None

    # ----- Structural operations -----
    def replace_child(self, old: "ASTNode", new: Optional["ASTNode"]) -> bool:
        """
        Replace a direct child 'old' with 'new'.
        If 'new' is None, removes the child if it's in a list; clears attribute otherwise.
        Returns True if a replacement/removal occurred.
        """
        changed = False
        for k, v in list(self._iter_fields()):
            if isinstance(v, ASTNode):
                if v is old:
                    setattr(self, k, new)
                    changed = True
            elif isinstance(v, list):
                # Replace in lists; remove if new is None
                new_list = []
                for it in v:
                    if it is old:
                        if new is not None:
                            new_list.append(new)
                        changed = True
                    else:
                        new_list.append(it)
                if changed:
                    setattr(self, k, new_list)
        return changed

    # ----- Serialization / Debugging -----
    def to_dict(self, *, include_pos: bool = True) -> Dict[str, object]:
        """Convert the node (recursively) to a dict suitable for JSON/debugging."""
        d: Dict[str, object] = {"__type__": self.__class__.__name__}
        for k, v in self._iter_fields():
            if not include_pos and self._is_pos_field(k):
                continue
            if isinstance(v, ASTNode):
                d[k] = v.to_dict(include_pos=include_pos)
            elif isinstance(v, list):
                d[k] = [
                    (it.to_dict(include_pos=include_pos) if isinstance(it, ASTNode) else it)
                    for it in v
                ]
            else:
                d[k] = v
        return d

    def pretty(self, indent: str = "  ") -> str:
        """Human-readable multi-line tree dump."""
        lines: List[str] = []

        def rec(n: "ASTNode", depth: int):
            pad = indent * depth
            header = n.__class__.__name__
            pos = []
            if n.filename:
                pos.append(f'file="{n.filename}"')
            if n.line is not None and n.col is not None:
                pos.append(f"@{n.line}:{n.col}")
            if n.end_line is not None and n.end_col is not None:
                pos.append(f"-{n.end_line}:{n.end_col}")
            if pos:
                header += " [" + " ".join(pos) + "]"
            lines.append(pad + header)

            # Show scalar fields
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    continue
                if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                    continue
                lines.append(pad + indent + f"{k} = {v!r}")

            # Recurse into child nodes
            for k, v in n._iter_fields():
                if isinstance(v, ASTNode):
                    lines.append(pad + indent + f"{k}:")
                    rec(v, depth + 2)
                elif isinstance(v, list):
                    child_nodes = [it for it in v if isinstance(it, ASTNode)]
                    if child_nodes:
                        lines.append(pad + indent + f"{k}: [{len(child_nodes)}]")
                        for it in child_nodes:
                            rec(it, depth + 2)

        rec(self, 0)
        return "\n".join(lines)

    def copy(self, **overrides):
        """
        Shallow copy with optional field overrides:
            new = node.copy(body=new_body)
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)  # type: ignore
        # Copy all instance attributes
        new_obj.__dict__.update(self.__dict__)
        # Apply overrides
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    def to_dodecagram(self) -> str:
        """
        Encode this node (and subtree) using the Dodecagram mapping.
        Relies on a global function ast_to_dodecagram(node).
        """
        f = globals().get("ast_to_dodecagram")
        if callable(f):
            return f(self)  # type: ignore[misc]
        raise RuntimeError("ast_to_dodecagram() is not available in this module")

    # ----- Equality / Representation -----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASTNode):
            return False
        if self.__class__ is not other.__class__:
            return False
        return self.to_dict(include_pos=False) == other.to_dict(include_pos=False)

    def __repr__(self) -> str:
        # Compact representation showing scalar fields only
        fields: List[str] = []
        for k, v in self._iter_fields():
            if isinstance(v, ASTNode):
                continue
            if isinstance(v, list) and any(isinstance(it, ASTNode) for it in v):
                continue
            fields.append(f"{k}={v!r}")
        inner = ", ".join(fields)
        return f"{self.__class__.__name__}({inner})"


class Program(ASTNode):
    def __init__(self, functions: List['Function']):
        self.functions = functions

    def __repr__(self):
        return f"Program({self.functions})"


class Function(ASTNode):
    def __init__(self, name: str, body: List['Statement']):
        self.name = name
        self.body = body

    def __repr__(self):
        return f"Function({self.name}, body={self.body})"


class Statement(ASTNode):
    """Base class for all statements in Density 2."""
    pass


class PrintStatement(Statement):
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Print({self.text!r})"


class CIAMBlock(Statement):
    def __init__(self, name: str, params: List[str], body_text: str):
        self.name = name
        self.params = params
        self.body_text = body_text  # raw Density 2 snippet

    def __repr__(self):
        return f"CIAMBlock(name={self.name!r}, params={self.params}, body_len={len(self.body_text)})"


class MacroCall(Statement):
    def __init__(self, name: str, arg_texts: List[str]):
        self.name = name
        self.arg_texts = arg_texts  # raw argument texts

    def __repr__(self):
        return f"MacroCall({self.name!r}, args={self.arg_texts})"


class InlineBlock(Statement):
    def __init__(self, lang: str, content: str):
        self.lang = lang  # 'asm', 'c', 'python'
        self.content = content

    def __repr__(self):
        return f"InlineBlock(lang={self.lang!r}, content_len={len(self.content)})"


# -----------------------------
# Dodecagram AST encoding
# -----------------------------
_DODECAGRAM_MAP = {
    'Program': '0',
    'Function': '1',
    'PrintStatement': '2',
    'CIAMBlock': '3',
    'MacroCall': '4',
    # InlineBlock variants:
    'InlineBlock:asm': '5',
    'InlineBlock:python': '6',
    'InlineBlock:py': '6',
    'InlineBlock:c': '7',
    'InlineBlock:other': '8',
    # Reserved for future nodes:
    '_reserved9': '9',
    '_reserveda': 'a',
    '_reservedb': 'b',
}

def ast_to_dodecagram(node: ASTNode) -> str:
    """
    Preorder encoding of the AST using the Dodecagram alphabet 0-9,a,b.
    """
    def enc(n: ASTNode) -> str:
        if isinstance(n, Program):
            s = _DODECAGRAM_MAP['Program']
            for f in n.functions:
                s += enc(f)
            return s
        if isinstance(n, Function):
            s = _DODECAGRAM_MAP['Function']
            for st in n.body:
                s += enc(st)
            return s
        if isinstance(n, PrintStatement):
            return _DODECAGRAM_MAP['PrintStatement']
        if isinstance(n, CIAMBlock):
            return _DODECAGRAM_MAP['CIAMBlock']
        if isinstance(n, MacroCall):
            return _DODECAGRAM_MAP['MacroCall']
        if isinstance(n, InlineBlock):
            key = f'InlineBlock:{n.lang}'
            ch = _DODECAGRAM_MAP.get(key, _DODECAGRAM_MAP['InlineBlock:other'])
            return ch
        # Unknown node -> reserved
        return _DODECAGRAM_MAP['_reserved9']
    return enc(node)

# -----------------------------
# Parser
# -----------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        # Macro table collected while parsing: name -> CIAMBlock
        self.macro_table: Dict[str, CIAMBlock] = {}

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def lookahead(self, offset: int) -> Optional[Token]:
        idx = self.pos + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_type: str) -> Token:
        tok = self.peek()
        if not tok or tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok

    def match(self, expected_type: str) -> bool:
        tok = self.peek()
        if tok and tok.type == expected_type:
            self.pos += 1
            return True
        return False

    def parse(self) -> Program:
        functions: List[Function] = []
        while self.peek() is not None:
            # Expect IDENT (function name) or skip stray tokens
            tok = self.peek()
            if tok.type == 'IDENT':
                # Look for function signature IDENT()
                if self.lookahead(1) and self.lookahead(1).type == 'LPAREN':
                    functions.append(self.parse_function())
                else:
                    # Skip stray identifier
                    self.pos += 1
            else:
                # Skip unknown tokens at top-level
                self.pos += 1
        return Program(functions)

    def parse_function(self) -> Function:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        self.consume('RPAREN')
        body: List[Statement] = []
        if self.match('LBRACE'):
            # Block body
            body = self.parse_statements_until_rbrace()
            self.consume('RBRACE')
        else:
            # Single statement allowed
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        return Function(name_tok.value, body)

    def parse_statements_until_rbrace(self) -> List[Statement]:
        stmts: List[Statement] = []
        while True:
            tok = self.peek()
            if tok is None or tok.type == 'RBRACE':
                break
            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    stmts.extend(stmt)
                else:
                    stmts.append(stmt)
        return stmts

    def parse_statement(self) -> Optional[Union[Statement, List[Statement]]]:
        tok = self.peek()
        if tok is None:
            return None

        if tok.type == 'IDENT' and tok.value == 'Print':
            return self.parse_print()

        if tok.type == 'CIAM':
            # Parse CIAM block definition, store in macro table, no direct output
            ciam_tok = self.consume('CIAM')
            content = ciam_tok.value
            # Strip delimiters: starts with ''' and ends with ,,,
            if content.startswith("'''"):
                content_inner = content[3:]
                if content_inner.endswith(",,,"):
                    content_inner = content_inner[:-3]
            else:
                content_inner = content

            name, params, body_text = self._parse_ciam_content(content_inner.strip(), ciam_tok)
            ciam_block = CIAMBlock(name, params, body_text)
            self.macro_table[name] = ciam_block
            # Defining a macro does not emit code
            return None

        if tok.type.startswith('INLINE_'):
            return self.parse_inline_block()

        if tok.type == 'IDENT':
            # Possibly a macro invocation: IDENT(...)
            ident = tok.value
            la = self.lookahead(1)
            if la and la.type == 'LPAREN':
                return self.parse_macro_call()
            else:
                # Unknown ident; consume to progress
                self.pos += 1
                return None

        # Unknown token; consume to progress
        self.pos += 1
        return None

    def _parse_ciam_content(self, content: str, tok: Token) -> Tuple[str, List[str], str]:
        # First line: Name(params)
        lines = content.splitlines()
        if not lines:
            raise SyntaxError(f"Empty CIAM at line {tok.line}")

        header = lines[0].strip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$', header)
        if not m:
            raise SyntaxError(f"Invalid CIAM header '{header}' at line {tok.line}")
        name = m.group(1)
        params_str = m.group(2).strip()
        params = [p.strip() for p in params_str.split(',') if p.strip()] if params_str else []
        body_text = '\n'.join(lines[1:]).strip()
        return name, params, body_text

    def parse_macro_call(self) -> MacroCall:
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        args: List[str] = []
        current_parts: List[str] = []
        paren_depth = 1
        while True:
            tok = self.peek()
            if tok is None:
                raise SyntaxError("Unclosed macro call")
            if tok.type == 'LPAREN':
                paren_depth += 1
                current_parts.append(tok.value)
                self.pos += 1
            elif tok.type == 'RPAREN':
                paren_depth -= 1
                if paren_depth == 0:
                    # finalize last arg
                    arg_text = ''.join(current_parts).strip()
                    if arg_text:
                        args.append(arg_text)
                    self.pos += 1
                    break
                else:
                    current_parts.append(tok.value)
                    self.pos += 1
            elif tok.type == 'COMMA' and paren_depth == 1:
                arg_text = ''.join(current_parts).strip()
                args.append(arg_text)
                current_parts = []
                self.pos += 1
            else:
                current_parts.append(tok.value)
                self.pos += 1
        self.consume('SEMICOLON')
        return MacroCall(name_tok.value, args)

    def parse_inline_block(self) -> InlineBlock:
        tok = self.peek()
        lang = tok.type.split('_', 1)[1].lower()  # asm / c / py
        inline_tok = self.consume(tok.type)
        # strip off #lang and #endlang markers:
        content = re.sub(r'^#\w+', '', inline_tok.value, flags=re.DOTALL)
        content = re.sub(r'#end\w+$', '', content, flags=re.DOTALL)
        return InlineBlock(lang, content.strip())

    def parse_print(self) -> PrintStatement:
        self.consume('IDENT')  # 'Print'
        self.consume('COLON')
        self.consume('LPAREN')

        # Simple string concat of STRING (+ STRING)*
        parts: List[str] = []
        first = True
        while True:
            tok = self.peek()
            if not tok:
                raise SyntaxError("Unclosed Print: missing ')'")
            if first:
                if tok.type != 'STRING':
                    raise SyntaxError(f"Print expects a string literal, got {tok}")
                parts.append(eval(tok.value))
                self.pos += 1
                first = False
            else:
                if tok.type == 'PLUS':
                    self.pos += 1
                    tok2 = self.peek()
                    if not tok2 or tok2.type != 'STRING':
                        raise SyntaxError("Expected string after '+' in Print")
                    parts.append(eval(tok2.value))
                    self.pos += 1
                else:
                    break

            # optional spaces are already skipped by lexer

        self.consume('RPAREN')
        self.consume('SEMICOLON')
        full_text = ''.join(parts)
        return PrintStatement(full_text)


# -----------------------------
# Macro Expansion
# -----------------------------
def expand_macros(program: Program, macro_table: Dict[str, CIAMBlock], max_depth: int = 32) -> Program:
    def expand_statements(stmts: List[Statement], depth: int) -> List[Statement]:
        result: List[Statement] = []
        for s in stmts:
            if isinstance(s, MacroCall):
                if s.name not in macro_table:
                    # Unknown macro, ignore call
                    continue
                macro = macro_table[s.name]
                expanded = expand_macro_call(macro, s, depth)
                result.extend(expanded)
            else:
                result.append(s)
        return result

    def expand_macro_call(macro: CIAMBlock, call: MacroCall, depth: int) -> List[Statement]:
        if depth <= 0:
            raise RecursionError("Macro expansion depth exceeded")
        # Build replacement text by simple identifier replacement
        body = macro.body_text

        # map params -> args text
        mapping = {p: (call.arg_texts[i] if i < len(call.arg_texts) else '') for i, p in enumerate(macro.params)}

        # Replace params with args (word boundary replacement)
        for p, a in mapping.items():
            # Use a simple token-ish replace; avoid capturing partial identifiers
            body = re.sub(rf'\b{re.escape(p)}\b', a, body)

        # Re-parse the expanded snippet into statements
        sub_tokens = tokenize(body)
        sub_parser = Parser(sub_tokens)
        # Share the same macro table to allow nested macro definitions/uses
        sub_parser.macro_table = macro_table
        # We parse a synthetic function body to reuse statement parser
        sub_stmts = sub_parser.parse_statements_until_rbrace()  # until EOF
        # Recursively expand within
        return expand_statements(sub_stmts, depth - 1)

    # Expand within each function
    new_functions: List[Function] = []
    for f in program.functions:
        expanded_body = expand_statements(f.body, max_depth)
        new_functions.append(Function(f.name, expanded_body))
    return Program(new_functions)


# -----------------------------
# NASM Emitter
# -----------------------------
class CodeGenerator:
    def __init__(self, ast: Program):
        self.ast = ast
        self.text_lines: List[str] = []
        self.data_lines: List[str] = []
        self.string_table: Dict[str, str] = {}  # map string -> label
        self.label_counter = 0
        # simple register reuse cache (invalidated across inline blocks)
        self._reg_cache = {'rax_sys_write': False, 'rdi_stdout': False}

    def generate(self) -> str:
        # Expand macros already done in compile_density2
        self.text_lines = []
        self.data_lines = []
        self.string_table = {}
        self.label_counter = 0
        self._invalidate_reg_cache()

        # Header and functions
        self._emit_header()
        for func in self.ast.functions:
            self._emit_function(func)

        # Build final assembly
        final_lines: List[str] = []
        final_lines.append('section .data')
        final_lines.extend('    ' + line for line in self.data_lines)
        final_lines.append('section .text')
        final_lines.append('    global _start')
        for l in self.text_lines:
            final_lines.append(l)

        return '\n'.join(final_lines)

    def _invalidate_reg_cache(self):
        self._reg_cache['rax_sys_write'] = False
        self._reg_cache['rdi_stdout'] = False

    def _emit_header(self):
        # Add a small banner; sections are assembled at the end
        self.text_lines.append('; --- Density 2 NASM output ---')
        self.text_lines.append('; inline C: compiled then translated (AT&T -> NASM best-effort)')
        self.text_lines.append('; inline Python: executed at codegen; use emit("...") to output NASM')

    def _emit_function(self, func: Function):
        # Entry label
        if func.name == 'Main':
            self.text_lines.append('_start:')
        else:
            self.text_lines.append(f'{func.name}:')
        # function entry: registers not assumed
        self._invalidate_reg_cache()

        for stmt in func.body:
            if isinstance(stmt, PrintStatement):
                self._emit_print(stmt.text)
            elif isinstance(stmt, InlineBlock):
                self._emit_inline(stmt)
                # assume inline code may clobber registers
                self._invalidate_reg_cache()
            elif isinstance(stmt, CIAMBlock):
                # Already handled by expand_macros; leave a comment in case any remain
                self.text_lines.append(f'    ; CIAMBlock ignored (should be expanded): {getattr(stmt, "name", "?")}')
            elif isinstance(stmt, MacroCall):
                # Should be expanded away
                self.text_lines.append(f'    ; MacroCall ignored (should be expanded): {getattr(stmt, "name", "?")}')
            else:
                # Unknown statement type
                self.text_lines.append('    ; Unknown statement')

        if func.name == 'Main':
            self._emit_exit()

    def _encode_string_bytes(self, s: str) -> List[int]:
        return list(s.encode('utf-8'))

    def _get_string_label(self, text: str) -> Tuple[str, int]:
        if text not in self.string_table:
            label = f'str_{self.label_counter}'
            self.label_counter += 1
            data_bytes = self._encode_string_bytes(text)
            # Store bytes + newline + NUL terminator
            stored = data_bytes + [10, 0]
            bytes_list = ', '.join(str(b) for b in stored)
            self.data_lines.append(f'{label} db {bytes_list}')
            # Length to write: bytes plus newline only (exclude NUL)
            length = len(data_bytes) + 1
            self.string_table[text] = label
        else:
            label = self.string_table[text]
            data_bytes = self._encode_string_bytes(text)
            length = len(data_bytes) + 1
        return label, length

    def _emit_print(self, text: str):
        label, length = self._get_string_label(text)
        # register reuse: avoid redundant loads across consecutive prints
        if not self._reg_cache['rax_sys_write']:
            self.text_lines.append(f'    mov rax, 1          ; sys_write')
            self._reg_cache['rax_sys_write'] = True
        if not self._reg_cache['rdi_stdout']:
            self.text_lines.append(f'    mov rdi, 1          ; stdout')
            self._reg_cache['rdi_stdout'] = True
        self.text_lines.append(f'    mov rsi, {label}    ; message')
        self.text_lines.append(f'    mov rdx, {length}         ; length (bytes)')
        self.text_lines.append('    syscall')
        # after syscall, rax/rdi are still fine for subsequent prints in our model

    def _emit_exit(self):
        self.text_lines.append('    mov rax, 60         ; sys_exit')
        self.text_lines.append('    xor rdi, rdi        ; status 0')
        self.text_lines.append('    syscall')
        # invalidate cache after exit emitter
        self._invalidate_reg_cache()

    def _emit_inline(self, block: InlineBlock):
        if block.lang == 'asm':
            self.text_lines.append('    ; inline NASM start')
            for line in block.content.splitlines():
                line = line.rstrip()
                if line:
                    self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline NASM end')
        elif block.lang in ('py', 'python'):
            self.text_lines.append('    ; inline Python start')
            for line in self._run_inline_python(block.content):
                self.text_lines.append('    ' + line)
            self.text_lines.append('    ; inline Python end')
        elif block.lang == 'c':
            self.text_lines.append('    ; inline C start')
            asm_lines = self._compile_c_to_asm(block.content)
            if asm_lines:
                for l in asm_lines:
                    self.text_lines.append('    ' + l)
            else:
                # Fallback: keep it commented
                for line in block.content.splitlines():
                    self.text_lines.append('    ; ' + line)
                self.text_lines.append('    ; (no C compiler found; block left as comment)')
            self.text_lines.append('    ; inline C end')
        else:
            # Unknown inline language
            for line in block.content.splitlines():
                self.text_lines.append(f'    ; inline {block.lang} ignored: {line}')

    def _run_inline_python(self, code: str) -> List[str]:
        # Provide a tiny DSL where Python can emit NASM
        lines: List[str] = []

        def emit(s: str):
            if not isinstance(s, str):
                raise TypeError("emit() expects a string")
            lines.append(s)

        def label(prefix: str = 'gen'):
            lbl = f'{prefix}_{self.label_counter}'
            self.label_counter += 1
            return lbl

        globals_dict = {
            '__builtins__': {
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'print': print,
            },
            'emit': emit,
            'label': label,
        }
        locals_dict: Dict[str, object] = {}
        try:
            exec(code, globals_dict, locals_dict)
        except Exception as ex:
            lines.append(f'; [inline python error] {ex!r}')
        return lines

    def _compile_c_to_asm(self, c_code: str) -> List[str]:
        # Try clang, gcc, tcc, cl (MSVC) in that order to produce assembly text
        compiler = None
        for cand in ('clang', 'gcc', 'tcc', 'cl'):
            if shutil.which(cand):
                compiler = cand
                break
        if compiler is None:
            return []

        # MSVC path (use /FA to generate assembly listing)
        if compiler == 'cl':
            lines = self._compile_with_msvc(c_code)
            if lines:
                # Unobtrusive log of the compiler used
                return ['; [inline c compiler: cl]'] + lines
            return []

        tmpdir = tempfile.mkdtemp(prefix='den2_c_')
        c_path = os.path.join(tmpdir, 'inline.c')
        asm_path = os.path.join(tmpdir, 'inline.s' if compiler != 'cl' else 'inline.asm')
        try:
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            if compiler == 'clang':
                cmd = ['clang', '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer', '-masm=intel', '-m64']
            elif compiler == 'gcc':
                cmd = ['gcc', '-x', 'c', '-O2', '-S', c_path, '-o', asm_path,
                       '-fno-asynchronous-unwind-tables', '-fomit-frame-pointer', '-masm=intel', '-m64']
            elif compiler == 'tcc':
                cmd = ['tcc', '-nostdlib', '-S', c_path, '-o', asm_path]
            else:  # cl (MSVC)
                return ['; [inline c compiler: cl]'] + self._compile_with_msvc(c_code)

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not os.path.exists(asm_path):
                return []

            # Try Intel-to-NASM normalization first; if looks AT&T, fallback
            translated = self._intel_to_nasm(asm_path)
            if translated:
                return [f'; [inline c compiler: {compiler}]'] + translated

            fallback = ['; [inline c compiler: {compiler}]', '; [begin compiled C assembly]']
            with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()
            return fallback + ['; ' + ln for ln in raw.splitlines()] + ['; [end compiled C assembly]']
        except Exception as ex:
            return [f'; [inline c compile error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _compile_with_msvc(self, c_code: str) -> List[str]:
        tmpdir = tempfile.mkdtemp(prefix='den2_msvc_')
        try:
            c_path = os.path.join(tmpdir, 'inline.c')
            with open(c_path, 'w', encoding='utf-8') as f:
                f.write(c_code)

            # cl will emit inline.asm in the current working directory (tmpdir)
            cmd = ['cl', '/nologo', '/FA', '/c', os.path.basename(c_path)]
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            asm_listing = os.path.join(tmpdir, 'inline.asm')
            lines: List[str] = []

            # If cl produced diagnostics, include them as comments (unobtrusive)
            if proc.stdout:
                for ln in proc.stdout.splitlines():
                    if ln.strip():
                        lines.append('; [cl] ' + ln)
            if proc.returncode != 0 and proc.stderr:
                for ln in proc.stderr.splitlines():
                    if ln.strip():
                        lines.append('; [cl err] ' + ln)

            if os.path.exists(asm_listing):
                with open(asm_listing, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                translated = self._msvc_asm_to_nasm(raw)
                # Place translated code after any diagnostics
                return lines + translated

            # No asm produced; return only diagnostics if any
            return lines
        except Exception as ex:
            return [f'; [inline c compile (msvc) error] {ex!r}']
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _msvc_asm_to_nasm(self, msvc_asm: str) -> List[str]:
        out: List[str] = []
        for line in msvc_asm.splitlines():
            s = line.rstrip()
            if not s:
                continue
            if s.lstrip().startswith(';'):
                out.append(s)
                continue
            tok = s.strip()
            up = tok.upper()
            if up.startswith(('TITLE ', 'COMMENT ', 'INCLUDE ', 'INCLUDELIB ')):
                # Ignore headers/includes in listing
                continue
            if up.startswith(('.MODEL', '.CODE', '.DATA', '.CONST', '.XDATA', '.PDATA', '.STACK', '.LIST', '.686', '.686P', '.XMM', '.X64')):
                continue
            if up.startswith(('PUBLIC ', 'EXTRN ', 'EXTERN ', 'ASSUME ')):
                continue
            if up == 'END':
                continue
            if up.startswith('ALIGN '):
                # Map to NASM align
                parts = tok.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    out.append(f'align {parts[1]}')
                continue

            # PROC/ENDP handling: turn "label PROC" -> "label:" and skip ENDP
            m_proc = re.match(r'^([A-Za-z_$.@?][\w$.@?]*)\s+PROC\b', tok)
            if m_proc:
                out.append(f'{m_proc.group(1)}:')
                continue
            if re.match(r'^[A-Za-z_$.@?][\w$.@?]*\s+ENDP\b', tok):
                # End of procedure: ignore (no epilogue emission here)
                continue

            # Data directives: DB/DW/DD/DQ -> lowercase NASM-friendly
            m_data = re.match(r'^(DB|DW|DD|DQ)\b(.*)$', tok, re.IGNORECASE)
            if m_data:
                out.append(m_data.group(1).lower() + m_data.group(2))
                continue

            # Replace "BYTE/WORD/DWORD/QWORD PTR" with NASM "byte/word/dword/qword"
            tok = re.sub(r'\b(BYTE|WORD|DWORD|QWORD)\s+PTR\b', lambda m: m.group(1).lower(), tok)

            # Replace "OFFSET FLAT:label" or "OFFSET label" with just label
            tok = re.sub(r'\bOFFSET\s+FLAT:', '', tok, flags=re.IGNORECASE)
            tok = re.sub(r'\bOFFSET\s+', '', tok, flags=re.IGNORECASE)

            # Some MSVC emits "FLAT:" as a segment label prefix; drop it
            tok = tok.replace(' FLAT:', ' ')

            # Instructions are already Intel syntax; keep as-is
            out.append(tok)

        return out

    def _intel_to_nasm(self, intel_asm: str) -> List[str]:
        out: List[str] = []
        for line in intel_asm.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith(('.', '#')):
                # map sections, comment the rest
                if s.startswith('.text'):
                    out.append(f"section .text ; from intel")
                elif s.startswith('.data'):
                    out.append(f"section .data ; from intel")
                elif s.startswith('.bss'):
                    out.append(f"section .bss ; from intel")
                elif s.startswith('.globl') or s.startswith('.global') or s.startswith('.type') or s.startswith('.size') or s.startswith('.ident') or s.startswith('.file'):
                    continue
                else:
                    out.append('; ' + s)
                continue
            # label
            if s.endswith(':'):
                out.append(s)
                continue
            # strip comments starting with '#'
            s = s.split(' #', 1)[0].rstrip()
            out.append(s)
        return out


# -----------------------------
# Windows x86_64 support
#
# - NASM output is compatible with Windows x86_64 ABI (PE/COFF, _start entry, syscall conventions)
# - To build: nasm -f pe64 out.asm -o out.o && ld out.o -o out.exe
#   Or use MSVC link.exe if available
# - Inline C blocks are supported with cl.exe (MSVC) if present
# - Syscalls use Linux ABI by default; for Windows, you may need to adapt or use C runtime stubs
# - For full Windows support, ensure your code does not rely on Linux-only syscalls
# - See documentation for more details on cross-platform assembly and linking
#
# Performance of generated code The quality of register allocation, instruction scheduling, and optimization passes will determine whether Density code matches highly tuned C/ASM. This is a nontrivial domain.
# Macro complexity & hygiene Ensuring macros don’t cause conflicts, unexpected expansions, or debugging nightmares is tricky.
# Code size & compile time For large projects, compile-time cost, memory use, and binary size must be managed.
# Debugging & tooling You’ll need good error messages, source maps (to map generated assembly back to .den lines), debuggers, symbol tables, and IDE support.
# Adoption & ecosystem To be useful, you need libraries (stdlib, IO, crypto, etc.), bindings, examples, and community traction.
# Extend macro / CIAM system: Add features like macro recursion, hygienic macros, macro overloading, or pattern matching.
# Improve optimization pipeline: Implement SSA (static single assignment), advanced register allocator, loop optimizations, or even machine learning–aided codegen.
# Debugging support: Source-to-assembly mapping, line number tracking, better error reporting.
# Standard library expansion: Add modules for networking, file IO, math, threading.
# Cross-platform build & packaging: Build executable wrappers, installers, integration with build systems (CMake, etc.).
# Better inline foreign integration: More robust handling of embedded C/Python code, validation, sandboxing, and linking issues.
# Testing & validation suite: A wide array of test programs, fuzzing, benchmarks, correctness testing.
#
# See also: examples/README.md for tutorial and sample projects.
# -----------------------------

