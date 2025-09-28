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


