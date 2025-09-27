import os
import re
import shutil
import subprocess
import tempfile
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

    print("\nTo assemble and link (Linux x86-64), run:")
    print("  nasm -f elf64 out.asm -o out.o")
    print("  ld out.o -o out")

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


