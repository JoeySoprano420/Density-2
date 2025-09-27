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
    pass


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

    def generate(self) -> str:
        # Expand macros (compile-time)
        if isinstance(self.ast, Program):
            # macros already expanded in compile_density2
            pass

        # Emit
        self.text_lines = []
        self.data_lines = []
        self.string_table = {}
        self.label_counter = 0

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

        for stmt in func.body:
            if isinstance(stmt, PrintStatement):
                self._emit_print(stmt.text)
            elif isinstance(stmt, InlineBlock):
                self._emit_inline(stmt)
            elif isinstance(stmt, CIAMBlock):
                # Already handled by expand_macros; leave a comment in case any remain
                self.text_lines.append(f'    ; CIAMBlock ignored (should be expanded): {stmt.name}')
            elif isinstance(stmt, MacroCall):
                # Should be expanded away
                self.text_lines.append(f'    ; MacroCall ignored (should be expanded): {stmt.name}')
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
        self.text_lines.append(f'    mov rax, 1          ; sys_write')
        self.text_lines.append(f'    mov rdi, 1          ; stdout')
        self.text_lines.append(f'    mov rsi, {label}    ; message')
        self.text_lines.append(f'    mov rdx, {length}         ; length (bytes)')
        self.text_lines.append('    syscall')

    def _emit_exit(self):
        self.text_lines.append('    mov rax, 60         ; sys_exit')
        self.text_lines.append('    xor rdi, rdi        ; status 0')
        self.text_lines.append('    syscall')

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

        # Very restricted globals/locals
        globals_dict = {
            '__builtins__': {
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'print': print,  # debug if needed
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
                return translated

            # Fallback to comments if translation failed
            commented = []
            commented.append('; [begin compiled C assembly]')
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

    # --- AT&T -> NASM best-effort translation helpers ---

    def _translate_att_to_nasm(self, att_asm: str) -> List[str]:
        out: List[str] = []
        for line in att_asm.splitlines():
            s = line.strip()
            if not s:
                continue

            # Ignore sectioning and many metadata directives
            if s.startswith('.'):
                if s.startswith(('.globl', '.global', '.text', '.data', '.bss', '.rodata', '.section',
                                 '.type', '.size', '.file', '.ident', '.cfi', '.p2align', '.intel_syntax', '.att_syntax')):
                    continue
                out.append(f'; {s}')
                continue

            # Preserve labels
            if s.endswith(':'):
                out.append(s)
                continue

            # Remove trailing comments
            s = s.split('\t#', 1)[0].split(' #', 1)[0].strip()
            if not s:
                continue

            parts = s.split(None, 1)
            op = parts[0]
            rest = parts[1] if len(parts) > 1 else ''
            op_n = re.sub(r'(q|l|w|b)$', '', op)  # drop size suffix

            ops = [o.strip() for o in rest.split(',')] if rest else []
            ops = [self._att_operand_to_nasm(o) for o in ops]

            # Reverse operand order for two-operand instructions
            if len(ops) == 2:
                ops = [ops[1], ops[0]]

            if ops:
                out.append(f'{op_n} ' + ', '.join(ops))
            else:
                out.append(op_n)

        return out

    def _att_operand_to_nasm(self, o: str) -> str:
        o = o.strip()
        # Immediate: $val -> val
        if o.startswith('$'):
            return o[1:]
        # Registers: %rax -> rax
        o = re.sub(r'%([a-zA-Z][a-zA-Z0-9]*)', r'\1', o)

        # RIP-relative like foo(%rip) or disp(%reg,idx,scale)
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

        # Bare symbol or already simple
        return o


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
