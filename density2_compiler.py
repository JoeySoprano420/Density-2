import re
from typing import List, Tuple, Union

# -----------------------------
# Token Definitions
# -----------------------------
TOKEN_SPECIFICATION = [
    ('CIAM', r"'''(.*?),,,"),
    ('INLINE_ASM', r"#asm(.*?)#endasm"),
    ('INLINE_C', r"#c(.*?)#endc"),
    ('INLINE_PY', r"#python(.*?)#endpython"),
    ('COMMENT', r'//[^\n]*'),
    ('MCOMMENT', r'/\*.*?\*/'),
    ('STRING', r'"(?:\\.|[^"\\])*"'),
    ('IDENT', r'[A-Za-z_][A-Za-z0-9_]*'),
    ('LBRACE', r'\{'),
    ('RBRACE', r'\}'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('COLON', r':'),
    ('SEMICOLON', r';'),
    ('NEWLINE', r'\n'),
    ('SKIP', r'[ \t]+'),
    ('MISMATCH', r'.'),
]


token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATION)

class Token:
    def __init__(self, type_, value, line, column):
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
    tokens = []
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
        elif kind == 'SKIP' or kind == 'COMMENT' or kind == 'MCOMMENT':
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

class PrintStatement(ASTNode):
    def __init__(self, text: str):
        self.text = text
    def __repr__(self):
        return f"Print({self.text})"

# -----------------------------
# Parser (very simple)
# -----------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: str) -> Token:
        tok = self.peek()
        if not tok or tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok

    def parse(self) -> Program:
        functions = []
        while self.peek() is not None:
            functions.append(self.parse_function())
        return Program(functions)

    def parse_function(self) -> Function:
        # Expect IDENT (function name)
        name_tok = self.consume('IDENT')
        self.consume('LPAREN')
        self.consume('RPAREN')
        self.consume('LBRACE')
        body = self.parse_statements()
        self.consume('RBRACE')
        return Function(name_tok.value, body)

    def parse_statements(self) -> List[ASTNode]:
        stmts = []
        while self.peek() and self.peek().type != 'RBRACE':
            tok = self.peek()
            if tok.type == 'IDENT' and tok.value == 'Print':
                stmts.append(self.parse_print())
            elif tok.type == 'CIAM':
                ciam_tok = self.consume('CIAM')
                content = ciam_tok.value[3:-3] if ciam_tok.value.startswith("'''") else ciam_tok.value
                stmts.append(CIAMBlock(content.strip()))
            elif tok.type.startswith('INLINE_'):
                lang = tok.type.split('_', 1)[1].lower()  # asm/c/py
                inline_tok = self.consume(tok.type)
                # strip off #lang and #endlang markers:
                content = re.sub(r'^#\w+', '', inline_tok.value, flags=re.DOTALL)
                content = re.sub(r'#end\w+$', '', content, flags=re.DOTALL)
                stmts.append(InlineBlock(lang, content.strip()))
            else:
                self.pos += 1
        return stmts
    
    def parse_statements(self) -> List[ASTNode]:
        stmts = []
        while self.peek() and self.peek().type != 'RBRACE':
            if self.peek().type == 'IDENT' and self.peek().value == 'Print':
                stmts.append(self.parse_print())
            else:
                # For now just consume unknown tokens to progress
                self.pos += 1
        return stmts

    def parse_print(self) -> PrintStatement:
        self.consume('IDENT')  # 'Print'
        self.consume('COLON')
        self.consume('LPAREN')
        string_tok = self.consume('STRING')
        self.consume('RPAREN')
        self.consume('SEMICOLON')
        return PrintStatement(eval(string_tok.value))

# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    code = '''
    Main() {
        Print: ("Hello, World!");
    }
    '''
    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse()
    print("AST:")
    print(ast)

    gen = CodeGenerator(ast)
    asm = gen.generate()
    print("\nGenerated NASM:\n")
    print(asm)

    # write to file
    with open('out.asm', 'w') as f:
        f.write(asm)
    print("\nWritten to out.asm")


class CIAMBlock(ASTNode):
    def __init__(self, content: str):
        self.content = content
    def __repr__(self):
        return f"CIAMBlock({self.content!r})"

class InlineBlock(ASTNode):
    def __init__(self, lang: str, content: str):
        self.lang = lang
        self.content = content
    def __repr__(self):
        return f"InlineBlock(lang={self.lang!r}, content={self.content!r})"

# -----------------------------
# NASM Emitter
# -----------------------------
class CodeGenerator:
    def __init__(self, ast: Program):
        self.ast = ast
        self.lines = []

    def generate(self) -> str:
        self.lines = []
        self._emit_header()
        for func in self.ast.functions:
            self._emit_function(func)
        return '\n'.join(self.lines)

    def _emit_header(self):
        self.lines.append('section .data')
        self.lines.append('    hello_msg db "Hello, World!", 10, 0')
        self.lines.append('section .text')
        self.lines.append('    global _start')

    def _emit_function(self, func: Function):
        if func.name == 'Main':
            self.lines.append('_start:')
        else:
            self.lines.append(f'{func.name}:')
        for stmt in func.body:
            if isinstance(stmt, PrintStatement):
                self._emit_print(stmt.text)
            elif isinstance(stmt, CIAMBlock):
                # naïve: just put comment for now; expansion can be added later
                self.lines.append(f'; CIAMBlock ignored: {stmt.content}')
            elif isinstance(stmt, InlineBlock):
                self.lines.append(f'; inline {stmt.lang} start')
                self.lines.append(stmt.content)
                self.lines.append(f'; inline {stmt.lang} end')
        if func.name == 'Main':
            self._emit_exit()

    def _emit_print(self, text: str):
        # For now always print our static hello_msg for demonstration
        self.lines.append('    mov rax, 1          ; sys_write')
        self.lines.append('    mov rdi, 1          ; stdout')
        self.lines.append('    mov rsi, hello_msg  ; message')
        self.lines.append('    mov rdx, 14         ; length')
        self.lines.append('    syscall')

    def _emit_exit(self):
        self.lines.append('    mov rax, 60         ; sys_exit')
        self.lines.append('    xor rdi, rdi        ; status 0')
        self.lines.append('    syscall')

