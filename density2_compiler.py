import re
from typing import List, Tuple, Union

# -----------------------------
# Token Definitions
# -----------------------------
TOKEN_SPECIFICATION = [
    ('CIAM_START', r"'''"),            # start of CIAM
    ('CIAM_END', r",,,"),
    ('INLINE_START', r"#(asm|c|python)?"),  # start of inline block
    ('INLINE_END', r"#end\1"),         # will be replaced dynamically
    ('COMMENT', r'//[^\n]*'),          # single-line comment
    ('MCOMMENT', r'/\*.*?\*/'),        # multi-line comment
    ('STRING', r'"(?:\\.|[^"\\])*"'),  # string literal
    ('IDENT', r'[A-Za-z_][A-Za-z0-9_]*'), # identifiers and keywords
    ('LBRACE', r'\{'),
    ('RBRACE', r'\}'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('COLON', r':'),
    ('SEMICOLON', r';'),
    ('NEWLINE', r'\n'),
    ('SKIP', r'[ \t]+'),               # spaces/tabs
    ('MISMATCH', r'.'),                # any other character
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
    // Density 2 Hello World Program

    Main() {
        Print: ("Hello, World!");
    }
    '''
    tokens = tokenize(code)
    print("TOKENS:")
    for t in tokens:
        print(t)
    parser = Parser(tokens)
    ast = parser.parse()
    print("\nAST:")
    print(ast)
