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
