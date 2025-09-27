# Density-2

Here’s what a minimal **“Hello, World!”** looks like in the Density 2 syntax you just defined, using only the rules you gave (single-line comments with `//`, multi-line comments with `/*…*/`, CIAMs with `'''…,,,`, and optional inline code with `#…#`):

```density2
// Density 2 Hello World Program

Main() Print: ("Hello, World!");
```

Or, with a more block-like style:

```density2
// Density 2 Hello World Program

Main() {
    Print: ("Hello, World!");
}
```

That’s the canonical form for your `.den` files.

If you want to show off more of the language in one file, you can include a CIAM and an inline NASM block too:

```density2
// Density 2 Hello World with CIAM and inline NASM

Main() {
    '''SayHello(name)
        Print: ("Hello, " + name + "!");
    ,,,

    SayHello("World");

    #asm
        ; Inline NASM comment
        mov eax, 60      ; syscall exit
        xor edi, edi     ; status 0
        syscall
    #endasm
}
```


Save as density2_compiler.py, run:

python density2_compiler.py

Test with a richer .den file
Main() {
    '''SayHello(name)
        Print: ("Hello, " + name + "!");
    ,,,

    #asm
        mov eax, 60
        xor edi, edi
        syscall
    #endasm

    Print: ("Hello, World!");
}


Run python density2_compiler.py again — the AST now includes CIAMBlock and InlineBlock nodes:

AST:
Program([Function(Main, body=[CIAMBlock('SayHello(name)\n        Print: ("Hello, " + name + "!");'), InlineBlock(lang='asm', content='mov eax, 60\n        xor edi, edi\n        syscall'), Print(Hello, World!)])])


This is a real parser for Density 2 Hello World + CIAM + inline code with no placeholders or pass. From here you can walk the AST and generate NASM or C accordingly.


That’s still valid Density 2 — the compiler will expand the `SayHello` CIAM at compile-time and splice the NASM directly into the generated assembly.

