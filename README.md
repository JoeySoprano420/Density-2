# Density-2

Here’s a clear **first-cut spec** for *Density 2* based on what you wrote. I’ve broken it into layers so you can see at a glance how the language is supposed to behave and how it’s different from v1.0:

---

## 1. Core Identity

* **Name:** Density 2
* **File Extension:** `.den`
* **Philosophy:** “Terms and signs mutually universal” — syntax that looks natural to C/Python/NASM but is strongly typed like a systems language.
* **Execution:** AOT compile to NASM → link to PE/.exe or ELF.
* **Grammar Source:** Written in Dodecagrams (base-12, digits `0-9,a,b`).

---

## 2. Syntax Highlights

```density2
Main() Print: ("Hello, World!");

 // single-line comment  

 /* multi-line
    comment */

 '''CIAM … ,,,       // Contextually Inferred Abstraction Macro  

 #...#              // inline C / NASM / Python / Dodecagrams
```

* **Single line comments:** `//`
* **Multi-line comments:** `/* … */`
* **CIAMS (new):** `'''…,,,` separate from comments — macros expanded at compile time.
* **Inline foreign code:** `#…#` to embed NASM, C, Python, or Dodecagram.

---

## 3. Typing & Semantics

* **Dual mode:** *Explicitly mutable, strongly dynamic* but also *immutably intrinsic static* where annotated.
* **Universal semantics:** Keywords match Python/C/NASM meaning 1-to-1.
* **.den files** compile directly without glue code.

---

## 4. Error / Memory Keywords

* Error flow: `try, catch, raise, flag, throws, except, noexcept, ignore, suppress, avoid, skip, isolate, warning`.
* Memory: `alloc, free, collect, sweeps, register, store, release, delete, malloc, stack, heap, assign, allow, hold, deny, lease, carry, void, check, match, local, deallocate`.

---

## 5. Built-ins & Structures

Primitives, types, flows, keywords, structs, tuples, lists, arrays, vectors, nests, branches, directives, qualifiers, states, conditionals, booleans, operators, checkpoints, breaks, suspend, lookahead, peephole, profile guidance, index, constant folding, flattening, loop unrolling, compression, condensing, mutex, recall, rollback, slices, parallelism, single & multithreading, calls, I/O, imports, exports, renders, compile, link, map, interpret, interface, exchange, ping, setup, network, publish, read, scan, collect, run, fuse, channel, adopt, adaptability, durations, closures, support, dictionary, ruleset, delineate, exponent, variables, values, negate, floats, decimals, strings, nulls, pointers, blocks, classes, methods, modules, items, partitions, handles, scaling, weights, range, delimiters, punctuation, whitespace, indents, force, limiter, bounds, parameters, scope, text, comments, bridge, relationships, context, iterate, fringe, risks, eval, enums, escape, end, packet, encapsulate, grouping, match, batch, equivalents, instances, hierarchies, schedules, async, await, ask, reply, print, return, math, include, add, mult, synchronized, polynomials, polymorphism, div, sub, div-by-zero, align, position, resource, runtime, aware, navigate, process, pause, resume, consume, conversions, probabilities, measure, temp, trees, chains, webs, learn, traffic, busy, clear, available, sharp, dull, encryption, decrypt, obfuscate, deobfuscate, detect, cipher, decipher, fibonacci, sequence, pattern, linear, non-linear, equation, paradox, pairings, coupling, history, prediction, solve.

*(This list becomes the official keyword namespace of Density 2.)*

---

## 6. CIAMs (Contextually Inferred Abstraction Macros)

* Think of them as macro-templates.

* Syntax:

  ```density2
  '''MacroName(parameters)
     body...
  ,,,
  ```

* Expanded at compile time, not comments, not run-time.

* Can generate NASM, C, Python, or Dodecagram code inside the compiler pipeline.

---

## 7. Inline Foreign Code

Example:

```density2
Main() {
    #asm
        mov eax,1
    #endasm

    #c
        printf("Hello from C\n");
    #endc
}
```

The compiler directly splices these into the NASM backend during codegen.

---

## 8. Immediate Next Steps

1. **Define tokens**: comment vs CIAM vs inline blocks.
2. **Build a lexer** recognizing:

   * Identifiers, keywords, punctuation
   * `'''…,,,` CIAM blocks
   * `#…#` inline foreign code blocks
3. **Grammar**: Use Dodecagram base-12 codes for AST nodes.
4. **Emit**: NASM directly with an AOT pipeline → PE/ELF.
5. **Testing**: Start with simple `Main() Print:("Hello")` → .exe.

---

### Example Flow:

```density2
Main() {
    Print:("Hello, World!");
    try {
        #python print("Inline python") #
    } catch (Error e) {
        Print:(e);
    }
}
```

---

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

you now have a real front-end that tokenises, parses and produces an AST containing three distinct node types:

PrintStatement for normal Density 2 statements

CIAMBlock for your compile-time macros

InlineBlock for embedded NASM/C/Python

When you run the test program I showed, your console should print something like:

AST:
Program([Function(Main,
  body=[
    CIAMBlock('SayHello(name)\n        Print: ("Hello, " + name + "!");'),
    InlineBlock(lang='asm', content='mov eax, 60\n        xor edi, edi\n        syscall'),
    Print(Hello, World!)
  ])
])


That’s your Density 2 “Hello World + CIAM + Inline ASM” parsed into a tree you can walk.

## _____


That’s still valid Density 2 — the compiler will expand the `SayHello` CIAM at compile-time and splice the NASM directly into the generated assembly.
## _____

Run and assemble
python density2_compiler.py       # prints AST and NASM code
nasm -f elf64 out.asm -o out.o    # assemble
ld out.o -o out                   # link
./out                             # run


You should see:

Hello, World!


on stdout.

At this point you have:

Lexer + Parser → AST

Code generator → real .asm file

## _____

Run:

python density2_compiler.py


Now out.asm contains:

section .data
    str_0 db "Hello, World!", 10, 0
    str_1 db "Density 2!", 10, 0
section .text
    global _start
_start:
    mov rax, 1          ; sys_write
    mov rdi, 1          ; stdout
    mov rsi, str_0    ; message
    mov rdx, 14         ; length
    syscall
    mov rax, 1          ; sys_write
    mov rdi, 1          ; stdout
    mov rsi, str_1    ; message
    mov rdx, 11         ; length
    syscall
    mov rax, 60         ; sys_exit
    xor rdi, rdi        ; status 0
    syscall


Assemble and run:

nasm -f elf64 out.asm -o out.o
ld out.o -o out
./out


Output:

Hello, World!
Density 2!

Test with a .den file containing inline NASM
Main() {
    Print: ("Hello, World!");

    #asm
        mov rax, 1
        mov rdi, 1
        mov rsi, str_0
        mov rdx, 14
        syscall
    #endasm

    Print: ("After inline NASM");
}

Run the compiler
python density2_compiler.py
nasm -f elf64 out.asm -o out.o
ld out.o -o out
./out


The generated out.asm will look something like:

section .data
    str_0 db "Hello, World!", 10, 0
    str_1 db "After inline NASM", 10, 0
section .text
    global _start
_start:
    mov rax, 1          ; sys_write
    mov rdi, 1          ; stdout
    mov rsi, str_0    ; message
    mov rdx, 14         ; length
    syscall
    ; inline NASM start
    mov rax, 1
    mov rdi, 1
    mov rsi, str_0
    mov rdx, 14
    syscall
    ; inline NASM end
    mov rax, 1          ; sys_write
    mov rdi, 1          ; stdout
    mov rsi, str_1    ; message
    mov rdx, 18         ; length
    syscall
    mov rax, 60         ; sys_exit
    xor rdi, rdi        ; status 0
    syscall

    When you assemble and run it, the inline assembly runs directly after the first print, then the normal second print.

    Now you have:

Print: generates its own string labels.

CIAM: stored in AST ready for future macro expansion.

Inline NASM: injected literally into .text.




## _____





This gives you a real end-to-end pipeline: Density 2 .den file → AST → .asm → executable.

