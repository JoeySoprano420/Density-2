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
