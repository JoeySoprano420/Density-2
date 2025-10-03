[Setup]
AppName=Density 2 Compiler
AppVersion=2.0.0
DefaultDirName={pf}\Density2
DefaultGroupName=Density 2
UninstallDisplayIcon={app}\bin\density2c.bat
OutputDir=.
OutputBaseFilename=Density2-x64-setup
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
Compression=lzma
SolidCompression=yes

[Files]
Source: "density2_compiler.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "density2c.bat"; DestDir: "{app}\bin"; Flags: ignoreversion

[Icons]
Name: "{group}\Density 2 Compiler"; Filename: "{app}\bin\density2c.bat"
Name: "{commondesktop}\Density 2 Compiler"; Filename: "{app}\bin\density2c.bat"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\bin\density2c.bat"; Parameters: "--version"; Description: "Verify installation"

[Registry]
; Add bin folder to system PATH
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; \
    ValueType: expandsz; ValueName: "Path"; \
    ValueData: "{olddata};{app}\bin"; Flags: preservestringtype

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

; --- Candy Wrapper additions ---
[Files]
Source: "bin\candywrapper.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\candy.bat";       DestDir: "{app}\bin"; Flags: ignoreversion
; If you freeze candywrapper with PyInstaller, ship CandyWrapper.exe instead:
; Source: "bin\CandyWrapper.exe"; DestDir: "{app}\bin"; Flags: ignoreversion

[Tasks]
Name: "addpath_candy"; Description: "Add Candy Wrapper to PATH"; Flags: checkedonce

[Registry]
; PATH extension (system-wide)
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment";
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}\bin"; Tasks: addpath_candy; Flags: preservestringtype

; Right-click context menu: "Run with Candy Wrapper"
Root: HKCU; Subkey: "Software\Classes\*\shell\Run with Candy Wrapper"; ValueType: string; ValueData: "Run with Candy Wrapper"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\*\shell\Run with Candy Wrapper\command";
    ValueType: string; ValueData: """{app}\bin\candy.bat"" ""%1"""; Flags: uninsdeletekey

; Optional: file associations for .asm / .obj / .den (double-click)
Root: HKCU; Subkey: "Software\Classes\.asm\shell\open\command";
    ValueType: string; ValueData: """{app}\bin\candy.bat"" ""%1"""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.obj\shell\open\command";
    ValueType: string; ValueData: """{app}\bin\candy.bat"" ""%1"""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.den\shell\open\command";
    ValueType: string; ValueData: """{app}\bin\candy.bat"" ""%1"""; Flags: uninsdeletevalue

[Icons]
Name: "{group}\Candy Wrapper"; Filename: "{app}\bin\candy.bat"
Name: "{group}\Candy Wrapper GUI"; Filename: "{app}\bin\candy_gui.py"; Parameters: ""; WorkingDir: "{app}\bin"; Tasks:


