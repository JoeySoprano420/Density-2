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
