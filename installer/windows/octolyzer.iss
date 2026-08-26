; Inno Setup script for the OCTolyzer desktop launcher.
;
; Wraps the Nuitka-built OCTolyzerGUI.exe (see build/build_gui.py) in a
; per-machine Windows installer: Start Menu shortcut, Add/Remove Programs
; entry, and a proper uninstaller. Requires admin elevation (a UAC prompt at
; install time) since it installs to Program Files.
;
; Build with (from the repository root, after `python -m build.build_gui`):
;   ISCC installer\windows\octolyzer.iss /DAppVersion=1.2.3
;
; /DAppVersion defaults to 0.0.0-dev when not supplied (e.g. local test builds).
; The Nuitka artifact is expected at dist\gui\OCTolyzerGUI.exe.

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

[Setup]
; Fixed permanently -- generated once via `python -c "import uuid; print(uuid.uuid4())"`.
; Changing this would make every future release install side-by-side instead
; of upgrading in place.
AppId={{EDBB7125-003F-46E2-B490-62AF7BF9BC27}
AppName=OCTolyzer
AppVersion={#AppVersion}
AppPublisher=OCTolyzer
DefaultDirName={autopf}\OCTolyzer
DefaultGroupName=OCTolyzer
DisableProgramGroupPage=yes
PrivilegesRequired=admin
OutputDir=..\..\dist\gui\installer
OutputBaseFilename=OCTolyzerSetup
SetupIconFile=..\..\gui\assets\icon.ico
UninstallDisplayIcon={app}\OCTolyzerGUI.exe
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\..\dist\gui\OCTolyzerGUI.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\OCTolyzer"; Filename: "{app}\OCTolyzerGUI.exe"
Name: "{group}\Uninstall OCTolyzer"; Filename: "{uninstallexe}"
Name: "{autodesktop}\OCTolyzer"; Filename: "{app}\OCTolyzerGUI.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\OCTolyzerGUI.exe"; Description: "Launch OCTolyzer"; Flags: nowait postinstall skipifsilent
