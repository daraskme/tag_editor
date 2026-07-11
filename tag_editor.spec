# -*- mode: python ; coding: utf-8 -*-
# Onedir, windowed build. Do NOT switch to --onefile or enable UPX for this
# app: the PyQt6 + onnxruntime-directml payload is large enough that a
# self-extracting onefile adds a 10-30s cold start and is a common
# antivirus false-positive trigger; UPX corrupts/flags Qt and VC++ DLLs.
# The 1.9 GB AI model is never bundled -- it downloads at runtime into
# %USERPROFILE%\.cache\ai_tagger.

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='tag_editor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='tag_editor',
)
