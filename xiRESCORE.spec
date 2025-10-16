# -*- mode: python ; coding: utf-8 -*-
import xirescore

a = Analysis(
    ['xirescore\\__main__.py'],
    pathex=[],
    binaries=[],
    datas=[('xirescore/assets', 'xirescore/assets')],
    hiddenimports=[
        'pyarrow.vendored.version',
        'asyncio.base_events',
        'asyncio.events',
        'typing_extensions',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
splash = Splash(
    'xirescore\\assets\\xirescore_logo.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    #splash,
    [],
    name=f'xiRESCORE_{xirescore.__version__}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['xirescore\\assets\\xirescore_logo.ico'],
)
