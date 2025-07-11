# -*- mode: python ; coding: utf-8 -*-


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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='xiRESCORE',
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
