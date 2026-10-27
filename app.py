import sys

if __name__ == '__main__':
    try:
        from chipannotator.gui import main
    except ImportError as exc:
        print(f'依存ライブラリがありません: {exc}\nsetup.bat を実行してください。')
        sys.exit(1)
    sys.exit(main())
