"""Download MobileSAM from its official repository. No Git installation needed."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'f706ad9c4eb7f219c00d9050e46328518ffb65d2'
WEIGHTS_SHA256 = '6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f'


def download(url, path):
    print(f'Download: {url}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'ChipAnnotator/1.0'})
    with urllib.request.urlopen(request, timeout=180) as response, open(path, 'wb') as out:
        while chunk := response.read(1024*1024):
            out.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--target', type=Path, help='Optional pip target for isolated verification')
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp)/'mobilesam.zip'
        download(f'https://codeload.github.com/ChaoningZhang/MobileSAM/zip/{REVISION}', source)
        if not args.download_only:
            command = [sys.executable, '-m', 'pip', 'install', '--no-deps', str(source)]
            if args.target:
                command.extend(['--target', str(args.target)])
            subprocess.run(command, check=True)
        target = args.models/'mobile_sam.pt'
        partial = args.models/'mobile_sam.pt.part'
        try:
            download(f'https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/{REVISION}/weights/mobile_sam.pt', partial)
            if partial.stat().st_size < 30_000_000:
                raise RuntimeError('Unexpected checkpoint size; file was not installed')
            digest = hashlib.sha256(partial.read_bytes()).hexdigest()
            if digest != WEIGHTS_SHA256:
                raise RuntimeError('Checkpoint SHA256 mismatch; file was not installed')
            partial.replace(target)
            (args.models/'provenance.json').write_text(json.dumps(dict(
                repository='https://github.com/ChaoningZhang/MobileSAM', revision=REVISION,
                sha256=digest), indent=2), encoding='utf-8')
            print(f'Model installed: {target}\nSHA256: {digest}', flush=True)
        finally:
            partial.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Installation failed: {exc}\nCheck internet access and retry. See README.md.', file=sys.stderr)
        sys.exit(1)
