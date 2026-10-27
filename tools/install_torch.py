"""Resumable, checksum-verified downloads from the official PyTorch origin.

Using small HTTP ranges avoids large-response stalls on some networks.
Only Windows CPython 3.12 x64 wheels are selected; no arbitrary package URLs.
"""
import argparse
import concurrent.futures
import hashlib
import html
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 8*1024*1024


def response(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60)


def wheel(name, version, flavor):
    filename = f'{name}-{version}+{flavor}-cp312-cp312-win_amd64.whl'
    with response(f'https://download.pytorch.org/whl/{flavor}/{name}/') as source:
        index = source.read().decode('utf-8')
    links = [html.unescape(s) for s in re.findall(r'href="([^"]+)"', index)]
    link = next((s for s in links if urllib.parse.unquote(s).split('#')[0].endswith('/'+filename)), None)
    if link is None or '#sha256=' not in link:
        raise RuntimeError(f'Official wheel/hash not found: {filename}')
    expected = link.split('#sha256=')[1]
    url = f'https://download.pytorch.org/whl/{flavor}/'+urllib.parse.quote(filename)
    cache = Path(os.environ.get('CHIPANNOTATOR_CACHE', str(ROOT/'.downloadcache')))
    folder = cache/filename
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/filename
    if target.exists():
        with target.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() == expected:
                print(f'Using verified cache: {filename}',flush=True)
                return target
    with response(url+'?size=1', {'Range':'bytes=0-0'}) as source:
        content_range = source.headers.get('Content-Range','')
        if source.status != 206 or not re.fullmatch(r'bytes 0-0/\d+',content_range):
            raise RuntimeError('Server did not honor range requests; retry or use manual pip setup in README')
        total = int(content_range.split('/')[1])

    def fetch(offset):
        end = min(offset+BLOCK,total)-1
        part = folder/f'{offset:012d}.part'
        if part.exists() and part.stat().st_size == end-offset+1:
            return part
        for attempt in range(3):
            try:
                with response(url+f'?chunk={offset}', {'Range':f'bytes={offset}-{end}'}) as source:
                    if source.status != 206 or source.headers.get('Content-Range') != f'bytes {offset}-{end}/{total}':
                        raise RuntimeError('Invalid range response')
                    with part.open('wb') as out:
                        while data := source.read(1024*1024):
                            out.write(data)
                if part.stat().st_size != end-offset+1:
                    raise RuntimeError('Incomplete download')
                return part
            except Exception:
                if attempt == 2:
                    raise

    offsets = list(range(0,total,BLOCK))
    done = 0
    print(f'Downloading {filename}: {total/1024**2:.0f} MiB',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(fetch,o) for o in offsets]
        for i,future in enumerate(concurrent.futures.as_completed(pending),1):
            done += future.result().stat().st_size
            if i % 8 == 0 or i == len(offsets):
                print(f'  {100*done/total:.1f}% ({done/1024**2:.0f} MiB)',flush=True)
    digest = hashlib.sha256()
    with target.open('wb') as out:
        for offset in offsets:
            with (folder/f'{offset:012d}.part').open('rb') as part:
                while data := part.read(1024*1024):
                    digest.update(data)
                    out.write(data)
    if digest.hexdigest() != expected:
        target.unlink(missing_ok=True)
        raise RuntimeError('SHA256 mismatch. Delete this wheel cache folder and retry.')
    print('SHA256 verified:', expected,flush=True)
    for offset in offsets:
        (folder/f'{offset:012d}.part').unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flavor', choices=['cu128','cpu'],default='cu128')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'win32' or sys.version_info[:2] != (3,12) or struct.calcsize('P') != 8:
        raise RuntimeError('Use Windows Python 3.12 x64 for this installer')
    files = [wheel('torch','2.7.1',args.flavor),wheel('torchvision','0.22.1',args.flavor)]
    if not args.download_only:
        subprocess.run([sys.executable,'-m','pip','install',*[str(p) for p in files]],check=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'PyTorch setup failed: {exc}\nRetry to resume cached download. See README.md.',file=sys.stderr)
        sys.exit(1)
