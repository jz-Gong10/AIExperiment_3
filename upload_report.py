"""Upload the report-material ZIP to the owner's GitHub Release for download."""
import argparse
from datetime import datetime, timezone
from getpass import getpass
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

REPOSITORY = 'jz-Gong10/AIExperiment_3'
ROOT = Path(__file__).resolve().parent


def request_json(url, token, method='GET', payload=None, data=None):
    headers = {
        'Authorization': 'Bearer ' + token,
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'AIExperiment3-report-download',
    }
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    elif data is not None:
        headers['Content-Type'] = 'application/zip'
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', nargs='?', type=Path, default=ROOT / 'report_materials.zip')
    args = parser.parse_args()
    path = args.file.resolve()
    if not path.is_file():
        parser.error('Archive not found: ' + str(path))
    if not zipfile.is_zipfile(path):
        parser.error('The selected file is not a ZIP archive')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            parser.error('ZIP archive integrity check failed')
    print('File:', path)
    print('Size: {:.2f} MiB'.format(path.stat().st_size / 1024 / 1024))
    print('Destination:', 'https://github.com/' + REPOSITORY + '/releases')
    print('This public repository makes the uploaded attachment public.')
    token = getpass('Paste GitHub token (hidden), then press Enter: ').strip()
    if not token:
        parser.error('No token entered')
    base = 'https://api.github.com/repos/' + REPOSITORY
    try:
        try:
            release = request_json(base + '/releases/tags/report-materials', token)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            release = request_json(base + '/releases', token, method='POST', payload={
                'tag_name': 'report-materials',
                'target_commitish': 'main',
                'name': 'Server report materials',
                'body': 'Actual application output frames and training records collected from the DCU container.',
                'draft': False,
                'prerelease': True,
            })
        name = 'report_materials_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.zip'
        upload_url = release['upload_url'].split('{', 1)[0] + '?name=' + quote(name)
        print('Uploading...')
        asset = request_json(upload_url, token, method='POST', data=path.read_bytes())
        print('\nUpload completed.')
        print('DOWNLOAD URL:', asset['browser_download_url'])
        print('RELEASE PAGE:', release['html_url'])
        return 0
    except HTTPError as exc:
        print('GitHub HTTP error:', exc.code, file=sys.stderr)
        if exc.code in (401, 403, 404):
            print('Check that the token belongs to the owner, selects this repository, and grants Contents: Read and write.', file=sys.stderr)
        else:
            print('Upload did not complete. The source ZIP is still available on the server.', file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError) as exc:
        print('Network or file error:', type(exc).__name__, file=sys.stderr)
        print('The source ZIP is still available on the server.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
