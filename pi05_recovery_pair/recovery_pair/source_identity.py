"""Verify installed RLinf copies as well as editable checkouts against pinned Git."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse, unquote


def verify_source_tree(package_root, repository, expected_commit):
    package_root, repository = Path(package_root).resolve(), Path(repository).resolve()
    revision = subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
    if revision != expected_commit:
        raise RuntimeError(f'Expected RLinf {expected_commit}; found {revision}')
    paths = sorted((package_root/'models/embodiment/openpi').rglob('*.py'))
    paths += [package_root/'config.py']
    if len(paths)<2:
        raise RuntimeError('RLinf package is missing its OpenPI source tree')
    digests = {}
    for path in paths:
        relative = str(Path('rlinf') / path.relative_to(package_root))
        expected = subprocess.check_output(['git','-C',str(repository),'show',f'{expected_commit}:{relative}'])
        installed = path.read_bytes()
        if installed != expected:
            raise RuntimeError(f'Installed RLinf differs from pinned source: {relative}')
        digests[relative] = hashlib.sha256(installed).hexdigest()
    digest = hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()
    return dict(reference_commit=revision,reference_checkout=str(repository),
        installed_source_root=str(package_root),files_checked=len(digests),source_sha256=digest)


def verify_installed_rlinf(module, expected_commit):
    package_root = Path(module.__file__).resolve().parent
    repository = next((p for p in package_root.parents if (p/'.git').exists()),None)
    if repository is None:
        # pip install /path/to/RLinf installs a copy; direct_url retains provenance.
        dist = importlib.metadata.distribution('rlinf')
        direct = json.loads(dist.read_text('direct_url.json') or '{}')
        url = urlparse(direct.get('url',''))
        if url.scheme == 'file':
            candidate = Path(unquote(url.path))
            if (candidate/'.git').exists():
                repository = candidate
    if repository is None:
        raise RuntimeError('Need pinned RLinf checkout to verify installed sources; see README')
    return verify_source_tree(package_root,repository,expected_commit)
