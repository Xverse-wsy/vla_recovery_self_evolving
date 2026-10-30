"""Frozen, supplied TorchScript visual encoder. No model download or random encoder."""
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=2)
def _image_column(path, column):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=[column])[column].to_pylist()


def read_start_image(ref):
    """No flip: existing exported RGB already has deployment orientation."""
    if isinstance(ref, str):
        with Image.open(ref) as im:
            return np.asarray(im.convert('RGB')).copy()
    value = _image_column(ref['parquet'], ref['column'])[int(ref['row'])]
    if isinstance(value, dict):
        if value.get('bytes') is not None:
            with Image.open(io.BytesIO(value['bytes'])) as im:
                return np.asarray(im.convert('RGB')).copy()
        with Image.open(Path(ref['parquet']).parent / value['path']) as im:
            return np.asarray(im.convert('RGB')).copy()
    return np.asarray(value, dtype=np.uint8)


def extract_features(records, encoder_path, preprocessing_path, cache_dir, device='cpu'):
    """Encoder contract: preprocessed [B,3,H,W] -> finite [B,F]."""
    spec = json.loads(Path(preprocessing_path).read_text())
    encoder_sha, prep_sha = sha256(encoder_path), sha256(preprocessing_path)
    encoder = torch.jit.load(str(encoder_path), map_location=device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(spec['mean'], device=device)[:, None, None]
    std = torch.tensor(spec['std'], device=device)[:, None, None]
    if (std <= 0).any() or spec.get('resize_mode') != 'bilinear_stretch':
        raise ValueError('Specify positive std and explicit bilinear_stretch preprocessing')
    for r in records:
        views = {}
        for name in ('front', 'wrist'):
            ref = r.image_refs[name]
            rgb = read_start_image(ref)
            rgb_sha = hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()
            key = hashlib.sha256((encoder_sha + prep_sha + rgb_sha).encode()).hexdigest()
            dest = cache / (key + '.npy')
            if dest.exists():
                feature = np.load(dest, allow_pickle=False)
            else:
                im = Image.fromarray(rgb).resize(tuple(spec['size_wh']), Image.Resampling.BILINEAR)
                x = torch.as_tensor(np.asarray(im).copy(), device=device).permute(2, 0, 1).float() / 255
                with torch.no_grad():
                    out = encoder(((x - mean) / std)[None])
                if not isinstance(out, torch.Tensor) or out.ndim != 2 or out.shape[0] != 1:
                    raise ValueError('Frozen encoder must return [B,F]')
                feature = out[0].float().cpu().numpy()
                if not np.isfinite(feature).all() or not np.linalg.norm(feature):
                    raise ValueError('Invalid visual feature')
                np.save(dest, feature)
            views[name] = feature.tolist()
        r.visual_features = dict(encoder_sha256=encoder_sha, calibration_id=prep_sha, views=views)
    return records
