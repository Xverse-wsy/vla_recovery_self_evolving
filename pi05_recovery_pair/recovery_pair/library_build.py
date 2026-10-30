"""Portable, CPU-only library construction over existing classified segments."""
import json
from pathlib import Path

from .config import Config
from .data import manifest_segmented, manifest_autonomous, read_jsonl, build_windows, QuantileAssets
from .features import sha256, read_start_image
from .libraries import BANKS, create_libraries, audit_libraries
from .schema import load_windows


def build_existing_libraries(sources_config, data_root, output):
    """Resolve data paths on this machine; never depend on A800-generated indexes."""
    config_path = Path(sources_config).expanduser().resolve()
    sources = json.loads(config_path.read_text())
    allowed = {'segmented_datasets', 'autonomous_datasets', 'manifests', 'norm_stats',
               'algorithm_config', 'expected_segment_counts', 'expected_image_shape'}
    if set(sources) - allowed:
        raise ValueError(f'Unknown source config fields: {sorted(set(sources) - allowed)}')
    root = Path(data_root).expanduser().resolve(strict=True)

    def data_path(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve(strict=True)

    datasets = {}
    for kind in ('segmented_datasets', 'autonomous_datasets', 'manifests'):
        values = sources.get(kind, [])
        if not isinstance(values, list) or not all(isinstance(p, str) for p in values):
            raise ValueError(f'{kind} must be a list of paths')
        datasets[kind] = [data_path(p) for p in values]
    if not any(datasets.values()):
        raise ValueError('Select at least one existing data source')
    norm = data_path(sources['norm_stats'])
    algorithm = Path(sources['algorithm_config']).expanduser()
    if not algorithm.is_absolute():
        algorithm = config_path.parent / algorithm
    config = Config.load(algorithm)
    config.validate()
    assets = QuantileAssets(norm)
    metadata = {config_path, algorithm.resolve(), norm, *datasets['manifests']}
    for source in datasets['segmented_datasets'] + datasets['autonomous_datasets']:
        metadata.update(p for p in (source / n for n in
                        ('meta/info.json', 'meta/provenance.jsonl', 'selection.json')) if p.is_file())
    before = {str(p): sha256(p) for p in sorted(metadata)}
    out = Path(output).expanduser().resolve()
    # Avoid nesting generated indexes/payloads inside an input dataset.
    for source in datasets['segmented_datasets'] + datasets['autonomous_datasets']:
        if out == source or source in out.parents:
            raise ValueError('Output must be outside the original dataset folders')
    out.mkdir(parents=True, exist_ok=False)
    status = out / 'status.json'
    try:
        status.write_text(json.dumps({'state': 'reading_existing_classification'}))
        manifest = []
        for source in datasets['segmented_datasets']:
            manifest.extend(manifest_segmented(source))
        for source in datasets['autonomous_datasets']:
            manifest.extend(manifest_autonomous(source))
        for source in datasets['manifests']:
            manifest.extend(read_jsonl(source))
        if not manifest:
            raise ValueError('No existing segments found')
        status.write_text(json.dumps({'state': 'building_segment_libraries'}))
        library_report = create_libraries(manifest, config, out / 'libraries')
        counts = {k: v['segments'] for k, v in library_report['banks'].items()}
        expected = sources.get('expected_segment_counts')
        if expected is not None and counts != expected:
            raise ValueError(f'Existing bank counts differ: actual={counts}, expected={expected}')
        entries = audit_libraries(out / 'libraries')
        status.write_text(json.dumps({'state': 'building_windows', 'segment_counts': counts}))
        records = build_windows(entries, assets, config, out / 'windows')
        image_shape = sources.get('expected_image_shape')
        for bank in BANKS:
            selected = load_windows(out / 'windows/banks' / f'{bank}.jsonl')
            if any(r.bank_class != bank for r in selected):
                raise RuntimeError('Incorrect existing class window index')
            if selected:
                for ref in selected[0].image_refs.values():
                    shape = read_start_image(ref).shape
                    if image_shape is not None and tuple(image_shape) != shape:
                        raise RuntimeError(f'Unexpected original image layout: {shape}')
        if {str(p): sha256(p) for p in sorted(metadata)} != before:
            raise RuntimeError('Source/config metadata changed during import')
        result = dict(state='complete', output=str(out), data_root=str(root),
            sources_config=str(config_path), segment_libraries=library_report,
            window_libraries=json.loads((out/'windows/window_libraries.json').read_text()),
            source_metadata_unchanged=True, original_program_boundaries_reused=True,
            training_started=False, cuda_used=False, source_metadata_sha256=before)
        (out/'report.json').write_text(json.dumps(result, indent=2))
        status.write_text(json.dumps({'state':'complete', 'segments':counts, 'windows':len(records)}))
        return result
    except Exception as exc:
        status.write_text(json.dumps({'state':'failed', 'error':str(exc)}))
        raise
