"""Dispatch to independent dataset implementations over common tensor contracts."""

from importlib import import_module


def _dataset_name(dataset):
    if dataset not in ("duke", "mimic"):
        raise ValueError("Dataset must be explicitly identified as duke or mimic")
    return dataset


def _module(dataset, stage):
    return import_module(f"datasets.{dataset}.{stage}")


def load_prepared(path):
    from ehrdib.utils import load_tensors

    payload = load_tensors(path)
    if "manifest" not in payload:
        raise ValueError("Expected converted data.pt; run prepare on the exported .pt first")
    dataset = _dataset_name(payload["manifest"]["dataset"])
    return _module(dataset, "preprocessing").restore_prepared(payload)


def prepared_digest(path):
    """Identify the prepared tensor data and its fitted preprocessing information."""
    from ehrdib.utils import file_digest

    return file_digest(path)


def prepare_dataset(
    source,
    destination,
    *,
    seed=None,
    dataset,
    expected_representation=None,
):
    # Conversion preserves the partitions fitted upstream for this experiment seed.
    dataset = _dataset_name(dataset)
    if expected_representation is not None:
        if dataset != "duke" or expected_representation not in ("binned", "continuous"):
            raise ValueError("Expected representation applies only to Duke visit exports")
        return _module(dataset, "preprocessing").prepare(
            source, destination, seed=seed, expected_representation=expected_representation
        )
    return _module(dataset, "preprocessing").prepare(source, destination, seed=seed)


def evaluate_prepared(data, generated, config, *, seed, evidence=None):
    dataset = _dataset_name(data.manifest.get("dataset"))
    module = _module(dataset, "evaluation")
    return module.evaluate(data, generated, config, seed=seed, evidence=evidence)


def evaluate_privacy(dataset, data_path, synthetic_path, *, seed, decoder="native", cache_dir=None):
    dataset = _dataset_name(dataset)
    return _module(dataset, "evaluation.privacy").privacy(
        data_path, synthetic_path, seed=seed, decoder=decoder, cache_dir=cache_dir
    )


def evaluate_run(run, data_path, *, methods=None, config=None):
    data = load_prepared(data_path)
    dataset = _dataset_name(data.manifest.get("dataset"))
    return _module(dataset, "evaluation").evaluate_run(
        run, data_path, methods=methods, config=config, _data=data
    )
