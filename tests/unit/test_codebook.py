"""Codebook construction, validation, io."""

from __future__ import annotations

import numpy as np
import pytest

from dckit.codebook import Codebook


@pytest.mark.unit
def test_codebook_validates_shape() -> None:
    with pytest.raises(ValueError, match="2-D"):
        Codebook(
            embeddings=np.zeros((4,)),
            labels=("a",) * 4,
            min_val=-1.0,
            max_val=1.0,
            metadata={},
        )


@pytest.mark.unit
def test_codebook_validates_labels_match() -> None:
    with pytest.raises(ValueError, match="!="):
        Codebook(
            embeddings=np.zeros((4, 8), dtype=np.float32),
            labels=("a", "b"),
            min_val=-1.0,
            max_val=1.0,
            metadata={},
        )


@pytest.mark.unit
def test_codebook_validates_calibration() -> None:
    with pytest.raises(ValueError, match="max_val"):
        Codebook(
            embeddings=np.zeros((1, 4), dtype=np.float32),
            labels=("only",),
            min_val=1.0,
            max_val=0.5,
            metadata={},
        )


@pytest.mark.unit
def test_save_load_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((3, 5)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cb = Codebook(
        embeddings=emb,
        labels=("00 — A", "10 — B", "20 — C"),
        min_val=-0.5,
        max_val=0.9,
        metadata={"model": "stub", "seed": 0},
    )
    path = tmp_path / "cb.npz"
    cb.save(path)
    assert path.exists()
    assert path.with_suffix(".json").exists()
    loaded = Codebook.load(path)
    np.testing.assert_allclose(loaded.embeddings, cb.embeddings, rtol=1e-6)
    assert loaded.labels == cb.labels
    assert loaded.min_val == pytest.approx(cb.min_val)
    assert loaded.max_val == pytest.approx(cb.max_val)
    assert loaded.metadata == cb.metadata


@pytest.mark.unit
def test_content_hash_stable() -> None:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((2, 4)).astype(np.float32)
    cb1 = Codebook(emb, ("a", "b"), -1.0, 1.0, {"x": 1})
    cb2 = Codebook(emb.copy(), ("a", "b"), -1.0, 1.0, {"y": 2})
    assert cb1.content_hash() == cb2.content_hash()


@pytest.mark.unit
def test_from_class_means_rejects_empty_clusters() -> None:
    vectors = np.eye(4, dtype=np.float32)
    assignments = np.array([0, 0, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="empty cluster"):
        Codebook.from_class_means(
            vectors, assignments, labels=["a", "b", "c"]
        )


@pytest.mark.unit
def test_from_class_means_normalises() -> None:
    vectors = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 5.0], [0.0, 7.0]], dtype=np.float32)
    assignments = np.array([0, 0, 1, 1], dtype=np.int64)
    cb = Codebook.from_class_means(vectors, assignments, labels=["A", "B"])
    norms = np.linalg.norm(cb.embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)
