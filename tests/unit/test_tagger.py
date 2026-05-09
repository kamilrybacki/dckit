"""Tagger projection + classification."""

from __future__ import annotations

import numpy as np
import pytest

from dckit.codebook import Codebook
from dckit.tagger import Tagger


@pytest.mark.unit
def test_tagger_dim_mismatch_raises(small_codebook: Codebook) -> None:
    tagger = Tagger(small_codebook)
    with pytest.raises(ValueError, match="dim"):
        tagger.tag(np.zeros((2, 9), dtype=np.float32))


@pytest.mark.unit
def test_tagger_argmax_matches_class_mean() -> None:
    rng = np.random.default_rng(0)
    means = rng.standard_normal((3, 6)).astype(np.float32)
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    cb = Codebook(means, ("a", "b", "c"), -1.0, 1.0, {})
    tagger = Tagger(cb)
    # Each class mean should tag to its own index.
    tags = tagger.tag(means)
    np.testing.assert_array_equal(tags, [0, 1, 2])


@pytest.mark.unit
def test_tagger_normalised_in_range(small_codebook: Codebook) -> None:
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((10, 8)).astype(np.float32)
    norm = Tagger(small_codebook).project_normalised(vecs)
    assert (norm >= 0.0).all()
    assert (norm <= 1.0).all()


@pytest.mark.unit
def test_tagger_detailed(small_codebook: Codebook) -> None:
    rng = np.random.default_rng(2)
    vecs = rng.standard_normal((3, 8)).astype(np.float32)
    results = Tagger(small_codebook).tag_detailed(vecs)
    assert len(results) == 3
    for r in results:
        assert 0 <= r.argmax < small_codebook.k
        assert r.label == small_codebook.labels[r.argmax]
        assert 0.0 <= r.confidence <= 1.0


@pytest.mark.unit
def test_labels_for(small_codebook: Codebook) -> None:
    tagger = Tagger(small_codebook)
    out = tagger.labels_for(np.array([0, 2, 1]))
    assert out == [
        small_codebook.labels[0],
        small_codebook.labels[2],
        small_codebook.labels[1],
    ]
