import numpy as np

from actproto.activations.prototypes import fit_prototypes
from actproto.activations.similarity import cosine_scores, rank_scores
from actproto.activations.threshold import choose_threshold


def _clusters():
    rng = np.random.default_rng(0)
    calc = rng.normal(loc=[5, 0], scale=0.1, size=(20, 2))
    look = rng.normal(loc=[0, 5], scale=0.1, size=(20, 2))
    fin = rng.normal(loc=[-5, -5], scale=0.1, size=(20, 2))
    vectors = np.vstack([calc, look, fin]).astype(np.float32)
    labels = ["calculator"] * 20 + ["lookup_record"] * 20 + ["finish"] * 20
    return vectors, labels


def test_separated_clusters_classify_perfectly():
    vectors, labels = _clusters()
    train_v, train_y = vectors[:45:2], labels[:45:2]
    # use a proper train slice
    train_v = np.vstack([vectors[:10], vectors[20:30], vectors[40:50]])
    train_y = labels[:10] + labels[20:30] + labels[40:50]
    test_v = np.vstack([vectors[10:20], vectors[30:40], vectors[50:60]])
    test_y = labels[10:20] + labels[30:40] + labels[50:60]
    prototypes = fit_prototypes(train_v, train_y)
    pred = [rank_scores(cosine_scores(v, prototypes))["top1"] for v in test_v]
    assert pred == test_y


def test_threshold_uses_only_val_indices():
    vectors, labels = _clusters()
    val_v, val_y = vectors[:30], labels[:30]
    test_v, test_y = vectors[30:], labels[30:]
    prototypes = fit_prototypes(val_v, val_y)
    chosen = choose_threshold(val_v, val_y, prototypes, min_coverage=0.5)
    assert "min_score" in chosen
    assert chosen["coverage"] >= 0.5 - 1e-9
    # Sanity: applying the same cutoff to test should not require peeking at test during choose.
    assert test_v.shape[0] == len(test_y)
    assert chosen["n"] == float(len(val_y))
