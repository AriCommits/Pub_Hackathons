import numpy as np

from actproto.training.alignment_loss import contrastive_loss_numpy, total_loss


def test_loss_drops_when_correct_prototype_moves_closer():
    z = np.array([[1.0, 0.0]], dtype=np.float32)
    far = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    close = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.array([0])
    loss_far = contrastive_loss_numpy(z, far, labels)
    loss_close = contrastive_loss_numpy(z, close, labels)
    assert loss_close < loss_far
    assert total_loss(1.0, 0.5, 0.1) == 1.05
