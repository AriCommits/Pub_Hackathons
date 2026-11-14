import numpy as np

from actproto.activations.similarity import cosine_scores
from actproto.stretch.steering.steer import contrast_direction, steer


def test_steering_increases_target_cosine():
    hidden = np.zeros(8, dtype=np.float32)
    hidden[0] = 1.0
    target = np.zeros(8, dtype=np.float32)
    target[1] = 1.0
    mean = np.zeros(8, dtype=np.float32)
    direction = contrast_direction(target, mean)
    prototypes = {"tool": target}
    scores = []
    for alpha in (0.0, 0.5, 2.0):
        scores.append(cosine_scores(steer(hidden, direction, alpha), prototypes)["tool"])
    assert scores[0] < scores[1] < scores[2]
