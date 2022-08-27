import tensorflow as tf

from metrics import IOU


def _check_float(true, pred, epsilon=1e-4):
    return true - epsilon <= pred <= true + epsilon


def test_iou():
    y_true = tf.constant(
        [
            [0.25, 0.1, 0.75, 0.5],
        ]
    )
    y_pred = tf.constant(
        [
            [0.25, 0.4, 0.75, 0.2],
        ]
    )

    iou = IOU()
    iou.update_state(y_true, y_pred)
    assert _check_float(0, iou.result())
