import tensorflow as tf


class IOU(tf.keras.metrics.Metric):
    def __init__(self, name: str = "iou", **kwargs: dict) -> None:
        super().__init__(name=name, **kwargs)
        self.iou = self.add_weight(name="iou", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, _: tf.Tensor) -> None:
        y_pred = tf.clip_by_value(y_pred, 0, 1)

        x_a = tf.math.maximum(y_true[:, 0], y_pred[:, 0])
        y_a = tf.math.maximum(y_true[:, 1], y_pred[:, 1])
        x_b = tf.math.minimum(y_true[:, 2], y_pred[:, 2])
        y_b = tf.math.minimum(y_true[:, 3], y_pred[:, 3])

        validator = tf.math.logical_and(
            y_pred[:, 0] < y_pred[:, 2],
            y_pred[:, 1] < y_pred[:, 3],
        )

        inter_area = (
            tf.math.maximum(0.0, x_b - x_a + 1e-6)
            * tf.math.maximum(0.0, y_b - y_a + 1e-6)
            * tf.cast(validator, tf.float32)
        )

        y_true_area = tf.math.abs(
            (y_true[:, 2] - y_true[:, 0] + 1e-6) * (y_true[:, 3] - y_true[:, 1] + 1e-6)
        )
        y_pred_area = tf.math.abs(
            (y_pred[:, 2] - y_pred[:, 0] + 1e-6) * (y_pred[:, 3] - y_pred[:, 1] + 1e-6)
        )

        iou = inter_area / tf.cast(y_true_area + y_pred_area - inter_area, tf.float32)

        self.iou.assign(tf.reduce_mean(iou))

    def result(self):
        return self.iou
