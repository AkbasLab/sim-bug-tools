"""
Derived from Wang et al. "Safety Performance Boundary Identification of Highly
Automated Vehicles: A Surrogate Model-Based Gradient Descent Searching Approach" 

Notes:
    model.predict() requires numpy array. More specifically, np.array([[]]), not np.array([])
"""
import os
import numpy as np
import tensorflow as tf

from typing import Generator, Any, Callable
from collections.abc import Sized
from abc import ABC, abstractmethod as abstract

# from sim_bug_tools.experiment import Experiment, ExperimentParams
from sim_bug_tools.structs import Point, Domain
from sim_bug_tools.rng.lds.sequences import Sequence, SobolSequence

SAVE_LOCATION = "./.models/"
MODEL_NAME = "wang-gd-model"

# tf.Variable(np.random.normal(size=))


def train_exmpl():
    mnist = tf.keras.datasets.mnist
    tanh = tf.keras.activations.tanh
    relu = tf.keras.activations.relu
    input_dims = 3
    output_dims = 1
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dims,), name="parameters"),
            tf.keras.layers.Dense(32, activation=tanh),
            tf.keras.layers.Dense(64, activation=tanh),
            tf.keras.layers.Dense(32, activation=tanh),
            tf.keras.layers.Dense(output_dims, activation=relu),
        ]
    )

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print("Type, shape", type(x_train[0]), len(x_train[0]),
    # len(x_train[0][0]))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.mse,
        metrics=["accuracy"],
    )

    x_train = np.array([[1, 2, 3]])

    pred = model(x_train[:1]).numpy()
    print(pred)


def get_bare_model(input_dims: int, output_dims: int):
    "Creates an ANN according to Wang et al."
    tanh = tf.keras.activations.tanh
    relu = tf.keras.activations.relu

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dims,), name="parameters"),
            tf.keras.layers.Dense(32, activation=tanh),
            tf.keras.layers.Dense(64, activation=tanh),
            tf.keras.layers.Dense(32, activation=tanh),
            tf.keras.layers.Dense(output_dims, activation=relu),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.mse,
        metrics=["accuracy"],
    )

    return model


def get_dataset_partitions(
    ds: np.ndarray,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    shuffle=True,
):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        np.random.shuffle(ds)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    test_size = int(ds_size - val_split * ds_size)

    train_ds = ds[:train_size]
    val_ds = ds[train_size : train_size + val_size]
    test_ds = ds[-test_size:]

    return train_ds, val_ds, test_ds


def train_gd(
    classified_points: list[tuple[Point, bool]],
    model: tf.keras.Sequential,
    name: str = MODEL_NAME,
):
    "(As side effect,) trains and saves @model and returns testig results"
    BATCH_SIZE = 256
    EPOCHS = 500

    train, val, test = get_dataset_partitions(np.array(classified_points))

    X_train, Y_train = zip(*train)
    X_test, Y_test = zip(*test)

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    pred = model(X_test[:1])
    print(pred)

    model.save(f"{SAVE_LOCATION}/{name}")

    results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE * 2)

    pred = model(X_test[:1])
    print(pred)
    return results


def _infer_input_output_size(classified_point: tuple[Point, Any]):
    "Returns input/output sizes"
    input_size = len(classified_point[0])
    output_size = (
        len(classified_point[1]) if isinstance(classified_point[1], Sized) else 1
    )
    return input_size, output_size


def lazily_get_model(
    classified_points: list[tuple[Point, bool]], name: str = MODEL_NAME
):
    if os.path.isdir(f"{SAVE_LOCATION}/{name}"):
        return tf.keras.models.load_model(f"{SAVE_LOCATION}/{name}")
    else:
        model = get_bare_model(*_infer_input_output_size(classified_points[0]))
        train_gd(classified_points, model, name)
        return model


def predict_gradient_at(model: tf.keras.Sequential, points: tuple[Point]):

    # input_shape = model.layers[0].input_shape
    # inp = tf.Variable(np.random.normal(size=input_shape), dtype=tf.float32)
    inp = tf.Variable(np.array(points), dtype=tf.float32)

    with tf.GradientTape() as tape:
        preds = model(inp)

    return np.array(tape.gradient(preds, inp)).squeeze()


## envelope
class Scorable:
    @abstract
    def score(self, p: Point) -> float:
        raise NotImplementedError()

    @abstract
    def classify(self, p: Point) -> bool:
        raise NotImplementedError()

    def __contains__(self, other):
        return self.classify(other)

    @abstract
    def gradient_at(self, p: Point) -> np.ndarray:
        raise NotImplementedError()

    @property
    def ndims(self):
        return self._ndims

    @ndims.setter
    def ndims(self, ndims: int):
        self._ndims = ndims


class SolidSphere(Scorable):
    def __init__(self, loc: Point, radius: float):
        self.loc = loc
        self.radius = radius
        self.ndims = len(loc)

    def score(self, p: Point) -> float:
        return 1 if self.loc.distance_to(p) < self.radius else 0


class ProbilisticSphere(Scorable):
    def __init__(self, loc: Point, radius: float, lmbda: float):
        """
        Probability density is formed from the base function f(x) = e^-(x^2),
        such that f(radius) = lmbda and is centered around the origin with a max
        of 1.

        Args:
            loc (Point): Where the sphere is located
            radius (float): The radius of the sphere
            lmbda (float): The density of the sphere at its radius
        """
        self.loc = loc
        self.radius = radius
        self.lmda = lmbda
        self.ndims = len(loc)

        self._c = 1 / radius**2 * np.log(1 / lmbda)

    def score(self, p: Point) -> float:
        "Returns between 0 (far away) and 1 (center of) envelope"
        dist = self.loc.distance_to(p)

        return 1 / np.e ** (self._c * dist**2)

    def classify(self, p: Point) -> bool:
        return self.score(p) < self.lmda

    def gradient_at(self, p: Point) -> np.ndarray:
        s = p - self.loc
        s /= np.linalg.norm(s)

        return s * self._dscore(p)

    def _dscore(self, p: Point) -> float:
        return -self._c * self.score(p) * self.loc.distance_to(p)


def generate_score_data(envelope: Scorable, n: int, seq: Sequence = None):
    if seq is None:
        ndims = envelope.ndims
        domain = Domain.normalized(ndims)
        seq = SobolSequence(domain, [str(i) for i in range(ndims)])

    scores = []

    for p in seq.get_sample(n)._points:
        scores.append((p, envelope.score(p)))

    return scores


## Exploring using gradient


class GradientExplorer:
    def __init__(
        self,
        model: tf.keras.Sequential,
        classifier: Callable[[Point], bool],
        h: Callable[[Point], np.ndarray],
        domain: Domain,
        max_steps: int = 1000,
    ):
        self.model = model
        self.classifier = classifier
        self.h = h
        self.max_steps = max_steps
        self.domain = domain

    def find_boundary_from(self, init_p: Point, init_cls: bool):

        i = 0
        prev = init_p
        prev_cls = init_cls
        cur = init_p
        cur_cls = prev_cls

        points: tuple[Point, bool] = []

        # Loop if max_steps not exceeded and boundary has not been found
        while i < self.max_steps and cur_cls == prev_cls and cur in self.domain:
            prev = cur
            # Make a step according to gradient descent solution
            cur += self.h(cur) if cur_cls else -self.h(cur)

            prev_cls = cur_cls
            cur_cls = self.classifier(cur)
            points.append((cur, cur_cls))
            i += 1

            yield cur, cur_cls

        if cur_cls != prev_cls:
            b = cur if cur_cls else prev
        else:
            b = None

        self._previous_result = b, points

    @property
    def result(self):
        return self._previous_result


## Tests
def test_gradient():
    ndims = 3
    domain = Domain.normalized(ndims)

    num_samples = 2**13

    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)

    data = generate_score_data(envelope, num_samples)

    model = lazily_get_model(data)

    points = np.array(
        [
            (0.5, 0.5, 0.5),
            (0.9, 0.5, 0.5),
            (1.0, 0.5, 0.5),
        ]
    )

    # model.summary()

    pr = np.array(predict_gradient_at(model, points))
    tr = [envelope.gradient_at(Point(p)) for p in points]

    # pr = model(points)
    # tr = [envelope.score(Point(p)) for p in points]

    # exp_grad = predict_gradient_at(model)

    print(f"Predicted: {pr},\nTrue: {tr}")


def test_model_accuracy():
    from sim_bug_tools.graphics import Grapher

    ndims = 3
    domain = Domain.normalized(ndims)

    num_samples = 2**15

    g = Grapher(ndims == 3, domain)

    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)

    data = generate_score_data(envelope, num_samples)

    model = get_bare_model(*_infer_input_output_size(data[0]))
    model.summary()

    results = train_gd(data, model)

    print(results)


def test_lazy():
    ndims = 3

    num_samples = 2**15

    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)

    data = generate_score_data(envelope, num_samples)

    # model = lazily_get_model(data)
    # model.summary()
    model = get_bare_model(*_infer_input_output_size(data[0]))
    model.summary()
    results = train_gd(data, model)
    print(model.predict(np.array([0.5, 0.5, 0.5])))


def test_compare_pred_theo():
    ndims = 3
    domain = Domain.normalized(ndims)

    num_samples = 2**15

    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)

    data = generate_score_data(envelope, num_samples)

    model = get_bare_model(*_infer_input_output_size(data[0]))
    train_gd(data, model)
    # model = lazily_get_model(data)

    points = np.random.rand(300, 3)

    import pandas as pd

    differences = pd.DataFrame(
        map(lambda p: float(envelope.score(Point(p)) - model(p[None])), points)
    )

    # avg_dif = sum(differences) / len(points)

    print(differences.describe())

    model.summary()

    pr = model(points)
    tr = [envelope.score(Point(p)) for p in points]

    # print(f"Predicted: {pr}, True: {tr}")


def train_new_model():
    ndims = 3
    num_samples = 2**15

    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)
    data = generate_score_data(envelope, num_samples)

    model = get_bare_model(*_infer_input_output_size(data[0]))

    train_gd(data, model)


def test_exp():
    from sim_bug_tools.graphics import Grapher
    import matplotlib.pyplot as plt

    ndims = 3
    domain = Domain.normalized(ndims)
    num_samples = 2**15

    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)
    data = generate_score_data(envelope, num_samples)
    model = lazily_get_model(data)

    classifier = envelope.classify

    alpha = 0.1

    def movement(p: Point):
        g = predict_gradient_at(model, [p])
        return alpha * g

    exp = GradientExplorer(model, classifier, movement, domain)

    boundary: list[Point] = []
    all_points = []
    paths: list[list[tuple[Point, bool]]] = []

    g = Grapher(ndims == 3, domain)

    for p, score in data[: 2**7]:
        for cur, cur_cls in exp.find_boundary_from(p, classifier(p)):
            pass

        b, points = exp.result

        if b is not None:
            boundary.append(b)
            paths.append(points)

    for path in paths:
        if len(path) < 2:
            continue

        verts = list(zip(*path))[0]
        g.draw_path(verts, "-", color="grey")

        g.plot_point(path[0][0], color=("yellow" if path[0][1] else "blue"))
        g.plot_point(
            path[-1][0], color=("yellow" if path[-1][1] else "blue"), marker="x"
        )

        if len(path) > 2:
            g.plot_all_points(verts[1:-1], color="grey")

        g.plot_point(verts[-2], color="red")  # b-point
    plt.show()

    print(
        boundary,
        len(boundary),
    )


if __name__ == "__main__":

    test_exp()
