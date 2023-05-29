import numpy as np
import tensorflow as tf

from datetime import datetime
from numpy import ndarray
from typing import Callable
from abc import abstractmethod as abstract

from sim_bug_tools.structs import Point, Domain
from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults
from sim_bug_tools.rng.lds.sequences import Sequence, SobolSequence, RandomSequence
from sim_bug_tools.simulation.simulation_core import Scorable, Graded


class ProbilisticSphere(Graded):
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

    def score(self, p: Point) -> ndarray:
        "Returns between 0 (far away) and 1 (center of) envelope"
        dist = self.loc.distance_to(p)

        return np.array(1 / np.e ** (self._c * dist**2))

    def classify_score(self, score: ndarray) -> bool:
        return np.linalg.norm(score) > self.lmda

    def gradient(self, p: Point) -> np.ndarray:
        s = p - self.loc
        s /= np.linalg.norm(s)

        return s * self._dscore(p)

    def get_input_dims(self):
        return len(self.loc)

    def get_score_dims(self):
        return 1

    def generate_random_target(self):
        v = np.random.rand(self.get_input_dims())
        v = self.loc + Point(self.radius * v / np.linalg.norm(v) * np.random.rand(1))
        return v

    def generate_random_nontarget(self):
        v = np.random.rand(self.get_input_dims())
        v = self.loc + Point(
            self.radius * v / np.linalg.norm(v) * (1 + np.random.rand(1))
        )
        return v

    def boundary_err(self, b: Point) -> float:
        "Negative error is inside the boundary, positive is outside"
        return self.loc.distance_to(b) - self.radius

    def _dscore(self, p: Point) -> float:
        return -self._c * self.score(p) * self.loc.distance_to(p)


class ANNParams(ExperimentParams):
    def __init__(
        self,
        name: str,
        envelope: Scorable,
        seq: Sequence,
        training_size=2**10,
        batch_size=2**7,
        n_epochs=500,
        optimizer="adam",
        desc: str = None,
    ):
        """
        Args:
            envelope (Scorable): The envelope to model
            seq (Sequence): The sequence used to sample the envelope with for
                constructing training set
            n_samples (_type_, optional): The number of @seq samples from
                @envelope to train the model. Defaults to 2**10.
            batch_size (_type_, optional): How many samples per batch. Defaults to 2**7.
            n_epochs (int, optional): How many epochs. Defaults to 500.
            optimizer (str, optional): Tensorflow optimizer to use in training
                the model. Defaults to "adam".
        """
        super().__init__(name, desc)

        self.envelope = envelope
        self.seq = seq
        self.training_size = training_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.model_name = name


class ANNResults(ExperimentResults[ANNParams]):
    def __init__(
        self,
        params: ANNParams,
        model_path: str,
        scored_data: list[tuple[ndarray, ndarray]],
    ):
        super().__init__(params)
        self.model_path = model_path
        self.scored_data = scored_data


class ANNExperiment(Experiment[ANNParams, ANNResults]):
    def bare_model(self, input_dims: int, output_dims: int) -> tf.keras.Sequential:
        "Creates an ANN according to Wang et al."
        tanh = tf.keras.activations.tanh
        relu = tf.keras.activations.relu

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dims,), name="parameters"),
                tf.keras.layers.Dense(32, activation=tanh),
                tf.keras.layers.Dense(64, activation=tanh),
                tf.keras.layers.Dense(32, activation=tanh),
                tf.keras.layers.Dense(output_dims, activation=tanh),
            ]
        )

        model.compile(
            optimizer="adam",  # not sure if ADAM was used for training the ANN itself...
            loss="mse",
            metrics=[tf.keras.metrics.mean_squared_error],
        )

        return model

    def get_dataset_partitions(
        self,
        ds: list[tuple[Point, ndarray]],
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        shuffle=True,
    ):
        "Break data up into train, validation, and testing sets"
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

    def train_model(
        self,
        model: tf.keras.Sequential,
        scored_points: list[tuple[Point, ndarray]],
        batch: int,
        epochs: int,
    ):
        train, val, test = self.get_dataset_partitions(np.array(scored_points))

        X_train, Y_train = zip(*train)

        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)

        log_dir = ".tmp/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        model.fit(
            X_train,
            Y_train,
            batch_size=batch,
            epochs=epochs,
            callbacks=[tensorboard_callback],
        )

    def _shotgun_sampling(
        self, params: ANNParams, subdomain: Domain, num: int, desired_cls=True
    ):
        ndims = params.envelope.get_input_dims()
        domain = Domain.normalized(ndims)

        data: list[tuple[Point, ndarray]] = []
        for i in range(num):
            p = params.seq.get_sample(1).points[0]
            p = Domain.translate_point_domains(p, domain, subdomain)

            score = params.envelope.score(p)

            if params.envelope.classify_score(score) == desired_cls:
                data.append((p, score))

        return data

    def _sample_nearby(
        self, params: ANNParams, center: Point, size: float, num: int, desired_cls=True
    ):
        ndims = params.envelope.get_input_dims()
        domain = Domain.normalized(ndims)
        origin = center - Point([size / 2] * ndims)

        subdomain = Domain.from_dimensions([size] * ndims, origin).clip(domain)

        return self._shotgun_sampling(params, subdomain, num, desired_cls)

    def _sample_within(
        self, params: ANNParams, cluster: list[Point], num: int, desired_cls=True
    ):
        subdomain = Domain.from_point_cloud(cluster)
        return self._shotgun_sampling(params, subdomain, num, desired_cls)

    def generate_data_targetdistr(
        self, params: ANNParams, target_percentage=0.5, min_score=None
    ):
        """
        Generate inputs labeled with the results from the envelope's scoring.
        Returns list[tuple[input (Point), score (NDArray)]]

        The distribution between Target and Nontarget is uniform.
        """
        # ts = [
        #     (params.envelope.generate_random_target(), True)
        #     for i in range(int(params.training_size * target_percentage))
        # ]
        # nonts = [
        #     (params.envelope.generate_random_nontarget(), False)
        #     for i in range(int(params.training_size * (1 - target_percentage)))
        # ]
        n_true = params.training_size * target_percentage
        n_false = params.training_size * (1 - target_percentage)

        false_data: list[tuple[Point, ndarray]] = []
        true_data: list[tuple[Point, ndarray]] = []
        true_cnt = 0
        false_cnt = 0
        while (
            true_cnt + false_cnt < (params.training_size // 2) * 2
        ):  # round nearest even
            for p in params.seq.get_sample(params.training_size).points:
                score = params.envelope.score(p)

                if params.envelope.classify_score(score) and true_cnt < n_true:
                    true_cnt += 1
                    true_data.append((p, score))
                    # subdata = self._shotgun_sampling(params, p, 0.01, 50)
                    x = 10

                    for i in range(50):
                        rescore = params.envelope.score(p)
                        reclass = params.envelope.classify_score(rescore)
                        pass

                    x = 10

                    # if true_cnt % 10:
                    #     subdata = self._sample_within(
                    #         params, [d[0] for d in true_data], 50
                    #     )
                    #     if len(subdata) + true_cnt > n_true:
                    #         true_data.extend(subdata[: n_true - true_cnt])
                    #         true_cnt = n_true
                    #     else:
                    #         true_data.extend(subdata)
                    #         true_cnt += len(subdata)
                elif (
                    (min_score is None or score >= min_score)
                    and not params.envelope.classify_score(score)
                    and false_cnt < n_false
                ):
                    false_cnt += 1
                    false_data.append((p, score))
                elif true_cnt + false_cnt >= (params.training_size // 2) * 2:
                    break

        return true_data + false_data  # ts + nonts

    def generate_data_spatialuniform(self, params: ANNParams):
        """
        Generate inputs labeled with the results from the envelope's scoring.
        Returns list[tuple[input (Point), score (NDArray)]]

        The distribution between Target and Nontarget is uniform.
        """
        return [
            (p, params.envelope.score(p))
            for p in params.seq.get_sample(params.training_size).points
        ]

    def experiment(self, params: ANNParams) -> ANNResults:
        model = self.bare_model(
            params.envelope.get_input_dims(), params.envelope.get_score_dims()
        )

        data = self.generate_data_targetdistr(params)

        self.train_model(model, data, params.batch_size, params.n_epochs)

        path = self.get_path(params.model_name)
        model.save(path)

        return ANNResults(params, path, data)

    @classmethod
    def get_path(cls, model_name: str) -> str:
        """
        Generates a path that follows a consistent formatting for caching models
        """
        return f"{cls.CACHE_FOLDER}/{cls.get_name()}/{model_name}"

    @staticmethod
    def predict_gradient(p: Point, model: tf.keras.Sequential) -> ndarray:
        inp = tf.Variable(np.array([p]), dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = model(inp)

        return np.array(tape.gradient(preds, inp)).squeeze()

    @classmethod
    def grad_accs(
        cls,
        model: tf.keras.Sequential,
        points: list[ndarray],
        envelope: Graded,
    ) -> list[float]:
        norm = lambda v: v / np.linalg.norm(v)

        def angle_between(u, v):
            u, v = norm(u), norm(v)
            return np.arccos(np.clip(np.dot(u, v), -1, 1.0))

        gs = [cls.predict_gradient(p, model) for p in points]
        return list(
            map(
                lambda p, v: 1 - 2 * angle_between(v, envelope.gradient(p)) / np.pi,
                points,
                gs,
            )
        )

    @staticmethod
    def calc_err(
        model: tf.keras.Sequential,
        scored_data: list[tuple[ndarray, ndarray]],
        envelope: Scorable,
    ):
        "Mean Absolute Percent Error"
        errs = [
            abs((score - (pred := model(p[None]))) / pred) for p, score in scored_data
        ]
        return sum(errs) / len(errs)

    @staticmethod
    def class_acc(
        model: tf.keras.Sequential,
        scored_data: list[tuple[ndarray, ndarray]],
        envelope: Scorable,
    ):
        """
        tn, tp
        fn, fp
        """
        table = np.zeros((2, 2))
        truth = np.identity(2)
        tp, tn, fp, fn = 0, 0, 0, 0

        for p, score in scored_data:
            true_cls = envelope.classify_score(score)
            pred_cls = envelope.classify_score(model(p[None]))

            if true_cls:
                if pred_cls:
                    tp += 1
                else:
                    fn += 1
            else:
                if not pred_cls:
                    tn += 1
                else:
                    fp += 1

        return tp, tn, fp, fn


def _ann_param_name(n_samples: int):
    return f"ANN-psphere-{n_samples}"


def test_ann():
    print("Tensorflow is required to run this test...")
    ndims = 3
    domain = Domain.normalized(ndims)

    ann_exp = ANNExperiment()

    # Probabilistic sphere with radius .4, where the prob-density that defines
    # the boundary is 0.25
    envelope = ProbilisticSphere(Point([0.5 for i in range(ndims)]), 0.4, 0.25)
    seq = RandomSequence(domain, [str(i) for i in range(ndims)])

    # how many samples from the envelope to train the ANN on
    ann_training_size = 1000

    ann_name = _ann_param_name(ann_training_size)
    ann_params = ANNParams(
        ann_name,
        envelope,
        seq,
        ann_training_size,
        ann_training_size // 10,
        n_epochs=100,
    )

    # ann_results = ann_exp.index.previous_result
    ann_results = ann_exp.run(ann_params)

    model = tf.keras.models.load_model(ann_results.model_path)
    data = ann_results.scored_data

    # determine True positive/negative and false positive/negative for all
    # scored data within the training set

    for p, score in data:
        print(
            score,
            np.array(model(p.array[None])).squeeze(),
            envelope.classify_score(score),
            envelope.classify_score(np.array(model(p.array[None])).squeeze()),
        )

    truth_table = ANNExperiment.class_acc(model, data, envelope)
    acc = lambda tab: sum(tab[:2]) / (tab)
    tp, tn, fp, fn = truth_table
    print(truth_table, f"\nAcc = {acc * 100}%")


def test_previous_ann():
    print("Tensorflow is required to run this test...")

    ann_exp = ANNExperiment()

    ann_results = ann_exp.index.previous_result

    model = tf.keras.models.load_model(ann_results.model_path)
    data = ann_results.scored_data
    envelope = ann_results.params.envelope

    # determine True positive/negative and false positive/negative for all
    # scored data within the training set

    # for p, score in data:
    #     print(
    #         score,
    #         np.array(model(p.array[None])).squeeze(),
    #         envelope.classify_score(score),
    #         envelope.classify_score(np.array(model(p.array[None])).squeeze()),
    #     )

    test_size = 1000
    target_p = 0.5
    target_points = [
        (envelope.generate_random_target(), True)
        for i in range(int(test_size * target_p))
    ]
    nontarget_points = [
        (envelope.generate_random_nontarget(), False)
        for i in range(int(test_size * (1 - target_p)))
    ]
    test_data = target_points + nontarget_points

    acc = lambda tab: sum(tab[:2]) / sum(tab)

    train_truth_table = ANNExperiment.class_acc(model, data, envelope)
    test_truth_table = ANNExperiment.class_acc(model, test_data, envelope)

    train_g_accs = ANNExperiment.grad_accs(
        model, list(map(lambda x: x[0], data)), envelope
    )
    train_g_avg_acc = sum(train_g_accs) / len(train_g_accs)
    test_g_accs = ANNExperiment.grad_accs(
        model, list(map(lambda x: x[0], test_data)), envelope
    )
    test_g_avg_acc = sum(test_g_accs) / len(test_g_accs)

    print(
        "Training data accuracy:",
        train_truth_table,
        f"\nAcc = {acc(train_truth_table) * 100}%",
        f"Gradient acc = {train_g_avg_acc}",
    )
    print(
        "Test data accuracy:",
        test_truth_table,
        f"\nAcc = {acc(test_truth_table) * 100}%",
        f"Gradient acc = {test_g_avg_acc}",
    )


if __name__ == "__main__":
    test_previous_ann()
