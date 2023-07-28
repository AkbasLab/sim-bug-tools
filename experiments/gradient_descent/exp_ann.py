import numpy as np
import tensorflow as tf

from datetime import datetime
from numpy import ndarray
import random

from sim_bug_tools.structs import Point, Domain
from sim_bug_tools.experiment import Experiment, ExperimentParams, ExperimentResults
from sim_bug_tools.rng.lds.sequences import Sequence, SobolSequence, RandomSequence
from sim_bug_tools.simulation.simulation_core import Scorable, Graded


class ProbilisticSphere(Graded):
    def __init__(self, loc: Point, radius: float, lmbda: float, height: float = 1):
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
        self.height = height

        self._c = 1 / radius**2 * np.log(height / lmbda)

    @property
    def const(self) -> float:
        return self._c

    def score(self, p: Point) -> ndarray:
        "Returns between 0 (far away) and 1 (center of) envelope"
        dist = self.loc.distance_to(p)

        return self.height * np.array(1 / np.e ** (self._c * dist**2))

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
        return abs(self.loc.distance_to(b) - self.radius)

    def _dscore(self, p: Point) -> float:
        return -self._c * self.score(p) * self.loc.distance_to(p)


class ProbilisticSphereCluster(Graded):
    def __init__(
        self,
        num_points_per_point: int,
        depth: int,
        r0: float,
        p0: Point,
        lmbda: float = 0.25,
        height: float = 100,
        min_dist_b_perc: float = -0.1,
        max_dist_b_perc: float = 0.1,
        min_rad_perc: float = 1.25,
        max_rad_perc: float = 0.25,
        domain: Domain = None,
        seed=333,
        harsh_boundary=False,
    ):
        """
        dist_b perc:
            Describes the percentage of the radius for one sphere
            extending from another, how far it is from the boundary.
        rad perc:
            Bounds for new sphere's radius as a percentage of its
            parent: r2 = r1 * rad_perc
        """
        assert (
            min_dist_b_perc < max_dist_b_perc
        ), "Minimum distance from boundary must be less than maximum distance from boundary!"
        # assert (
        #     min_rad_perc < max_rad_perc
        # ), "Minimum radius percentage must be less than maximum radius percentage!"
        assert (
            min_rad_perc >= 0
        ), "Minimum radius percentage must be positive! (this is just how it works.)"

        random.seed(seed)
        np.random.seed(seed)

        self._ndims = len(p0)
        self._lmbda = lmbda
        self._min_dist_b_perc = min_dist_b_perc
        self._max_dist_b_perc = max_dist_b_perc
        self._min_rad_perc = min_rad_perc
        self._max_rad_perc = max_rad_perc

        self._domain = domain
        self._height = height

        self.construct_cluster(p0, r0, num_points_per_point, depth)

        self._sph_radii = np.array([sph.radius for sph in self.spheres])
        self._sph_lmbda = (
            np.ones(self._sph_radii.shape) * lmbda
        )  # np.array([sph.lmda for sph in self.spheres])

        self._gradient_coef = -2 / self._sph_radii**2 * np.log(1 / lmbda)

        # self._base = np.array(
        #     [
        #         1 / np.e ** (1 / r**2 * np.log(1 / l))
        #         for r, l in zip(self._sph_radii, self._sph_lmbda)
        #     ]
        # )  # np.array([1 / np.e**sph.const for sph in self.spheres])
        # print(self._base)
        self._base = np.e ** (
            -self._sph_radii ** (-2) * np.log(height / self._sph_lmbda)
        )
        self._exp = -self._sph_radii ** (-2) * np.log(height / self._sph_lmbda)
        self._sph_locs = np.array([sph.loc.array for sph in self.spheres])

    @property
    def ndims(self):
        return self._ndims

    @property
    def lmbda(self):
        return self._lmbda

    @property
    def min_dist_b_perc(self):
        return self._min_dist_b_perc

    @property
    def max_dist_b_perc(self):
        return self._max_dist_b_perc

    @property
    def min_rad_perc(self):
        return self._min_rad_perc

    @property
    def max_rad_perc(self):
        return self._max_rad_perc

    def construct_cluster(self, p0: Point, r0: float, n: int, k: int):
        queue = [(p0, r0)]
        self.spheres: list[ProbilisticSphere] = []

        remaining = n**k
        while len(queue) > 0 and remaining > 0:
            p, r = queue.pop()

            self.spheres.append(ProbilisticSphere(p, r, self.lmbda, self._height))
            remaining -= 1

            queue = [self.create_point_from(p, r) for i in range(n)] + queue

    def create_point_from(self, p: Point, r: float) -> tuple[Point, float]:
        ## dist
        # r1 * (1 + min_dist_b_perc) < d < r1 * (1 + max_dist_b_perc)
        # min = 0, d must be beyond the border
        # min = -1, d must be beyond the r1's center
        # max = 0, d must be before the border
        # max = 1, d must be before twice the radius

        i = 0
        valid = False
        while not valid:
            if i > 100:
                raise Exception(
                    "100 failed attempts for generating a sphere in bounds of domain."
                )

            # pick random direction and distance to find location
            v = np.random.rand(self.ndims)
            v = v * 2 - 1
            v /= np.linalg.norm(v)
            d = self._random_between(
                r * (1 + self.min_dist_b_perc), r * (1 + self.max_dist_b_perc)
            )
            p2 = p + Point(v * d)

            # pick a radius that touches the parent sphere
            min_r = (1 + self.min_rad_perc) * (d - r)
            max_r = (1 + self.max_rad_perc) * r
            r2 = self._random_between(min_r, max_r)

            c = np.ones(p.array.shape)
            c = c / np.linalg.norm(c) * r2

            # if domain specified, do not leave domain.
            valid = (
                self._domain is None
                or (p2 + c) in self._domain
                and (p2 - c) in self._domain
            )
            i += 1

        return (p2, r2)

    def score(self, p: Point) -> ndarray:
        dif2 = np.linalg.norm(p.array - self._sph_locs, axis=1) ** 2
        # return sum(self._base**dif2)
        # closest_index = min(
        #     enumerate(
        #         abs(np.linalg.norm(p - self._sph_locs, axis=1) - self._sph_radii).T
        #     ),
        #     key=lambda pair: pair[1],
        # )[0]
        return sum(self._height * np.e ** (self._exp * dif2))  # sum(self._base**dif2)
        # return np.array([sum(map(lambda sph: sph.score(p), self.spheres))])

    def classify_score(self, score: ndarray) -> bool:
        return np.linalg.norm(score) > self.lmbda

    def get_input_dims(self):
        return self.ndims

    def get_score_dims(self):
        return 1

    def generate_random_target(self):
        sph_index = random.randint(0, len(self.spheres) - 1)
        return self.spheres[sph_index].generate_random_target()

    def generate_random_nontarget(self):
        raise NotImplementedError()

    def _nearest_sphere(self, b: Point) -> ProbilisticSphere:
        nearest_err = self.spheres[0].boundary_err(b)
        nearest = self.spheres[0]

        for sph in self.spheres[1:]:
            if err := sph.boundary_err(b) < nearest_err:
                nearest_err = err
                nearest = sph

        return nearest

    def boundary_err(self, b: Point) -> float:
        "distance from boundary"
        # return min(abs(np.linalg.norm(b - self._sph_locs, axis=1) - self._sph_radii))

        # return self._nearest_sphere(b).boundary_err(b)
        # return min(self.spheres, key=lambda sph:
        # sph.boundary_err(b)).boundary_err(b)

        # linearization approach - led to high-error :(

        # err_v[err_v > 1] = 0  # get rid of inf due to axis alignment
        return (self.score(b) - self.lmbda) / np.linalg.norm(self.gradient(b))

    def true_osv(self, b: Point) -> ndarray:
        sph = self._nearest_sphere(b)
        return self.normalize((b - sph.loc).array)

    def osv_err(self, b: Point, n: ndarray) -> float:
        return self.angle_between(self.true_osv(b), n)

    def gradient(self, p: Point) -> ndarray:
        # return sum(self.spheres, key=lambda sph: sph.gradient(p))
        return self._gradient_coef[None].T * (p.array - self._sph_locs) * self.score(p)

    @staticmethod
    def _random_between(a, b) -> float:
        return random.random() * (b - a) + a

    @staticmethod
    def normalize(v: ndarray) -> ndarray:
        return v / np.linalg.norm(v)

    @classmethod
    def angle_between(cls, u, v):
        u, v = cls.normalize(u), cls.normalize(v)
        return np.arccos(np.clip(np.dot(u, v), -1, 1.0))


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
        training_data_injection: list[tuple[Point, ndarray]] = None,
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
        self.training_data_injection = training_data_injection


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

    @staticmethod
    def generate_data_targetdistr(
        params: ANNParams, target_percentage=0.5, min_score=None
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
        n_true = int(params.training_size * target_percentage)
        n_false = int(params.training_size * (1 - target_percentage))

        false_cnt = 0

        false_data: list[tuple[Point, ndarray]] = []
        true_data: list[tuple[Point, ndarray]] = []

        seq = RandomSequence(params.seq.domain, params.seq.axes_names, params.seq.seed)

        while false_cnt < n_false:
            p = seq.get_sample(1).points[0]
            score = params.envelope.score(p)

            if not params.envelope.classify_score(score):
                false_data.append((p, score))
                false_cnt += 1

        for i in range(n_true):
            p = params.envelope.generate_random_target()
            score = params.envelope.score(p)
            true_data.append((p, score))
            assert params.envelope.classify_score(
                score
            ), "Generate random target failed, got non-target instead?"

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

        data = (
            params.training_data_injection
            if params.training_data_injection
            else self.generate_data_targetdistr(params)
        )

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


def test_cluster():
    import matplotlib.pyplot as plt
    from sim_bug_tools.graphics import Grapher

    ndims = 3
    domain = Domain.normalized(ndims)

    p0 = Point([0.5] * ndims)
    r0 = 0.15

    k = 4

    n = 5

    print(len(p0))

    clst = ProbilisticSphereCluster(
        n, k, r0, p0, min_dist_b_perc=0, min_rad_perc=0, max_rad_perc=0.01, seed=1
    )
    g = Grapher(ndims == 3, domain)
    for sph in clst.spheres:
        g.draw_sphere(sph.loc, sph.radius)

    # from sim_bug_tools.rng.lds.sequences import SobolSequence

    # seq = SobolSequence(domain, [f"x{i}" for i in range(ndims)])
    # ts = []
    # nonts = []
    # print("sampling")
    # for p in seq.get_sample(1000).points:
    #     if cls := clst.classify(p):
    #         ts.append(p)
    #     else:
    #         nonts.append(p)

    # print("displaying")
    # g.plot_all_points(ts, color="red")
    # g.plot_all_points(nonts, color="blue")

    plt.show()
    print("what")


if __name__ == "__main__":
    test_cluster()


"""
while (
            true_cnt + false_cnt < (params.training_size // 2) * 2
        ):  # round nearest even
            for p in seq.get_sample(params.training_size).points:
                score = params.envelope.score(p)

                if params.envelope.classify_score(score) and true_cnt < n_true:
                    true_cnt += 1
                    true_data.append((p, score))
                    # subdata = self._shotgun_sampling(params, p, 0.01, 50)

                    if true_cnt % 10:
                        subdata = self._sample_within(
                            params, [d[0] for d in true_data], 50
                        )
                        if len(subdata) + true_cnt > n_true:
                            true_data.extend(subdata[: n_true - true_cnt])
                            true_cnt = n_true
                        else:
                            true_data.extend(subdata)
                            true_cnt += len(subdata)
                elif (
                    (min_score is None or score >= min_score)
                    and not params.envelope.classify_score(score)
                    and false_cnt < n_false
                ):
                    false_cnt += 1
                    false_data.append((p, score))
                elif true_cnt + false_cnt >= (params.training_size // 2) * 2:
                    break

            i += 1
"""
