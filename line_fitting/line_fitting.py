import numpy as np
import warnings
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.misc import derivative
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel as C,
)
from PySide6.QtWidgets import QMessageBox

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.gaussian_process.kernels"
)


class Regressor:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_min = np.min(self.x)
        self.x_max = np.max(self.x)

    def cleanup(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def fit(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def second_derivative(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def generateValues(self, num_values):
        x_values = np.linspace(self.x_min, self.x_max, num_values)
        y_values = self.predict(x_values)
        return x_values, y_values

    def generateBoundedValues(self, num_values, a, b):
        x_values = np.linspace(a, b, num_values)
        y_values = self.predict(x_values)
        return x_values, y_values

    def max(self):
        result = minimize_scalar(
            lambda x: -1 * self.predict(x),
            bounds=(self.x_min, self.x_max),
            method="bounded",
        )
        x_maximum = result.x
        maximum = -result.fun
        return x_maximum, maximum

    def centroid(self, a, b):
        integral_f = quad(self.predict, a, b)[0]
        integral_xf = quad(lambda x: x * self.predict(x), a, b)[0]
        centroid = integral_xf / integral_f

        return centroid, self.predict(centroid)

    def full_centroid(self):
        return self.centroid(self.x_min, self.x_max)

    def bounded_centroid(self, bound):
        x_values = np.linspace(self.x_min, self.x_max, 100)
        y_values = self.predict(x_values)
        above_bound = x_values[y_values > bound]
        if len(above_bound) == 0:
            raise ValueError("No values above bound found.")

        a = above_bound[0]
        b = above_bound[-1]

        return self.centroid(a, b)

    def centroid_left_bound(self):
        bound = self.predict(self.x_min)
        return self.bounded_centroid(bound)

    def half_height_centroid(self):
        _, maximum = self.max()
        half_maximum = maximum / 2

        return self.bounded_centroid(half_maximum)

    def left_bound_half_height_centroid(self):
        _, maximum = self.max()
        bound = self.predict(self.x_min)
        bound = bound + (maximum - bound) / 2

        return self.bounded_centroid(bound)

    def inflection(self):
        x_values = np.linspace(
            self.max()[0], self.x_max
        )  # Xvalues between the maximum and the end of the range
        second_deriv = np.array([self.second_derivative(x) for x in x_values])
        sign_changes = np.where(np.diff(np.sign(second_deriv)))[0]
        inflection_x_values = x_values[sign_changes]
        if len(inflection_x_values) == 0:
            raise ValueError("No inflection point found after maximum.")
        if len(inflection_x_values) > 1:
            warnings.warn(
                "Multiple inflection points found. Using first after maximum."
            )
        return inflection_x_values[0], self.predict(inflection_x_values[0])

    def bounded_cross_correlation(self, reference):
        if self == reference:
            return 0, 0
        num_values = 10000
        _, ref = reference.generateBoundedValues(num_values, 1, 2.5)
        _, values = self.generateBoundedValues(num_values, 1, 2.5)

        cross_corr = np.correlate(
            values - np.mean(values), ref - np.mean(ref), mode="full"
        )
        print(np.max(cross_corr))
        lag = np.argmax(cross_corr) - (len(values) - 1)

        return lag, 0

    def cross_correlation(self, reference):
        if self == reference:
            return 0, 0
        num_values = 10000
        _, ref = reference.generateValues(num_values)
        _, values = self.generateValues(num_values)

        cross_corr = np.correlate(
            values - np.mean(values), ref - np.mean(ref), mode="full"
        )
        print(np.max(cross_corr))
        lag = np.argmax(cross_corr) - (len(values) - 1)

        return lag, 0


class GPRegression(Regressor):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.gp = None

    def cleanup(self):
        self.gp = None

    def fit(self):
        kernel = (C(1.0, (1e-3, 1e2)) * RationalQuadratic()) + WhiteKernel(
            noise_level=0.001, noise_level_bounds=(1e-5, 0.1)
        )
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp.fit(self.x.reshape(-1, 1), self.y)

    def predict(self, x_pred):
        x_pred = np.array(x_pred).reshape(-1, 1)
        y_pred = self.gp.predict(x_pred)
        if len(y_pred) == 1:
            y_pred = y_pred[0]
        return y_pred

    def second_derivative(self, x):
        return derivative(self.predict, x, dx=1e-6, n=2)


class No_Reg(Regressor):
    def __init__(self, x, y):
        # Get the sorted indices based on self.x
        x = np.array(x)
        y = np.array(y)
        sorted_indices = np.argsort(x)

        # Sort both arrays using the sorted indices
        self.x = x[sorted_indices]
        self.y = y[sorted_indices]

    def fit(self):
        pass

    def predict(self, x):
        pass

    def second_derivative(self, x):
        pass

    def generateValues(self, num_values):
        return self.x, self.y

    def generateBoundedValues(self, num_values, a, b):
        pass

    def max(self):
        return self.x[np.argmax(self.y)], np.max(self.y)

    def centroid(self, a, b):
        raise NotImplementedError("Bounded centroid not implemented for No_Reg")

    def full_centroid(self):
        centroid = np.sum(self.x * self.y) / np.sum(self.y)
        return centroid, 0

    def bounded_centroid(self, bound):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Not Implemented")
        msg.setText("Bounded centroids not implemented for No Regression.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        raise NotImplementedError("Bounded centroid not implemented for No_Reg")

    def centroid_left_bound(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Not Implemented")
        msg.setText("Bounded centroids are not implemented for No Regression.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        raise NotImplementedError("Bounded centroid not implemented for No_Reg")

    def left_bound_half_height_centroid(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Not Implemented")
        msg.setText("Bounded centroids are not implemented for No Regression.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        raise NotImplementedError("Bounded centroid not implemented for No_Reg")

    def inflection(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Not Implemented")
        msg.setText("Inflection not implemented for No Regression.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        raise NotImplementedError("Inflection not implemented for No_Reg")

    def bounded_cross_correlation(self, reference):
        raise NotImplementedError(
            "Bounded cross correlation not implemented for No_Reg"
        )

    def cross_correlation(self, reference):
        raise NotImplementedError("Cross correlation not implemented for No_Reg")


class Polynomial(Regressor):
    def __init__(self, x, y, degree):
        super().__init__(x, y)
        self.degree = degree
        self.polynomial = None

    def cleanup(self):
        self.polynomial = None

    def fit(self):
        self.polynomial = np.polynomial.Polynomial.fit(self.x, self.y, self.degree)

    def predict(self, x_pred):
        return self.polynomial(x_pred)

    def second_derivative(self, x):
        return self.polynomial.deriv(2)(x)

    def max(self):
        derivative = self.polynomial.deriv()
        roots = derivative.roots()
        critical_x_values = roots.real[abs(roots.imag < 1.0e-5)]
        end_points = np.array([self.x_min, self.x_max])
        critical_x_values = np.concatenate([critical_x_values, end_points])
        critical_x_values = critical_x_values[
            (critical_x_values > self.x_min) & (critical_x_values < self.x_max)
        ]

        critical_y_values = self.predict(critical_x_values)

        max_index = np.argmax(critical_y_values)
        maximum_x = critical_x_values[max_index]
        maximum = critical_y_values[max_index]

        return maximum_x, maximum

    def inflection(self):
        second_derivative = self.polynomial.deriv(2)
        roots = second_derivative.roots()
        inflection_x_values = roots.real[abs(roots.imag) < 1.0e-5]
        inflection_x_values = inflection_x_values[
            (inflection_x_values > self.max()[0]) & (inflection_x_values < self.x_max)
        ]

        if len(inflection_x_values) == 0:
            raise ValueError("No inflection point found after maximum.")
        if len(inflection_x_values) > 1:
            warnings.warn(
                "Multiple inflection points found. Using first after maximum."
            )
        return inflection_x_values[0], self.predict(inflection_x_values[0])
