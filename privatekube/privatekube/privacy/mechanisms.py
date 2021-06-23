"""
Helper functions for differential privacy.

These are prototyping functions. We don't use production-level randomness & snapping mechanism
against floating-point attacks.

"""

import numpy as np
import sympy


def add_gaussian_noise(x, sensitivity_l2, noise_multiplier):
    """
    Sanitizes the x array using the gaussian mechanism. Delta and epsilon must
    be greater than 0.

    :param x: The value to sanitize.
    :param sensitivity: The L2 sensitivity of the function.
    :param noise_multiplier: The noise multiplier (= sigma if sensitivity 1)
    :return: A sanitized version of x.
    """
    if noise_multiplier <= 0:
        return x
    return x + np.random.normal(
        loc=0.0, scale=sensitivity_l2 * noise_multiplier, size=np.shape(x)
    )


def add_laplace_noise(x, sensitivity_l1, noise_multiplier):
    """
    Sanitizes the values in x with noise from a laplacian distribution with
    the given sensitivity and privacy budget.

    :param x: The value to sanitize.
    :param sensitivity: The L1 sensitivity of the function.
    :param noise_multiplier: The noise multiplier (= lambda if sensitivity 1)
    :return: Sanitized version of x.
    """
    if noise_multiplier <= 0:
        return x
    return x + np.random.laplace(
        loc=0, scale=sensitivity_l1 * noise_multiplier, size=np.shape(x)
    )


def laplace_sanitize(x, sensitivity, epsilon):
    """
    Sanitizes the values in x with noise from a laplacian distribution with
    the given sensitivity and privacy budget.

    :param x: The value to sanitize.
    :param sensitivity: The L1 sensitivity of the function.
    :param epsilon: The privacy budget to use. If epsilon is <= 0 then x is
        returned unchanged.
    :return: Sanitized version of x.
    """
    if epsilon <= 0:
        return x
    return x + np.random.laplace(loc=0, scale=sensitivity / epsilon, size=np.shape(x))


def gaussian_sanitize(x, sensitivity, epsilon, delta):
    """
    Sanitizes the x array using the gaussian mechanism. Delta and epsilon must
    be greater than 0.

    :param x: The value to sanitize.
    :param sensitivity: The L2 sensitivity of the function.
    :param epsilon: The epsilon portion of the privacy budget to use.
    :param delta: The delta portion of the privacy budget to use.
    :return: A sanitized version of x.
    """
    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon * sensitivity
    return x + np.random.normal(loc=0.0, scale=sigma, size=np.shape(x))


def _j_list(x, beta, a, c, L, U):
    """
    Computes the j list for finding the smooth sensitivity of the quantile.

    :param x: The dataset overwhich the smooth sensitivity will be computed.
    :param a: Minimum bound of the quantile for this iteration.
    :param c: Upper bound for the quantile for this iteration.
    :param L: Global minimum bound for the quantile.
    :param U: Global uppder bound for the quantile.
    """
    if c < a:
        return []
    if L > U:
        raise ValueError("L < U required. L: %d, U: %d", (L, U))

    b = int(np.floor((a + c) / 2))
    xb = x[b]

    jb_sens = [(x[j] - xb) * np.exp(beta * (b - j + 1)) for j in range(L, U + 1)]
    jb_index = np.argmax(jb_sens)
    jb = jb_index + L
    if L == U:
        return [jb_sens[jb_index]]
    else:
        return (
            _j_list(x, beta, a, b - 1, L, jb)
            + [jb_sens[jb_index]]
            + _j_list(x, beta, b + 1, c, jb, U)
        )


def private_quantile(x, quantile, beta, epsilon, max_value, min_value):
    """
    Privately computes the given quantile of x using smooth sensitivity.

    :param x: Vector a data over which the quantile will be computed.
    :param quantile: The quantile to compute over x.
    :param beta: Parameter defining the tightness of the smooth sensitivity.
    :param epsilon: Privacy budget for computing the quantile.
    :param max_value: The maximum of the range of x.
    :param min_value: The minimum of the range of x.
    :return: Privacy preserving quantile of x.
    """

    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("Quantile must be in range [0.0, 1.0]")
    if min_value >= max_value:
        raise ValueError("min_value < max_value required.")

    n = np.size(x)
    x = np.sort(x)
    quantile_index = int(np.floor((n - 1) * quantile + 1))
    quantile_value = x[quantile_index]

    jlist = _j_list(x, beta, 0, quantile_index, quantile_index, n - 1)
    smooth_sensitivity = np.max(jlist)

    return (
        quantile_value
        + np.random.standard_cauchy() * 6.0 / epsilon * smooth_sensitivity
    )


def symbolic_exponential_mechanism_weights(q, epsilon, sensitivity):
    exp_sym_q = []
    e = sympy.S(np.e)
    epsilon_sym = sympy.S(epsilon)
    sensitivity_sym = sympy.S(sensitivity)
    for value in q:
        exp_sym_q.append(e ** (epsilon_sym * sympy.S(value) / (2 * sensitivity_sym)))
    sum_sym = sum(exp_sym_q)

    weights = [x / sum_sym for x in exp_sym_q]
    return weights


def exponential_mechanism(q, epsilon, sensitivity):
    if sensitivity <= 0.0:
        raise ValueError("Invalid sensitivity: %e" % sensitivity)
    if len(q) == 0:
        raise ValueError("Must submit at least 1 choice: len(q) = 0")
    exp_q = np.exp(epsilon * q / (2 * sensitivity))
    weights = exp_q / np.sum(exp_q)

    # If the sensitivity is very small or the utilities are large it leads to
    # overflows so we compute the weights using sympy.
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        weights = symbolic_exponential_mechanism_weights(q, epsilon, sensitivity)

    weight_sum = 0.0
    r = np.random.random()
    for i, v in enumerate(weights):
        weight_sum += v
        if r < weight_sum:
            return i
    return len(q) - 1


def large_margin(q, n, epsilon, delta):

    # For a function to be 1/n lipschitz it can't have a range over 1.0
    if np.any(q > 1.0):
        raise ValueError("Invalid q > 1.0: %s" % str(q))
    if np.any(q < 0.0):
        raise ValueError("Invalid q < 0.0: %s" % str(q))
    if len(q) == 0:
        raise ValueError("Must submit options for large margin mechanism.")
    assert delta > 0.0
    assert epsilon > 0.0

    # Must be sorted in descending order.
    assert all(q[i] >= q[i + 1] for i in range(len(q) - 1))

    f_n = float(n)

    # Set up the margins.
    K = len(q)
    r = np.arange(K) + 1
    t = 6.0 / f_n * (1 + np.log(r * 3 / delta) / epsilon)
    T = (
        3.0 / (f_n * epsilon) * np.log(3.0 / (2.0 * delta))
        + 6.0 / (f_n * epsilon) * np.log(3.0 / delta)
        + 12 / (f_n * epsilon) * np.log(3 * r * (r + 1) / delta)
        + t
    )

    # Set the parameters for choosing the margins.
    Z = laplace_sanitize(0.0, 1.0, 3.0 / epsilon)
    # Estimating the maximum value.
    m = q[0] + Z / float(n)
    G = laplace_sanitize(0.0, 1.0, 6.0 / epsilon)
    Zl = laplace_sanitize(np.zeros(K), 1.0, 12.0 / epsilon)

    # Find the margin
    l = 0
    while l < (K - 1):
        if m - q[l + 1] > (Zl[l] + G) / f_n + T[l]:
            break
        l += 1
    assert l < len(q)

    Ul = q[: (l + 1)]
    if len(Ul) == 1:
        return 0
    Ul_n = Ul * f_n
    return exponential_mechanism(Ul_n, epsilon, 3.0)


def private_mean(x, epsilon, sum_sensitivity):
    """
    Computes a private mean with privacy budget epsilon using the given
    sensitivity.

    :param x: Vector over which the mean will be computed.
    :param epsilon: The privacy budget.
    :param sum_sensitivity: The sensitivity of the sum calculation.
    :return: The private mean.
    """
    m = np.mean(x)
    l = x.shape[0]
    return laplace_sanitize(m, sum_sensitivity / float(l), epsilon)


def private_equiwidth_bin(x, epsilon, min_x, max_x, bins=10):
    """
    Computes equiwidth histogram of x.

    :param x: The vector over which the histogram will be computed.
    :param epsilon: The privacy budget.
    :param bins: The number of bins in the histogram.
    :return: Histogram of bin counts.
    """

    hist, bin_edges = np.histogram(x, bins, range=(min_x, max_x))
    return laplace_sanitize(hist, 1.0, epsilon), bin_edges


def private_top_k(x, k, epsilon):
    """
    Returns the top k elements by adding noise to the counts.

    :param x: Vector to count.
    :param epsilon: Privacy budget.
    :return: The top k elements in the vector with their private counts.
    """

    values, counts = np.unique(x, return_counts=True)
    counts = laplace_sanitize(counts, 1.0, epsilon)
    indices = np.argsort(counts)[-k:]
    return values[indices], counts[indices]


def private_sd(x, epsilon, sum_sensitivity):
    """
    Computes a private standard deviation.

    :param x: Vector over which to compute the standar deviation.
    :param epsilon: Privacy budget value.
    :param sum_sensitivity: The sensitivity of the summation.
    """
    m = np.mean(x)
    numerator = laplace_sanitize(
        np.sum(np.power(x - m, 2)), sum_sensitivity ** 2, epsilon / 2
    )
    denominator = laplace_sanitize(x.size, 1.0, epsilon / 2.0)
    return np.sqrt(numerator / denominator)
