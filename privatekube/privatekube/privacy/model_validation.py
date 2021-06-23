from collections import defaultdict

from statsmodels.stats.proportion import proportion_confint

import csv
import gzip
import numpy as np
import pandas as pd
import sys

# import mechanisms


def sanitize_negative_log_likelihood(nll, n, B, epsilon):
    """
    Sanitizes a negative log likelihood value using the laplace mechanism.

    :param nll: The non-private negative log likelihood value.
    :param n: The number of points used to compute the negative log likelihood.
    :param B: The bound on the negative log likelihood.
    :param epsilon: The epsilon value of the privacy budget.
    :return: Private version of the negative log likelihood.
    """
    if epsilon <= 0.0:
        return nll
    b = (B * B) / (epsilon * n * np.sqrt(2))
    return nll + np.random.laplace(scale=b)


def sanitize_mean_absolute_error(mae, n, B, epsilon):
    """
    Sanitizes a mean absolute error value using the laplace mechanism.

    :param mae: The non-private mean absolute error value.
    :param n: The number of points used to compute the mean absolute error.
    :param B: The bound on the mean absolute error.
    :param epsilon: The epsilon value of the privacy budget.
    :return: Private version of the mean absolute error.
    """
    if epsilon <= 0.0:
        return mae
    b = B / (n * epsilon)
    return mae + np.random.laplace(scale=b)


def sanitize_mean_squared_error(mse, n, B, epsilon):
    """
    Sanitizes a mean squared error value using the laplace mechanism.

    :param mae: The non-private mean squared error value.
    :param n: The number of points used to compute the mean squared error.
    :param B: The bound on the mean squared error.
    :param epsilon: The epsilon value of the privacy budget.
    :return: Private version of the mean squared error.
    """
    if epsilon <= 0.0:
        return mse
    b = (B ** 2) / (n * epsilon)
    return mse + np.random.laplace(scale=b)


def sanitize_mean_squared_log_error(msle, n, B, epsilon):
    """
    Sanitizes a mean squared log error value using the laplace mechanism.

    :param mae: The non-private mean squared log error value.
    :param n: The number of points used to compute the mean squared log error.
    :param B: The bound on the variable for which we compute the MSLE.
    :param epsilon: The epsilon value of the privacy budget.
    :return: Private version of the mean squared log error.
    """
    if epsilon <= 0.0:
        return msle
    # Sensitivity for L(u,v) = 1/n \sum (log(u + 1) - log(v + 1))^2
    b = np.log(B + 1) ** 2 / (n * epsilon)
    return msle + np.random.laplace(scale=b)


def noise_correction(epsilon, delta):
    """
    Computes the noise correction: 2/epsilon * log(3/delta)
    """
    assert delta > 0
    if epsilon <= 0 or np.isinf(epsilon):
        return 0
    return 2.0 / epsilon * np.log(3 / (2.0 * delta))


def _compute_loss_accept_bound(test_error_mean, n_test, B, delta_accept):
    return (
        test_error_mean
        + np.sqrt((2.0 * B * test_error_mean * np.log(3.0 / delta_accept)) / n_test)
        + 4.0 * B * np.log(3.0 / delta_accept) / n_test
    )


def _compute_loss_reject_bound(train_error_mean, n_train, B, delta_reject):
    return train_error_mean - B * np.sqrt(np.log(3.0 / delta_reject) / n_train)


def slaed_loss_validation(
    train_error,
    test_error,
    n_train,
    n_test,
    tau,
    train_epsilon,
    test_epsilon,
    B,
    delta_upper,
    delta_lower,
    correct_for_noise=True,
    include_accept=True,
    include_reject=True,
    block_size=None,
):
    assert n_test > 0
    if not include_reject and not include_accept:
        raise ValueError("include_accept and include_reject cannot both be false.")
    if delta_upper <= 0 or delta_lower <= 0:
        raise ValueError(
            "Invalid confidence values: %f %f" % (delta_lower, delta_upper)
        )
    train_error = np.clip(train_error, 0, B)
    test_error = np.clip(test_error, 0, B)
    noise_draws = 1
    if block_size is not None:
        N = int(n_train + n_test)
        noise_draws = int(N / block_size)
        if N % block_size != 0:
            noise_draws += 1

    assert B == 1
    assert delta_upper == delta_lower

    delta_accept = delta_upper
    delta_reject = delta_lower

    # Convert the loss, which should be an average, to a sum.
    train_error_sum = train_error * n_train
    test_error_sum = test_error * n_test

    if train_epsilon > 0:
        dp_train_error_sum = train_error_sum + np.sum(
            np.random.laplace(loc=0, scale=2.0 / train_epsilon, size=noise_draws)
        )
        dp_n_train = n_train + np.sum(
            np.random.laplace(loc=0, scale=2.0 / train_epsilon, size=noise_draws)
        )
    else:
        dp_train_error_sum = train_error_sum
        dp_n_train = n_train
    if test_epsilon > 0:
        dp_test_error_sum = test_error_sum + np.sum(
            np.random.laplace(loc=0, scale=2.0 / test_epsilon, size=noise_draws)
        )
        dp_n_test = n_test + np.sum(
            np.random.laplace(loc=0, scale=2.0 / test_epsilon, size=noise_draws)
        )
    else:
        dp_test_error_sum = test_error_sum
        dp_n_test = n_test

    if test_epsilon > 0 and correct_for_noise and include_accept:
        if dp_n_test < noise_correction(test_epsilon, delta_accept):
            return 0

    # These are computed without correction.
    dp_test_error_mean = dp_test_error_sum / dp_n_test
    dp_train_error_mean = dp_train_error_sum / dp_n_train

    # Compute the various corrected values.
    dp_n_train_upper = (
        dp_n_train + noise_correction(train_epsilon, delta_reject) * noise_draws
    )
    dp_n_train_lower = (
        dp_n_train - noise_correction(train_epsilon, delta_reject) * noise_draws
    )
    dp_n_test_lower = (
        dp_n_test - noise_correction(test_epsilon, delta_accept) * noise_draws
    )
    corrected_dp_train_error_sum = (
        dp_train_error_sum
        - noise_correction(train_epsilon / B, delta_reject) * noise_draws
    )
    corrected_dp_test_error_sum = (
        dp_test_error_sum
        + noise_correction(test_epsilon / B, delta_accept) * noise_draws
    )

    corrected_dp_test_error_mean = corrected_dp_test_error_sum / dp_n_test_lower
    corrected_dp_train_error_mean = corrected_dp_train_error_sum / dp_n_train_upper
    """
    upper_bound = dp_test_error_mean + np.sqrt(
        (2.0 * B * dp_test_error_mean * np.log(3.0 / delta_accept)) /
        dp_n_test_lower) + 4.0 * B * np.log(
            3.0 / delta_accept) / dp_n_test_lower
    """
    corrected_upper_bound = _compute_loss_accept_bound(
        corrected_dp_test_error_mean, dp_n_test_lower, B, delta_accept
    )
    uncorrected_upper_bound = _compute_loss_accept_bound(
        dp_test_error_mean, dp_n_test, B, delta_accept
    )

    # lower_bound = dp_train_error_mean - B * np.sqrt(
    #    np.log(3.0 / delta_reject) / dp_n_train_lower)
    corrected_lower_bound = _compute_loss_reject_bound(
        corrected_dp_train_error_mean, dp_n_train_lower, B, delta_reject
    )
    uncorrected_lower_bound = _compute_loss_reject_bound(
        dp_train_error_mean, dp_n_test, B, delta_reject
    )

    if np.any(
        np.isnan(
            [
                corrected_upper_bound,
                uncorrected_upper_bound,
                corrected_lower_bound,
                uncorrected_lower_bound,
            ]
        )
    ):
        return 0

    # print("%d %d %1.4f %1.4f %f %f %f %f" %
    #      (noise_draws, n_test, test_error, tau, test_epsilon,
    #       corrected_upper_bound, uncorrected_upper_bound,
    #       noise_correction(test_epsilon, delta_accept)))

    if correct_for_noise and corrected_upper_bound < uncorrected_upper_bound:
        print(
            "corrected: %f / %f = %f"
            % (
                corrected_dp_test_error_sum,
                dp_n_test_lower,
                corrected_dp_test_error_mean,
            )
        )
        print(
            "uncorrected: %f / %f = %f"
            % (dp_test_error_sum, dp_n_test, dp_test_error_mean)
        )
        print(dp_test_error_mean, corrected_dp_test_error_mean)
        raise ValueError(
            "Correction violation: %f %f"
            % (corrected_upper_bound, uncorrected_upper_bound)
        )

    if include_reject:
        if correct_for_noise and corrected_lower_bound > tau:
            return -1
        elif not correct_for_noise and uncorrected_lower_bound > tau:
            return -1
    if include_accept:
        if correct_for_noise and corrected_upper_bound <= tau:
            return 1
        elif not correct_for_noise and uncorrected_upper_bound <= tau:
            return 1
    # 0 is returned where there neither evidence to accept nor to reject.
    return 0


def non_adaptive_loss_profile(
    non_private_train_losses,
    non_private_test_losses,
    n_trains,
    n_tests,
    tau,
    epsilon_train,
    epsilon_test,
    B,
    delta_upper,
    delta_lower,
    sanitize_error_fn,
    correct_for_noise=True,
    use_upper_bound=True,
    use_lower_bound=True,
    block_size=None,
):
    n_trials = len(non_private_test_losses)
    assert len(non_private_test_losses) == n_trials
    assert len(n_trains) == n_trials
    assert len(n_tests) == n_trials

    for i in range(n_trials):
        if i == 0 and non_private_test_losses[0] == 0:
            continue
        iteration = i + 1
        ret_value = slaed_loss_validation(
            non_private_train_losses[i],
            non_private_test_losses[i],
            n_trains[i],
            n_tests[i],
            tau,
            epsilon_train,
            epsilon_test,
            1.0,
            delta_upper / (iteration * (iteration + 1)),
            delta_lower / (iteration * (iteration + 1)),
            correct_for_noise=correct_for_noise,
            include_accept=use_upper_bound,
            include_reject=use_lower_bound,
            block_size=block_size,
        )
        if ret_value == 1:
            return [(n_trains[i], n_trains[i])]
        elif ret_value == -1:
            return [(np.NINF, n_trains[i])]
    return [(np.inf, n_trains[-1])]


def independent_loss_profile(
    non_private_train_losses,
    non_private_test_losses,
    n_trains,
    n_tests,
    tau,
    epsilon_train,
    epsilon_test,
    B,
    delta_upper,
    delta_lower,
    sanitize_error_fn,
    correct_for_noise=True,
    use_upper_bound=True,
    use_lower_bound=True,
    block_size=None,
):
    """
    The same as the non adaptive version but tests each size independently.
    """
    n_trials = len(non_private_test_losses)
    assert len(non_private_test_losses) == n_trials
    assert len(n_trains) == n_trials
    assert len(n_tests) == n_trials

    return_values = []

    for i in range(n_trials):
        if i == 0 and non_private_test_losses[0] == 0:
            continue
        ret_value = slaed_loss_validation(
            non_private_train_losses[i],
            non_private_test_losses[i],
            n_trains[i],
            n_tests[i],
            tau,
            epsilon_train,
            epsilon_test,
            1.0,
            delta_upper,
            delta_lower,
            correct_for_noise=correct_for_noise,
            include_accept=use_upper_bound,
            include_reject=use_lower_bound,
            block_size=block_size,
        )
        if ret_value == 1:
            return_values.append((n_trains[i], n_trains[i]))
        elif ret_value == -1:
            return_values.append((np.NINF, n_trains[i]))
        else:
            return_values.append((np.inf, n_trains[i]))
    return return_values


def non_adaptive_profile(
    training_error,
    evaluation_error,
    training_sizes,
    evaluation_sizes,
    tau,
    train_epsilon,
    test_epsilon,
    B,
    delta_upper,
    delta_lower,
    sanitize_error_fn,
    correct_for_noise=True,
    use_upper_bound=True,
    use_lower_bound=True,
    block_size=None,
):
    """
    Non-adaptively profiles using the ERM based mechanism.

    :param training_error: The training error at iteration.
    :param evaluation_error: Array of evaluation error values.
    :param training_sizes: Array of integers increasing order representing the
        number of points on which the PTK was trained.
    :param evaluation_sizes: Array of integers representing the number of
        points on which the PTK was evaluated.
    :param tau: The goal value for PTK performance.
    :param train_epsilon: Privacy budget allocated for the training data.
    :param test_epsilon: Privacy budget allocated for the evaluation data.
    :param B: Bound on the evaluation error value.
    :param delta_upper: Probability of profiling failure on the upper bound or
        the performance guarantee.
    :param delta_lower: Probability of profiling failure on the lower bound or
        the existence guarantee.
    :param evalulation_sensitivity: The sensitivity of the evaluation function.
    :param sanitize_error_fn: Function that will return a sanitized version
        of the error value. The function takes three parameters: the error
        value, number of points used to compute error, the bound on the error
        value, and the privacy budget epsilon.
    :param correct_for_noise: Set to True to correct for noise in the bounds.
    :param use_upper_bound: True if the upper bound will be tested.
    :param use_lower_bound: True if the lower bound will be tested.
    :param block_size: The block size controlling the number of noise draws to
        add.
    :return: A tuple. The first element is sample complexity of the
        configuration, negative infinity if the model is rejected, and infinity
        if not converged. The second element is training size when aborted or
        None if validated.
    """
    if not use_lower_bound and not use_upper_bound:
        raise ValueError("use_upper_bound and use_lower_bound cannot both be false.")
    if delta_upper <= 0 or delta_lower <= 0:
        raise ValueError(
            "Invalid confidence values: %f %f" % (delta_lower, delta_upper)
        )

    if len(training_error) > 0 and len(evaluation_error) > 0:
        if len(training_error) != len(evaluation_error):
            raise ValueError(
                "If defined training and evaluation error must "
                "be of the same length."
            )

    if not np.all(training_sizes[:-1] <= training_sizes[1:]):
        raise ValueError("training_sizes must be sorted.")

    n_rounds = len(evaluation_error)
    if len(training_sizes) != n_rounds:
        raise ValueError(
            "Invalid training_sizes length: %d != %d" % (n_rounds, len(training_sizes))
        )
    if len(evaluation_sizes) != n_rounds:
        raise ValueError(
            "Invalid evaluation_sizes length: %d != %d"
            % (n_rounds, len(evaluation_sizes))
        )
    if block_size is not None and block_size <= 0:
        raise ValueError("block_size must be >= 0")

    for index in range(n_rounds):
        i = index + 1
        T_n = training_sizes[index]
        V_n = evaluation_sizes[index]
        noise_draws = 1
        if block_size is not None:
            N = int(T_n + V_n)
            noise_draws = int(N / block_size)
            if N % block_size != 0:
                noise_draws += 1

        if use_upper_bound:
            try:
                e_error = evaluation_error[index]
                for _ in range(noise_draws):
                    e_error = sanitize_error_fn(e_error, V_n, B, test_epsilon)
                upper_bound = (
                    e_error
                    + ((4.0 * B * np.log(2.0 * i * (i + 1) / delta_upper)) / V_n)
                    + np.sqrt(
                        (2.0 * B * e_error * np.log(2.0 * i * (i + 1) / delta_upper))
                        / V_n
                    )
                )
                if correct_for_noise and test_epsilon > 0:
                    upper_bound += (
                        noise_draws
                        * np.log(i * (i + 1) / delta_upper)
                        / (V_n * test_epsilon)
                    )
                if upper_bound <= 2.0 * tau:
                    return (T_n, T_n)
            except IndexError:
                pass

        if use_lower_bound:
            try:
                t_error = training_error[index]
                for _ in range(noise_draws):
                    t_error = sanitize_error_fn(t_error, V_n, B, train_epsilon)
                lower_bound = t_error - B * np.sqrt(
                    np.log(4.0 * i * (i + 1) / delta_lower) / (2.0 * T_n)
                )
                if correct_for_noise and train_epsilon > 0.0:
                    lower_bound -= (
                        noise_draws
                        * np.log((i * (i + 1)) / delta_lower)
                        / (T_n * train_epsilon)
                    )
                if lower_bound >= tau:
                    return (np.NINF, T_n)
            except IndexError:
                pass
    return (np.inf, np.inf)


def _compute_upper_bound(
    test_error, B, n_rounds, delta, n_test, epsilon_test, correct_for_noise=True
):
    """
    Computes the upper bound from the test error.

    :param B: The bound of the loss value.
    :param n_rounds: The number of rounds that'll be tested.
    :param delta: Probability of the bound failure.
    :param n_test: The number of testing points.
    :param epsilon_test: The privacy budget used to santize the test error.
    :param correct_for_noise: Set to True if the bound should be raised to
        account for the noise added.
    """
    upper_bound = (
        test_error
        + ((4.0 * B * np.log(2.0 * n_rounds / delta)) / n_test)
        + np.sqrt((2.0 * B * test_error * np.log(2.0 * n_rounds / delta)) / n_test)
    )
    if correct_for_noise:
        upper_bound += np.log(1.0 / delta) / (n_test * epsilon_test)
    return upper_bound


def _compute_lower_bound(
    train_error, B, n_rounds, delta, n_train, epsilon_train, correct_for_noise=True
):
    """
    Computes the upper bound from the train error.

    :param B: The bound of the loss value.
    :param n_rounds: The number of rounds that'll be tested.
    :param delta: Probability of the bound failure.
    :param n_train: The number of training points.
    :param epsilon_train: The privacy budget used to santize the train error.
    :param correct_for_noise: Set to True if the bound should be raised to
        account for the noise added.
    """
    lower_bound = train_error - B * np.sqrt(
        np.log(4.0 * n_rounds / delta) / (2.0 * n_train)
    )
    if correct_for_noise:
        lower_bound -= np.log(1.0 / delta) / (n_train * epsilon_train)
    return lower_bound


def privacy_budget_search(
    train_test_fn,
    train_data,
    test_data,
    epsilon_lower_bound,
    epsilon_upper_bound,
    lambd,
    tau,
    B,
    delta,
    sanitize_error_fn,
):
    """
    :param train_test_fn: Function that will take three parameters: train set,
        test set, and epsilon value. It will return a tuple where the first
        element is the train loss and the second element is the test loss. The
        first element of the tuple may be None in which case the lower bound
        will not be calculated.
    :param train_data: Array of train data vectors.
    :param test_data: Array of test vectors.
    :param epsilon_lower_bound: The smallest epsilon privacy budget to search.
    :param epsilon_upper_bound: The largest epsilon privacy budget to search.
    :param lambd: The growth value of the epsilon value.
    :param tau: The tau performance goal.
    :param B: The bound of the loss value.
    :param delta: Probability of the bound failure.
    :param sanitize_error_fn: Function that will return a sanitized version
        of the error value. The function takes three parameters: the error
        value, number of points used to compute error, the bound on the error
        value, and the privacy budget epsilon.
    """
    accumulated_epsilon = 0.0

    n_train = np.shape(train_data[0])[0]
    n_test = np.shape(test_data[0])[0]

    n_rounds = (
        int(np.log(epsilon_upper_bound / epsilon_lower_bound) / np.log(lambd)) + 1
    )
    for i in range(n_rounds):
        epsilon = epsilon_lower_bound * np.power(lambd, i)
        accumulated_epsilon += epsilon

        train_loss, test_loss = train_test_fn(train_data, test_data, epsilon)
        private_train_loss = sanitize_error_fn(train_loss, n_train, B, epsilon)
        del train_loss
        private_test_loss = sanitize_error_fn(test_loss, n_test, B, epsilon)
        del test_loss

        lower_bound = _compute_lower_bound(
            private_train_loss, B, n_rounds, delta, n_train, epsilon
        )
        upper_bound = _compute_upper_bound(
            private_test_loss, B, n_rounds, delta, n_test, epsilon, False
        )
        if upper_bound <= 2.0 * tau:
            return epsilon, accumulated_epsilon
        elif lower_bound >= tau:
            return np.NINF, accumulated_epsilon
    return np.inf, accumulated_epsilon


def mean_profile(counts, variances, dataset_size, B, tau, epsilon, delta):
    """
    Profiles simple mean statistics.

    :param counts: Vector of non-private counts of different keys.
    :param variance: The non-private variance of value of each key.
    :param dataset_size: The total size of the dataset.
    :param B: The bound on the key value.
    :param tau: The tau performance goal.
    :param epsilon: The privacy budget to maintain.
    :param delta: The probability of failure.
    :return: The estimated sample complexity.
    """
    if counts.ndim != 1 or variances.ndim != 1:
        raise ValueError("count and variances must be 1-d vectors.")
    if counts.shape != variances.shape:
        raise ValueError("counts and variances must have the same shape.")

    count_sensitivity = 1.0 / dataset_size
    private_counts = privatekube.privacy.mechanisms.laplace_sanitize(
        counts, count_sensitivity, epsilon
    )

    # Computes the sensitivity for each of the keys base d on the count.
    variance_sensitivity = B * B / (private_counts - 1.0)
    # Passing a vector of scales will use different scales for each entry.
    private_variances = privatekube.privacy.mechanisms.laplace_sanitize(
        variances, variance_sensitivity, epsilon
    )

    # Force all of the counts to be > 1 and a numerical hack.
    private_counts = np.maximum(1.00000001, private_counts)

    sample_sizes = (dataset_size / private_counts) * (
        4.0
        * B
        * B
        * np.log(2.0 / delta)
        / (
            private_variances
            + 2.0 * B * tau
            - np.sqrt(private_variances) * np.sqrt(private_variances + 4.0 * B * tau)
        )
    )
    return sample_sizes


def binomial_optimal_risk_test(
    tau,
    non_private_train_accuracy,
    non_private_test_accuracy,
    n_train,
    n_test,
    epsilon_train,
    epsilon_test,
    delta_upper,
    delta_lower,
    correct_for_noise=True,
    include_accept=True,
    include_reject=True,
):
    """

    :param tau: The accuracy goal.
    :param non_private_train_accuracy: The fraction of correctly classified
        examples in the train set.
    :param non_private_test_accuracy: The fraction of correctly classified
        examples in the test set.
    :param n_train: The cardinality of the train set.
    :param n_test: The cardinality of the test set.
    :param epsilon_train: Epsilon privacy budget used to sanitize the train
        set.
    :param epsilon_test: Epsilon privacy budget used to sanitize the test set.
    :param delta: Probability of failure.
    :param correct_for_noise: Boolean indicating if the function should correct
        for laplace noise.
    :return: 1 on accept, -1 on reject, and 0 on grow.
    """
    delta_accept = delta_upper
    delta_reject = delta_lower

    # The raw number of successes during training and testing.
    k_train_dp = privatekube.privacy.mechanisms.laplace_sanitize(
        non_private_train_accuracy * n_train, 1.0, epsilon_train / 2.0
    )
    k_test_dp = privatekube.privacy.mechanisms.laplace_sanitize(
        non_private_test_accuracy * n_test, 1.0, epsilon_test / 2.0
    )

    n_train_dp = privatekube.privacy.mechanisms.laplace_sanitize(
        n_train, 1.0, epsilon_train / 2.0
    )
    n_test_dp = privatekube.privacy.mechanisms.laplace_sanitize(
        n_test, 1.0, epsilon_test / 2.0
    )

    if epsilon_train <= 0:
        assert n_train_dp == n_train

    if correct_for_noise and epsilon_train > 0:
        k_train_dp += 2.0 / epsilon_train * np.log(3.0 / delta_reject)
        n_train_dp -= 2.0 / epsilon_train * np.log(3.0 / delta_reject)

    if correct_for_noise and epsilon_test > 0:
        k_test_dp -= 2.0 / epsilon_test * np.log(3.0 / delta_accept)
        n_test_dp += 2.0 / epsilon_test * np.log(3.0 / delta_accept)

    _, reject_bound = proportion_confint(
        np.ceil(k_train_dp), np.floor(n_train_dp), delta_reject / 3.0, method="beta"
    )
    accept_bound, _ = proportion_confint(
        np.floor(k_test_dp), np.ceil(n_test_dp), delta_accept / 3.0, method="beta"
    )

    if np.isnan(accept_bound) or np.isnan(reject_bound):
        return 0
    elif accept_bound >= tau and include_accept:
        return 1
    elif reject_bound < tau and include_reject:
        return -1
    else:
        return 0


def non_adaptive_accuracy_profile(
    non_private_train_accuracies,
    non_private_test_accuracies,
    n_trains,
    n_tests,
    tau,
    epsilon_train,
    epsilon_test,
    B,
    delta_upper,
    delta_lower,
    sanitize_error_fn,
    correct_for_noise=True,
    use_upper_bound=True,
    use_lower_bound=True,
    block_size=None,
):
    """
    :param tau: The accuracy goal.
    :param non_private_train_accuracy: The fraction of correctly classified
        examples in the train set for each size.
    :param non_private_test_accuracy: The fraction of correctly classified
        examples in the test set for each size.
    :param n_trains: The sizes of the training sets.
    :param n_tests: The sizes of the test sets.
    :param epsilon_train: Epsilon privacy budget used to sanitize the train
        set.
    :param epsilon_test: Epsilon privacy budget used to sanitize the test set.
    :param delta: Probability of failure.
    :param correct_for_noise: Boolean indicating if the function should correct
        for laplace noise.
    :return: 1 on accept, -1 on reject, and 0 on grow.
    """
    n_trials = len(non_private_test_accuracies)
    assert len(non_private_test_accuracies) == n_trials
    assert len(n_trains) == n_trials
    assert len(n_tests) == n_trials

    for i in range(n_trials):
        iteration = i + 1
        ret_value = binomial_optimal_risk_test(
            tau,
            non_private_train_accuracies[i],
            non_private_test_accuracies[i],
            n_trains[i],
            n_tests[i],
            epsilon_train,
            epsilon_test,
            delta_upper / (iteration * (iteration + 1)),
            delta_lower / (iteration * (iteration + 1)),
            correct_for_noise=correct_for_noise,
            include_accept=use_upper_bound,
            include_reject=use_lower_bound,
        )
        if ret_value == 1:
            return [(n_trains[i], n_trains[i])]
        elif ret_value == -1:
            return [(np.NINF, n_trains[i])]
    return [(np.inf, np.inf)]


def independent_accuracy_profile(
    non_private_train_accuracies,
    non_private_test_accuracies,
    n_trains,
    n_tests,
    tau,
    epsilon_train,
    epsilon_test,
    B,
    delta_upper,
    delta_lower,
    sanitize_error_fn,
    correct_for_noise=True,
    use_upper_bound=True,
    use_lower_bound=True,
    block_size=None,
):
    """
    The same as non_adaptive but runs each size independently and does not
    correct for iterative tests.
    """
    n_trials = len(non_private_test_accuracies)
    assert len(non_private_test_accuracies) == n_trials
    assert len(n_trains) == n_trials
    assert len(n_tests) == n_trials

    sample_complexity_termination_sizes = []

    for i in range(n_trials):
        ret_value = binomial_optimal_risk_test(
            tau,
            non_private_train_accuracies[i],
            non_private_test_accuracies[i],
            n_trains[i],
            n_tests[i],
            epsilon_train,
            epsilon_test,
            delta_upper,
            delta_lower,
            correct_for_noise=correct_for_noise,
            include_accept=use_upper_bound,
            include_reject=use_lower_bound,
        )
        if ret_value == 1:
            sample_complexity_termination_sizes.append((n_trains[i], n_trains[i]))
        elif ret_value == -1:
            sample_complexity_termination_sizes.append((np.NINF, n_trains[i]))
        else:
            sample_complexity_termination_sizes.append((np.inf, n_trains[i]))
    return sample_complexity_termination_sizes


def run_accuracy_test(
    training_errors,
    test_errors,
    training_sizes,
    test_sizes,
    tau,
    train_epsilon,
    test_epsilon,
    delta_upper,
    delta_lower,
    correct_for_noise,
    use_upper_bound,
    use_lower_bound,
    block_size,
    profile_fn=non_adaptive_profile,
):
    """
    :param training_errors: List or vector of training errors.
    :param test_errors: List or vector of test errors.
    :param training_sizes: The sizes on which the models were trained.
    :param test_sizes: The sizes on which the models where tested.
    :param tau: The tau goal of the model.
    :param train_epsilon: The epsilon privacy budget to be applied to the
        training loss.
    :param test_epsilon: The epsilon privacy budget to be applied to the test
        loss.
    :param delta_upper: The confidence value to apply to the upper bound.
    :param delta_lower: The confidence value to apply to the lower bound.
    :param use_upper_bound: Boolean indicating if the upper bound should be
        tested.
    :param use_lower_bound: Boolean indicating if the lower bound should be
        tested.
    :param block_size: The block_size for multiple draws.
    """
    train_test_ratios = training_sizes / test_sizes
    if not np.allclose(
        train_test_ratios.min(), train_test_ratios.max(), rtol=0.01, atol=0.01
    ):
        raise ValueError(
            "Train-test ratio is not constant: [%f, %f]"
            % (train_test_ratios.min(), train_test_ratios.max())
        )
    if np.any(training_sizes <= test_sizes):
        raise ValueError("n_train should be > n_test")

    return profile_fn(
        training_errors,
        test_errors,
        training_sizes,
        test_sizes,
        tau,
        train_epsilon,
        test_epsilon,
        1.0,
        delta_upper,
        delta_lower,
        sanitize_mean_squared_error,
        correct_for_noise=correct_for_noise,
        use_upper_bound=use_upper_bound,
        use_lower_bound=use_lower_bound,
        block_size=block_size,
    )


def run_accuracy_tests(
    training_errors,
    test_errors,
    training_sizes,
    test_sizes,
    taus=[
        1.0e-3,
        1.1e-3,
        1.2e-3,
        1.3e-3,
        1.4e-3,
        1.5e-3,
        1.6e-3,
        1.7e-3,
        1.8e-3,
        1.9e-3,
        2.0e-3,
        2.1e-3,
        2.2e-3,
        2.3e-3,
        2.4e-3,
        2.5e-3,
        2.6e-3,
        2.7e-3,
        2.8e-3,
        2.9e-3,
        3.0e-3,
        3.1e-3,
        3.2e-3,
        3.3e-3,
        3.4e-3,
        3.5e-3,
    ],
    epsilons=[
        1.0e-5,
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1e-1,
        2.5e-1,
        0.5e-1,
        1.0,
        2.0,
        5.0,
        10.0,
        0.0,
    ],
    upper_confidence_deltas=[0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
    lower_confidence_deltas=[0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
    correct_for_noise=True,
    use_upper_bound=True,
    use_lower_bound=True,
    block_size=None,
    profile_fn=non_adaptive_profile,
):
    """
    Runs the accuracy test for a variety of different parameters.

    :param training_errors: The train error at each training size.
    :param test_errors: The test errors at each training size.
    :param training_sizes: The size training data used for each model.
    :param test_sizes: The size of test data used for each model.
    :param taus: A list of tau values to test.
    :param epsilons: A list of privacy budgets to use to sanitize the training
        and testing error.
    :param confidence_deltas: Confidence values to test for the upper and lower
        bound.
    :return: List of result dictionaries.
    """

    delta_values = set(lower_confidence_deltas).union(upper_confidence_deltas)
    results = []
    for tau in taus:
        for epsilon in epsilons:
            for delta_value in delta_values:
                sc_t_n = run_accuracy_test(
                    training_errors,
                    test_errors,
                    training_sizes,
                    test_sizes,
                    tau,
                    epsilon,
                    epsilon,
                    delta_value,
                    delta_value,
                    correct_for_noise,
                    use_upper_bound,
                    use_lower_bound,
                    block_size,
                    profile_fn,
                )
                for sc, t_n in sc_t_n:
                    results.append(
                        {
                            "tau": tau,
                            "train_epsilon": epsilon,
                            "test_epsilon": epsilon,
                            "epsilon": epsilon,
                            "delta_upper": delta_value,
                            "delta_lower": delta_value,
                            "sc": sc,
                            "termination_size": t_n,
                        }
                    )
    return results


def _append_row_values(dd, row, header):
    """
    Appends each value in the row to the default dict with the appropriate key.
    row[i] will be appended to dd[header[i]].

    :param dd: defaultdict of lists.
    :param row: The row to be appended.
    :param header: The header values to be used as keys.
    :return: The default dict with values appended.
    """
    for i, c in enumerate(header):
        dd[c].append(float(row[i]))
    return dd


def load_iteration_data_frames(fd, sample_size=1024):
    """
    Parses the fd which should contain logs from iterative_train_test_model
    yielding a dataframe for each iteration.

    :param fd: The file descriptor to read.
    :param sample_size: The sample of the file to read to sniff for a header.
    :yield: Performance dataframe.
    """
    header = [
        "iteration",
        "n_train",
        "n_test",
        "epsilon",
        "delta",
        "mse_train",
        "mse_test",
        "mse_eval",
    ]

    # Sniff for a header to be skipped.
    has_header = csv.Sniffer().has_header(fd.read(sample_size))
    fd.seek(0)

    csv_reader = csv.reader(fd, delimiter=" ")
    if has_header:
        next(csv_reader)

    iteration_to_data = {}

    for i, row in enumerate(csv_reader):
        if len(row) == 9 and i == 0:
            header.append("train_time")
        if len(row) != len(header):
            raise ValueError("%d != %d" % (len(row), len(header)))
        assert len(row) == len(header)
        iteration = int(row[0])
        if iteration not in iteration_to_data:
            iteration_to_data[iteration] = defaultdict(list)

        # Append to the data to the appropriate dictionary.
        _append_row_values(iteration_to_data[iteration], row, header)

    for iteration in sorted(iteration_to_data.keys()):
        df = pd.DataFrame(iteration_to_data[iteration])
        yield df


def run_accuracy_tests_from_logs(
    read_path,
    out_path,
    taus=[
        1.5e-3,
        1.6e-3,
        1.7e-3,
        1.8e-3,
        1.9e-3,
        2.0e-3,
        2.1e-3,
        2.2e-3,
        2.3e-3,
        2.4e-3,
        2.5e-3,
        2.6e-3,
        2.7e-3,
        2.8e-3,
        2.9e-3,
        3.0e-3,
        3.1e-3,
        3.2e-3,
        3.3e-3,
        3.4e-3,
        3.5e-3,
    ],
    epsilons=[
        1.0e-5,
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1e-1,
        2.5e-1,
        0.5e-1,
        1.0,
        2.0,
        5.0,
        10.0,
        0.0,
    ],
    confidence_deltas=[
        0.005,
        0.025,
        0.05,
    ],
    correct_for_noise=True,
    use_upper_bound=True,
    use_lower_bound=True,
    iterations=20,
    block_size=None,
    profile_fn=None,
):
    """
    Runs the accuracy test with from the logs ad read_path and writes the
    resulting log to out_path.

    :param read_path: Path from which the logs will be read.
    :param out_path: Path to which the final log will be written.
    """
    out_columns = [
        "iteration",
        "model_iteration",
        "epsilon",
        "delta",
        "tau",
        "train_epsilon",
        "test_epsilon",
        "delta_upper",
        "delta_lower",
        "sc",
        "termination_size",
    ]

    with open(read_path, "r") as read_fd:
        with gzip.open(out_path, "w", 6) as write_fd:
            csv_writer = csv.DictWriter(write_fd, fieldnames=out_columns)
            csv_writer.writeheader()
            for df in load_iteration_data_frames(read_fd):
                for i in range(iterations):
                    # epsilon = df.epsilon.values[0]
                    delta = df.delta.values[0]
                    iteration = df.iteration.values[0]
                    accuracy_results = run_accuracy_tests(
                        df.mse_train.values,
                        df.mse_test.values,
                        df.n_train.values,
                        df.n_test.values,
                        taus=taus,
                        epsilons=epsilons,
                        upper_confidence_deltas=confidence_deltas,
                        lower_confidence_deltas=confidence_deltas,
                        correct_for_noise=correct_for_noise,
                        use_upper_bound=use_upper_bound,
                        use_lower_bound=use_lower_bound,
                        block_size=block_size,
                        profile_fn=profile_fn,
                    )
                    for result in accuracy_results:
                        result["iteration"] = i
                        result["model_iteration"] = iteration
                        result["delta"] = delta
                        csv_writer.writerow(result)
            write_fd.flush()
