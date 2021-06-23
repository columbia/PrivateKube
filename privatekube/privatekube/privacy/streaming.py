# Implementation of Private and Continual Release of Statistics
# https://eprint.iacr.org/2010/076.pdf
# With a preprocessing trick to achieve user-level DP


import numpy as np

from .mechanisms import laplace_sanitize


class BoundedEventStreamingCounter:
    """
    Event-level DP counter for a stream of bits.
    Returns the number of 1 in the stream with the Binary Mechanism.
    """

    def __init__(self, epsilon, T):
        self.T = T
        self.n_bits = int(np.ceil(np.log2(T)))
        self.ε = epsilon / self.n_bits
        self.counts = {}
        self.current_index = 1
        self.α = {}
        self.nα = {}

        for i in range(self.n_bits):
            self.α[i] = 0
            self.nα[i] = 0

    def feed(self, s):
        """
        Extend the stream with a new chunk.
        """
        for t in range(self.current_index, self.current_index + len(s)):
            i = 0
            while bin_digit(t, i) == 0:
                i += 1

            # All the j < i are added to the new (larger) psum and erased
            # α[i] is the psum of the 2^i last items before t
            self.α[i] = sum([self.α[j] for j in range(i)]) + s[t - self.current_index]
            for j in range(i):
                self.α[j] = 0
                self.nα[j] = 0

            # We sanitize and release this psum
            self.nα[i] = laplace_sanitize(self.α[i], 1, self.ε)

            # Add psums of the remaining items
            self.counts[t] = sum(
                [self.nα[j] for j in range(self.n_bits) if bin_digit(t, j) == 1]
            )

            # Map to the closest credible value. No stream consistency here.
            self.counts[t] = max(0, int(self.counts[t]))
        self.current_index += len(s)

    def count(self, index=None):
        if index is None:
            index = self.current_index
        return self.counts[index]


class BoundedUserLevelStreamingCounter:
    """
    DP counter for the number of different users, with user-level DP.
    Useful for selecting non-empty blocks if the user IDs are increasing.
    """

    def __init__(self, epsilon, T):
        self.users = {}
        self.event_counter = BoundedEventStreamingCounter(epsilon, T)

    def feed(self, s):
        for u in s:
            # Preprocessing to create a stream with count sensitivity 1 per user
            if not u in self.users:
                # We discovered a new user
                self.event_counter.feed([1])
                self.users[u] = 1
            else:
                self.event_counter.feed([0])

    def count(self, index):
        """
        Returns the DP number of different users at time t
        """
        # This query is event-level DP on the preprocessed stream
        self.event_counter.count(index)


def bin_digit(t, i):
    """Returns the ith digit of t in binary, 0th digit is the least significant.
    >>> bin_digit(5,0)
    1
    >>> bin_digit(16,4)
    1
    """

    return (t >> i) % 2


def test_count(T, epsilon):
    s = np.random.randint(2, size=T)
    print(s)
    c = BoundedEventStreamingCounter(epsilon, T)
    c.feed(s)
    print(c.counts)


def test_user_count(T, epsilon, n_users):
    s = np.random.randint(n_users, size=T)
    print(s)
    c = BoundedUserLevelStreamingCounter(epsilon, T)
    c.feed(s)
    print(c.event_counter.counts)


if __name__ == "__main__":
    # test_count(10000, 0.01)
    test_user_count(10_000, 0.1, 1_000)