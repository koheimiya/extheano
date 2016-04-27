from sys import path
path.insert(0, '.')

import theano.tensor as T
import extheano


def test_power():
    # Example usage of extheano.scan
    @extheano.jit
    def power(A, k):
        # Again, no updates.
        result = extheano.scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=T.ones_like(A),
            non_sequences=A,
            n_steps=k
        )

        # We only care about A**k, the last element.
        return result[-1]

    x = list(range(10))
    assert list(power(x, 2)) == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    assert list(power(x, 4)) == [0, 1, 16, 81, 256, 625, 1296, 2401, 4096, 6561]
