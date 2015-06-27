import theano.tensor as T
import extheano

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

print power(range(10), 2)
print power(range(10), 4)
