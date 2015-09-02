from collections import defaultdict
from itertools import chain

def MapReduce(mapper, reducer, input_data):
    """
    Simple implementation of map reducer
    :param mapper: a mapper generator must yield (key, value)
    :param reducer: a reducer generator
    :return: processed data

    # Character count example using MaoReduce
    def mapper(x):
        yield (x, 1)

    def reducer(k, v):
        yield (k, sum(v))

    print MapReduce(mapper, reducer, "TestInputString")
    """
    # Map data
    if mapper is None:
        mapped_data = input_data
    else:
        mapped_data = chain(*(mapper(x) for x in input_data))
    #
    if reducer is None:
        return list(mapped_data)
    #
    # Shuffle data
    shuffled_data = defaultdict(list)
    for key, value in mapped_data:
        shuffled_data[key].append(value)
    #
    # Reduce data
    reduced_data = (reducer(k, v) for k, v in shuffled_data.items())
    reduced_data = list(chain(*reduced_data))
    #
    return reduced_data
