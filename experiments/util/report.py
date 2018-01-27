import numpy as np


def _format(array):
    if isinstance(array[0], list):
        v = np.mean([np.array(a) for a in array], axis=0)
        return '[{}]'.format(', '.join(['{:.3f}'.format(a) for a in v]))
    else:
        return '{:.3f}'.format(np.mean(array))


def report(model, bgen, session):
    metrics = list(model.metrics.values())
    meal = (model.food(batch) for batch in bgen.gen_batches())
    accs = [session.run(metrics, feed_dict=m) for m in meal]
    accs = np.asarray(accs, dtype=object)
    reports = ['{}:{}'.format(name, _format(accs[:, c])) 
                for c, name in enumerate(model.metrics.keys())]

    return '\n'.join(reports)
