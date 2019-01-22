import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os


def predict_table(model, gen):
    """Uses a Keras model and generator to predict classes

    :param model: a Keras model compatible with `gen`
    :param gen: a Keras batch generator compatible with the model
    :returns: a pandas DataFrame with fields `d`irector,
    `f`ilename (within folder), and for each class, the actual and
    predicted values.
    :rtype: pandas DataFrame

    """
    gen.reset()
    rows = []
    seen = set()
    i2c = {i: c for c, i in gen.class_indices.items()}
    for imgs, tags in gen:
        # print('batch', gen.batch_index)
        if gen.batch_index in seen:
            break
        idx = (gen.batch_index - 1) * gen.batch_size
        # print(gen.filenames[idx:idx + gen.batch_size])
        outs = model.predict(imgs)
        for i in range(imgs.shape[0]):
            row = {
                'd': gen.directory,
                'f': gen.filenames[idx+i]}
            for tag in i2c:
                tc = i2c[tag]
                av = tags[i, tag]
                pv = outs[i, tag]
                row['a_%s' % tc] = av
                row['p_%s' % tc] = pv
            rows.append(row)
        seen.add(gen.batch_index)
    return rows


def abs_p_diff(predict_table, categA='sandwich', categB='sushi'):
    """Calculates the absolute distance between two category predictions

    :param predict_table: as returned by `predict_table`
    :param categA: the first of two categories to compare
    :param categB: the second of two categoreis to compare
    :returns: series with the absolute difference between the predictions
    :rtype: pandas Series

    """
    return abs(predict_table['p_%s' % categA] - predict_table['p_%s' % categB])


def plot_top_candidates(predict_table, k=5, p=0,
                        sort_field='sanushi',
                        categA='sandwich', categB='sushi'):
    """Plots `k` example candidate images

    :param predict_table: as returned by `predict_table` and enriched using
    `abs_p_diff`
    :param k: number of images to show
    :param p: "page" of results to show
    :param sort_field: the field to sort by, should be a numerical field
    :param categA: name of a category
    :param categB: name of a second category
    :returns: nothing, creates a plot
    :rtype: None

    """
    assert type(p) == int
    assert type(k) == int
    start = p*k
    end = (p+1)*k
    for index, candidate in predict_table.sort_values(
            by=[sort_field],
            ascending=True)[start:end].iterrows():
        # print('type candidate', type(candidate))
        d, f = candidate['d'], candidate['f']
        path = os.path.join(d, f)
        img = mpimg.imread(path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_axis_off()
        psa, psb = candidate['p_%s' % categA], candidate['p_%s' % categB]
        ax.set_title('%s? %s=%.3f %s=%.3f' % (sort_field,
                                              categA, psa,
                                              categB, psb))
