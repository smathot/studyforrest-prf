#!/usr/bin/env python3
# coding=utf-8

from matplotlib import pyplot as plt


def circle_plot(prf_map, lim=(160, 228), figsize=(8, 8)):

    x, y, sd, err = prf_map.params
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, aspect='equal')
    for x, y, sd in zip(x, y, sd):
        ax.add_artist(
            plt.Circle(
                xy=(x, y),
                radius=sd / 4,
                alpha=.05,
                fill=False
            )
        )
    plt.xlim(0, 160)
    plt.ylim(0, 128)
    plt.show()
