import argparse
import logging
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

logging.getLogger().setLevel(logging.INFO)


def plot_losses(
    losses: list,
    fig_title: str = "Figure",
    ylabel: str = "Objective",
    do_save_fig: bool = False,
):
    plt.plot(np.arange(len(losses)), losses)
    fig = plt.gcf()
    fig.set_size_inches(5, 3)
    plt.title(fig_title)
    plt.ylabel(ylabel)
    plt.xlabel("Optimization epoch")
    plt.tight_layout()
    if do_save_fig:
        plt.savefig(fig_title)
        print(f"Saved figure to {fig_title}")
    else:
        plt.show()


def parse_common_sim_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", "-i", action="store_true")
    parser.add_argument("--no-debug", "-nd", action="store_true")
    parser.add_argument("--trace", "-t", action="store_true")
    parser.add_argument("--do-plot", "-dp", action="store_true")
    parser.add_argument("--do-optim", "-do", action="store_true")
    parser.add_argument("--do-visualize", "-dv", action="store_true")
    parser.add_argument("--exclude-collisions", "-ec", action="store_true")
    parser.add_argument("--include_obstacle", "-io", action="store_true")
    parser.add_argument("--opt-steps", "-os", type=int, default=1)
    parser.add_argument("--sim-steps", "-ss", type=int, default=100)
    args, _ = parser.parse_known_args()
    args.debug = args.no_debug is False
    return args


class Printer:
    def print_iter_stats(self, i: int, loss: t.Any, pos: t.Any):
        print("#" * 10)
        print("Iter=", i, "Loss=", loss)
        print("pos=", pos)
        print("#" * 10)

    def print_final_optim_stats(self, pos: t.Any, target_pos: t.Any):
        print("#" * 10)
        print("FINAL optimization results\npos=", pos)
        print("target_pos=", target_pos)

    def print_grad_stats(self, **grads):
        """Accepts only taichi fields of shape=()"""
        for k, v in grads.items():
            try:
                print(f"{k}={v[None]}")
                print(f"{k}.grad={v.grad[None]}")
            except TypeError:
                print(f"{k}={v}")


@ti.kernel
def copy(f: ti.template(), f_with_timestep: ti.template(), timestep_to_copy: ti.i32):
    for i, j in f_with_timestep:
        if i == timestep_to_copy:
            f[j] = f_with_timestep[i, j]
