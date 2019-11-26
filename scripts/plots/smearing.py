#!/usr/bin/env python
from math import sqrt, exp, pi

mass_points = [251, 260, 280, 300, 325, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000]
width_low = [None, 9.5, 8.0, 8.5, 12.0, 12.5, 20.0, 30.0, 40.5, 46.0, 56.0, 76.5, 107.5, 129.0, 160.5]
width_high = [2.5, 4.0, 4.5, 7.0, 8.0, 11.0, 20.0, 25.0, 39.5, 55.5, 71.5, 104.5, 183.5, None, None]

print([m1 - m0 for m1, m0 in zip(mass_points[1:], mass_points[:-1])])

assert len(mass_points) == len(width_low)
assert len(mass_points) == len(width_high)


def norm(x):
    return exp(-x**2 / 2) / sqrt(2 * pi)

kernel = [norm(-1), norm(0), norm(1)]
kernel_lhs = [norm(-1), norm(0), 0]
kernel_rhs = [0, norm(0), norm(1)]

for i, (m, wl, wh) in enumerate(zip(mass_points, width_low, width_high)):
    print("Mass point: " + str(m))
    right_sided = wh is not None
    left_sided = wl is not None

    assert right_sided or left_sided

    k = None
    if right_sided:
        k = kernel_rhs
    elif left_sided:
        k = kernel_lhs
    else:
        k = kernel

    delta_up = mass_points[i + 1] - m if right_sided else None
    delta_down = m - mass_points[i - 1] if left_sided else None

    smearing_kernel = [(m, norm(0))]

    if delta_up:
        steps = int(delta_up / wh / 2.0)
        print(steps * wh)
        step_width = wh

        for step in range(1, steps + 1):
            smearing_kernel = smearing_kernel + [(m + step * step_width, norm(step / float(steps)))]


    if delta_down:
        steps = int(delta_down / wl / 2.0)
        print(steps * wl)
        step_width = wl

        for step in range(1, steps + 1):
            smearing_kernel = [(m - step * step_width, norm(step / float(steps)))] + smearing_kernel


    print(smearing_kernel)


