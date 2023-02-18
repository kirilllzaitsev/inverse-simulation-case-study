import sys
from time import sleep

import taichi as ti
from utils import parse_common_sim_args, plot_losses

ti.init(arch=ti.cpu, debug=True, flatten_if=True, verbose=False)

from taichi import _logging as logging

args = parse_common_sim_args()

if not args.info:
    logging.set_logging_level(ti.WARN)
if not args.trace:
    sys.tracebacklimit = 0

# Environment setup
dt = 4e-3
simulation_steps = args.sim_steps

gravity = ti.Vector([0, -9.8, 0])
drag_damping = 1
elasticity_coef = 0.6
friction_coef = 0.22

# Plane
plane_boundary = -1
plane_origin = ti.Vector([0.5, plane_boundary, 0.5])
plane_p1 = plane_origin - [0.1, plane_boundary, 0.1]
plane_p2 = plane_origin - [0.6, plane_boundary, 0.3]
# so that the normal points toward the ball
plane_normal = -plane_p1.cross(plane_p2)
plane_normal = plane_normal.normalized()

# Ball setup
n_balls = 2
ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
helper_ball_center = ti.Vector.field(3, dtype=float, shape=(simulation_steps))
contact_ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
target_ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
target_ball_idx = 0
contact_ball_idx = 1
particle_mass = 1.0

# Fields setup
x = ti.Vector.field(3, dtype=float)
v = ti.Vector.field(3, dtype=float)
init_x = ti.Vector.field(3, dtype=float)
init_v = ti.Vector.field(3, dtype=float)
impulse = ti.Vector.field(3, dtype=float)
force = ti.Vector.field(3, dtype=ti.f32, shape=(simulation_steps, n_balls))

# Optimization setup
optimization_steps = args.opt_steps
loss = ti.field(dtype=ti.float32)
# total number of contacts
num_contacts = ti.field(dtype=ti.float32)
# count iterations with sequential contact
n_sequential_contacts = ti.field(dtype=ti.float32)
speed_loss = ti.field(dtype=ti.float32)
lr = 0.5

# Fields finalization
ti.root.dense(ti.i, simulation_steps).dense(ti.j, n_balls).place(x, v, impulse)
ti.root.place(init_x, init_v)
ti.root.place(loss, num_contacts, n_sequential_contacts, speed_loss)
ti.root.lazy_grad()

# Init params setup
# collision with ground
init_x[None] = [0.75, -0.3, 0.0]
init_v[None] = [-1.5, -0.5, -1.5]


@ti.kernel
def init_points():
    x[0, contact_ball_idx] = [0.7, 0.7, -1.0]
    target_ball_center[0] = [-0.5, -0.5, 0.5]

    x[0, target_ball_idx] = init_x[None]
    v[0, target_ball_idx] = init_v[None]
    n_sequential_contacts[None] = 0


def forward(do_visualize=False):
    init_points()

    final_step = simulation_steps - 1
    for t in range(1, simulation_steps):
        substep(t)
        advance(t)

        # if not args.exclude_collisions:
        # handle_collisions(t, target_ball_idx)

        if do_visualize:
            visualize(t)

        if n_sequential_contacts[None] > 50.0:
                "Stop simulation. n_sequential_contacts[None]=",
                n_sequential_contacts[None],
            )
            final_step = t
            break
    if args.do_optim:
        compute_speed_loss(final_step)
        compute_loss(final_step)
    return final_step


@ti.kernel
def substep(t: ti.int32):
    for i in range(n_balls):
        imp = ti.Vector([0.0, 0.0, 0.0])
        compute_force(t - 1)
        imp += force[t - 1, i] * dt
        imp *= ti.exp(-drag_damping * dt)
        impulse[t, i] += imp


@ti.func
def compute_force(t):
    for i in range(n_balls):
        force[t, i] = gravity
        penalty_force = ti.Vector([0.0, 0.0, 0.0])
        normal_proj = (x[t, i] - plane_origin).dot(plane_normal)
        eps = 1e-10
        plane_dist = ti.abs(normal_proj) - ball_radius
        if normal_proj < 0:
            if plane_dist < 0:
                N = plane_normal.transpose() @ plane_normal
                T = ti.Matrix.identity(ti.f32, 3) - N
                t_ = T @ (v[t, i])
                kn = 5e3
                kt = kn * dt
                d = normal_proj
                fn = plane_normal * kn * ti.max(-d, 0) ** 0.5
                ft = (
                    -t_
                    / (t_.norm() + eps)
                    * friction_coef
                    * ti.tanh(kt * t_.norm() / (friction_coef * fn.norm()))
                )
                # ft = -t_ / (t_.norm() + eps) * ti.min(kt * t_.norm(), friction_coef * fn.norm())
                penalty_force = fn + ft
                force[t, i] += penalty_force
                # assert ft.norm() <= friction_coef * fn.norm(), 'friction'
                # assert False, 'debug'
                n_sequential_contacts[None] += 1
            else:
                n_sequential_contacts[None] = 0


@ti.kernel
def advance(t: ti.i32):
    for i in range(n_balls):
        v[t, i] = v[t - 1, i] + impulse[t, i]
        x[t, i] = x[t - 1, i] + dt * v[t, i]


@ti.kernel
def handle_collisions(t: ti.int32, i: ti.int32):
    detect_and_handle_plane_collision(t)


@ti.func
def detect_and_handle_plane_collision(t):
    next_tangent_v = ti.Vector([0.0, 0.0, 0.0])
    next_normal_v = ti.Vector([0.0, 0.0, 0.0])

    point_on_the_plane = plane_origin
    normal_proj = (x[t, target_ball_idx] - point_on_the_plane).dot(plane_normal)
    plane_dist = ti.abs(normal_proj)
    result = 0
    eps = 1e-8
    if normal_proj > 0:
        if plane_dist - ball_radius <= eps:
            result = 1

            normal_v = (v[t, target_ball_idx].dot(plane_normal)) * plane_normal
            tangent_v = v[t, target_ball_idx] - normal_v + eps

            next_normal_v = -elasticity_coef * normal_v
            next_tangent_v = (
                tangent_v
                - min(friction_coef * normal_v.norm(), tangent_v.norm())
                * tangent_v.normalized()
            )
            # simpler, less accurate alternative
            # next_tangent_v2 = (1 - friction_coef) * tangent_v
            v[t, target_ball_idx] = next_normal_v + next_tangent_v
            n_sequential_contacts[None] += 1
        else:
            n_sequential_contacts[None] = 0

    elif normal_proj == 0:
        result = 1
    else:
        # NOTE: since the plane is non-penetrating, this is extra for now
        if -plane_dist + ball_radius >= eps:
            result = 1
            assert 0, "Collision"

    return result


@ti.kernel
def compute_speed_loss(final_step: ti.i32):
    for i in range(final_step):
        if (v[i, target_ball_idx].norm()) > speed_loss[None]:
            speed_loss[None] = v[i, target_ball_idx].norm()


@ti.kernel
def compute_loss(t: ti.i32):
    contact_loss_scaler = 0.25
    dist_loss = (
        x[t, target_ball_idx] - target_ball_center[0]
    ).norm() ** 2 * contact_loss_scaler

    loss[None] = 0.9 * contact_loss_scaler * dist_loss + 0.1 * speed_loss[None]


from utils import Printer

printer = Printer()


def main():
    losses = []
    grads = []
    for i in range(optimization_steps):
        clear()
        with ti.ad.Tape(loss):
            is_vis_ter = i in [0, (optimization_steps - 1)]
            final_step = forward(
                do_visualize=(args.do_visualize and not args.do_optim)
                or (is_vis_ter and args.do_visualize)
            )

        losses.append(loss[None])
        grads.append(init_v.grad[None].norm())

        if args.do_optim:
            printer.print_iter_stats(
                i, loss=loss[None], pos=x[final_step, target_ball_idx]
            )
            update_inits()

        draw_trajectory(final_step)

    if args.do_optim:
        printer.print_final_optim_stats(
            pos=x[final_step, target_ball_idx], target_pos=target_ball_center[0]
        )
    if args.do_plot:
        plot_losses(losses, ylabel="Loss", fig_title="Rigid ball. Loss")
        plot_losses(
            grads,
            ylabel="V.grad.norm()",
            fig_title="Rigid ball. Velocity gradient (unclipped) norm",
        )


def draw_trajectory(final_step):
    helper_ball_center.fill([-1, -2, -1])
    for t in range(0, final_step, 5):
        helper_ball_center[t] = x[t, target_ball_idx]
    scene.particles(
        helper_ball_center, radius=ball_radius * 0.15, color=(0.25, 0.62, 0.18)
    )
    aux_update_scene()


def update_inits():
    cum_v_grad = 0.0
    for i in range(3):
        init_v.grad[None][i] = ti.min(ti.max(init_v.grad[None][i], -1), 1)
        init_v[None][i] -= lr * init_v.grad[None][i]
        cum_v_grad += init_v.grad[None][i]
        assert abs(init_v.grad[None][i]) < 100, "Exploding init_v.grad"
    assert abs(cum_v_grad) > 0, "init_v.grad is zero"
    printer.print_grad_stats(init_x=init_x, init_v=init_v)


@ti.kernel
def clear():
    for t, i in ti.ndrange(simulation_steps, n_balls):
        impulse[t, i] = ti.Vector([0.0, 0.0, 0.0])


window: ti.ui.Window = None
canvas: ti.ui.Canvas = None
scene: ti.ui.Scene = None
camera: ti.ui.Camera = None


def visualize(t):
    global window, canvas, scene, camera
    if window is None:
        window = ti.ui.Window("Rigid ball", (600, 600), vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color((1, 1, 1))
        scene = ti.ui.Scene()
        camera = ti.ui.make_camera()
    aux_update_scene()

    render_ball(t)


def aux_update_scene():
    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    canvas.scene(scene)
    window.show()


def render_ball(t):
    ball_center[0] = x[t, target_ball_idx]
    contact_ball_center[0] = x[t, contact_ball_idx]
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    scene.particles(
        target_ball_center, radius=ball_radius * 0.25, color=(0.05, 0.12, 0.18)
    )


if __name__ == "__main__":
    main()