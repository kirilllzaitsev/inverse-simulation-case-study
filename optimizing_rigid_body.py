import taichi as ti
from utils import Printer, parse_common_sim_args, plot_losses

screen_res = 512

args = parse_common_sim_args()
ti.init(arch=ti.cpu, debug=args.debug, kernel_profiler=True)

sim_timesteps = args.sim_steps
sim_cycles = args.opt_steps
num_particles = 1
num_aux_particles = 1
particle_radius = 0.3
particle_radius_adj = particle_radius / 1

target_particle_idx = 0
particle_mass = 1
dt = 5e-3
damping = 0.99

gravity = ti.Vector.field(3, float, ())
elasticity_coef = 0.8
friction_coef = 0.01

v = ti.Vector.field(3, dtype=ti.f32, shape=(sim_timesteps, num_particles))
x = ti.Vector.field(3, dtype=ti.f32, shape=(sim_timesteps, num_particles))
aux_ball_rel_pos = ti.Vector([particle_radius * 1.1, 0, 0])
rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(sim_timesteps, num_particles))
omega = ti.Vector.field(3, dtype=ti.f32, shape=(sim_timesteps, num_particles))
omega_inc = ti.Vector.field(3, dtype=ti.f32, shape=(sim_timesteps, num_particles))
com = ti.Vector.field(3, dtype=ti.f32, shape=())
inertia = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)
inertia_global = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)
ang_momentum = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
lin_momentum = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
num_forces = 3
forces = ti.Vector.field(3, dtype=ti.f32, shape=num_forces)

plane_origin = ti.Vector([-0.9, -0.9, 0])
plane_end = ti.Vector([0.9, -0.9, 0])
plane_normal = ti.Vector([0.0, 1, 0])

# misc
eps = 1e-10
pos_vis_buffer = ti.Vector.field(3, float, shape=num_particles)
aux_pos_vis_buffer = ti.Vector.field(3, float, shape=num_aux_particles)
points_pos = ti.Vector.field(3, dtype=ti.f32, shape=2)
printer = Printer()
num_contacts = ti.field(dtype=ti.i32, shape=())

# optim
init_v = ti.Vector.field(3, dtype=float, shape=())
loss = ti.field(dtype=ti.f32, shape=())
target_ball_center = ti.Vector.field(3, dtype=float, shape=(1,))

ti.root.lazy_grad()

init_v[None] = ti.Vector([1, -1.5, 0.0])


@ti.kernel
def init_plane_points_pos(points: ti.template()):
    points[0] = plane_origin
    points[1] = plane_end


def compute_local_inertia():
    for i in range(num_particles):
        inertia[i] = ti.Matrix(
            [
                [2 / 5 * particle_mass * particle_radius**2, 0, 0],
                [0, 2 / 5 * particle_mass * particle_radius**2, 0],
                [0, 0, 2 / 5 * particle_mass * particle_radius**2],
            ]
        )


@ti.kernel
def compute_global_inertia(t: ti.i32, i: ti.i32):
    inertia_global[i] = rotation[t, i] @ inertia[i] @ rotation[t, i].transpose()


@ti.kernel
def init_pos():
    target_ball_center[0] = [0.85, 0.25, 0.0]
    init_ball_center_pos = ti.Vector([-0.3, -0.0, 0.0])
    x[0, target_particle_idx] = init_ball_center_pos
    v[0, target_particle_idx] = init_v[None]
    rotation[0, target_particle_idx] = ti.Matrix(
        [
            [ti.cos(1), -ti.sin(1), 0],
            [ti.sin(1), ti.cos(1), 0],
            [0, 0, 1],
        ]
    )
    gravity[None] = ti.Vector([0, -9.8, 0])


@ti.func
def to_world(t, i, rela_x):
    rela_pos = rotation[t, i] @ rela_x
    rela_v = omega[t, i] * ti.Vector([-rela_pos[1], rela_pos[0]])

    world_x = x[t, i] + rela_pos
    world_v = v[t, i] + rela_v

    return world_x, world_v, rela_pos


def handle_collision(t):
    for i in range(num_particles):
        normal_proj_prev_step = (x[t - 1, i] - plane_origin).dot(
            plane_normal
        ) - particle_radius_adj
        normal_proj_step = (x[t, i] - plane_origin).dot(
            plane_normal
        ) - particle_radius_adj
        if (normal_proj_prev_step < eps and normal_proj_step) > eps or (
            normal_proj_prev_step > eps and normal_proj_step < eps
        ):
            normal_v = (v[t, i].dot(plane_normal)) * plane_normal
            tangent_v = v[t, i] - normal_v + eps

            next_normal_v = -elasticity_coef * normal_v
            next_tangent_v = (
                tangent_v
                - min(friction_coef * normal_v.norm(), tangent_v.norm())
                * tangent_v.normalized()
            )
            v[t, i] = next_normal_v + next_tangent_v
            num_contacts[None] += 1


@ti.ad.grad_replaced
@ti.kernel
def normalize_rotation(t: ti.i32):
    for i in range(num_particles):
        U, sig, V = ti.svd(rotation[t, i])
        R = U @ V.transpose()
        rotation[t, i] = R


@ti.ad.grad_for(normalize_rotation)
def backward(t: ti.i32):
    return 1


def compute_lin_and_ang_momentum(t):
    for i in range(num_particles):
        for f_idx in range(num_forces):
            lin_momentum[i] += forces[f_idx]
            point_of_application = ti.Vector([1, 1, 1])
            moment_arm = point_of_application - x[t, i]
            ang_momentum[i] += moment_arm.cross(forces[f_idx])


def get_skew_symmetric_matrix(
    v: ti.types.vector(3, ti.f32)
) -> ti.types.matrix(3, 3, ti.f32):
    _temp_mat = ti.Matrix(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return _temp_mat


def reset_fields():
    loss[None] = 0
    num_contacts[None] = 0


def forward(do_visualize=True):
    init_plane_points_pos(points_pos)
    init_pos()
    reset_fields()
    compute_local_inertia()

    s = ti.exp(-dt * damping)

    final_step = sim_timesteps - 1
    for t in range(final_step):
        handle_gui_actions()
        compute_forces(t)
        for i in range(num_particles):
            compute_global_inertia(t, i)
        compute_lin_and_ang_momentum(t)
        for i in range(num_particles):
            # omega[t + 1, i] = s * omega[t, i] + omega_inc[t + 1, i]
            omega[t + 1, i] = (
                omega[t, i] + s * inertia_global[i].inverse() @ ang_momentum[i]
            )
            rotation[t + 1, i] = rotation[t, i] + dt * get_skew_symmetric_matrix(
                omega[t + 1, i]
            ) @ (rotation[t, i])
        advance(t)

        handle_collision(t + 1)
        normalize_rotation(t + 1)

        if do_visualize:
            render(t)

        if num_contacts[None] > 10.0:
            print(
                "Stop simulation - too many contacts. num_contacts[None]=",
                num_contacts[None],
            )
            final_step = t
            break

    if args.do_optim:
        compute_loss(final_step)
    return final_step


@ti.kernel
def advance(t: ti.i32):
    for i in range(num_particles):
        s = ti.exp(-dt * damping)
        v[t + 1, i] = s * v[t, i] + dt * (particle_mass**-1) * forces[i]
        x[t + 1, i] = x[t, i] + dt * v[t + 1, i]
        x[t + 1, i][1] = ti.max(
            x[t + 1, i][1],
            plane_origin[1] + particle_radius_adj,
        )


@ti.kernel
def compute_loss(t: ti.i32):
    # optimize for COM of the ball
    # moving_particle_pos = x[t, target_particle_idx]
    # optimize for a dot orbiting the ball
    # print("x[t, target_particle_idx]=", x[t, target_particle_idx])
    moving_particle_pos = (
        x[t, target_particle_idx] + rotation[t, target_particle_idx] @ aux_ball_rel_pos
    )
    print(
        "x[t, target_particle_idx] + rotation[t, target_particle_idx] @ aux_ball_rel_pos=",
        x[t, target_particle_idx] + rotation[t, target_particle_idx] @ aux_ball_rel_pos,
    )
    print("rotation[t, target_particle_idx]=", rotation[t, target_particle_idx])
    dist_loss = (moving_particle_pos - target_ball_center[0]).norm() ** 2

    loss[None] = 1.0 * dist_loss


def main():
    losses = []
    grads = []
    for i in range(sim_cycles):
        with ti.ad.Tape(loss=loss):
            is_vis_ter = i in [0, (sim_cycles - 1)]
            final_step = forward(
                do_visualize=(args.do_visualize and not args.do_optim)
                or (is_vis_ter and args.do_visualize)
            )
        losses.append(loss[None])
        grads.append(init_v.grad[None].norm())
        if args.do_optim:
            moving_ball_center = (
                pos_vis_buffer[target_particle_idx]
                + rotation[final_step, 0] @ aux_ball_rel_pos
            )
            printer.print_iter_stats(i, loss=loss[None], pos=moving_ball_center)
            update_inits()

    if args.do_optim:
        printer.print_final_optim_stats(
            # optimize for dot that orbits RB
            pos=x[final_step, target_particle_idx]
            + rotation[final_step, target_particle_idx] @ aux_ball_rel_pos,
            # optimize for RB's COM
            # pos=x[final_step, target_particle_idx],
            target_pos=target_ball_center[0],
        )
    if args.do_plot:
        plot_losses(losses, ylabel="Loss", fig_title="Rigid ball. Loss")
        plot_losses(
            grads,
            ylabel="V.grad.norm()",
            fig_title="Rigid ball. Velocity gradient (unclipped) norm",
        )


def update_inits():
    cum_v_grad = 0.0
    lr = 1
    for i in range(3):
        init_v.grad[None][i] = ti.min(ti.max(init_v.grad[None][i], -1), 1)
        init_v[None][i] -= lr * init_v.grad[None][i]
        cum_v_grad += abs(init_v.grad[None][i])
        # assert abs(init_v.grad[None][i]) < 100, "Exploding init_v.grad"
    printer.print_grad_stats(init_v=init_v)
    assert abs(cum_v_grad) > 0, "init_v.grad is zero"


def compute_forces(t):
    for i in range(num_particles):
        force = ti.Vector([0, 0, 0])
        force += gravity[None]
        coriolis_force = ti.Vector([0, 0, 0])
        centrifugal_force = ti.Vector([0, 0, 0])
        force += coriolis_force
        force += centrifugal_force
        forces[i] = force


def handle_gui_actions():
    if window is None:
        return
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            window.running = False
        elif e.key == "r":
            init_pos()
        elif e.key in ("a", ti.ui.LEFT):
            gravity[None] = [-1, 0, 0]
        elif e.key in ("d", ti.ui.RIGHT):
            gravity[None] = [+1, 0, 0]
        elif e.key in ("s", ti.ui.DOWN):
            gravity[None] = [0, -1, 0]
        elif e.key in ("w", ti.ui.UP):
            gravity[None] = [0, +1, 0]


window: ti.ui.Window = None
canvas: ti.ui.Canvas = None
scene: ti.ui.Scene = None
camera: ti.ui.Camera = None


def render(t, visualize=True):
    global window, canvas, scene, camera
    if visualize:
        if window is None:
            window = ti.ui.Window("Rigid ball", (screen_res, screen_res), vsync=True)
            canvas = window.get_canvas()
            canvas.set_background_color((1, 1, 1))
            scene = ti.ui.Scene()
            camera = ti.ui.Camera()

        aux_update_scene(t)

        render_ball(t)


def aux_update_scene(t):
    # what is wrong with this?
    # camera.position(0.5, -4.0, -1)
    # camera.lookat(0.5, 0.0, 0)
    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    canvas.scene(scene)

    window.show()


def render_ball(t):
    copy_to_vis(t)
    scene.particles(pos_vis_buffer, radius=particle_radius, color=(0.5, 0.42, 0.8))
    scene.particles(
        aux_pos_vis_buffer, radius=particle_radius * 0.1, color=(0.15, 0.142, 0.88)
    )
    scene.particles(
        target_ball_center, radius=particle_radius * 0.25, color=(0.05, 0.12, 0.18)
    )
    scene.lines(
        points_pos,
        width=3,
        color=(
            0,
            0,
            0,
        ),
    )


@ti.kernel
def copy_to_vis(t: ti.int32):
    for i in range(num_particles):
        for j in ti.static(range(3)):
            pos_vis_buffer[i][j] = x[t, i][j]
    for i in range(num_aux_particles):
        aux_particle_word_pos = (
            pos_vis_buffer[target_particle_idx] + rotation[t, 0] @ aux_ball_rel_pos
        )
        aux_pos_vis_buffer[i] = aux_particle_word_pos


if __name__ == "__main__":
    main()
