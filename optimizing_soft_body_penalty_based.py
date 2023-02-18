
import taichi as ti
from utils import Printer, copy, parse_common_sim_args, plot_losses

args = parse_common_sim_args()
include_obstacle = args.include_obstacle
ti.init(arch=ti.cpu, debug=True)

# extra
n_sequential_contacts = ti.field(dtype=ti.f32, shape=())
n_vertices_at_rest = ti.field(dtype=ti.f32, shape=())

# optim
n_iters = args.opt_steps
n_timesteps = args.sim_steps
init_v = ti.Vector.field(3, dtype=float, shape=(), needs_grad=True)
init_v[None] = [1.8, -0.6, 2.0]
init_x_avg = ti.Vector.field(3, dtype=float, shape=(), needs_grad=True)

target_ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
target_ball_center[0] = [0.85, -0.25, 0.0]
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
x_avg = ti.Vector.field(3, dtype=float, shape=(), needs_grad=True)

# physics
N = 504
dt = 1e-3
dx = 0.005
rho = 4e1
NF = 2230  # number of faces
NV = 504  # number of vertices
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
h = 1
mu, lam = mu * h, lam * h
ball_pos, ball_radius = ti.Vector.field(3, dtype=ti.f32, shape=(1,)), 0.1
ball_pos[0] = [0.0, -0.8, 0.1]
gravity = ti.Vector([0.0, -20.0, 0.0])
damping = 12.5

force = ti.Vector.field(3, float, (n_timesteps, NV), needs_grad=False)
pos = ti.Vector.field(3, float, (n_timesteps, NV), needs_grad=True)
pos_draw = ti.Vector.field(3, float, NV, needs_grad=False)
vel = ti.Vector.field(3, float, (n_timesteps, NV), needs_grad=True)
f2v = ti.Vector.field(4, int, NF)  # ids of triangle/tetrahedra vertices
B = ti.Matrix.field(3, 3, float, NF)
F = ti.Matrix.field(3, 3, float, NF, needs_grad=True)  # deformation grad
V = ti.field(float, NF)
phi = ti.field(float, NF)  # potential energy of each face
U = ti.field(float, (), needs_grad=True)  # total potential energy


@ti.kernel
def init_pos():
    for k in range(NV):
        vel[0, k] = init_v[None]
    for i in range(NF):
        ia, ib, ic, id = f2v[i]
        a, b, c, d = pos[0, ia], pos[0, ib], pos[0, ic], pos[0, id]
        B_i_inv = ti.Matrix.cols([a - d, b - d, c - d])
        B[i] = B_i_inv.inverse()


@ti.kernel
def update_U(t: ti.i32):
    for i in range(NF):
        ia, ib, ic, id = f2v[i]
        a, b, c, d = pos[t, ia], pos[t, ib], pos[t, ic], pos[t, id]
        V[i] = abs(((a - d).cross(b - d)) @ (c - d))
        D_i = ti.Matrix.cols([a - d, b - d, c - d])
        F[i] = D_i @ B[i]
    for i in range(NF):
        F_i = F[i]
        log_J_i = ti.log(F_i.determinant())
        FF = F_i.transpose() @ F_i
        phi_nh = mu / 2 * (FF.trace() - 3) - mu * log_J_i + lam / 2 * log_J_i**2
        phi_i = phi_nh
        phi[i] = phi_i
        U[None] += V[i] * phi_i


rect_min_y = -1
rect_max_y = 1

plane_origin = ti.Vector([0.1, 0.1, 0.0])
plane_end = ti.Vector([0.9, 0.1, 0.0])
plane_normal = ti.Vector([0.0, 1.0, 0.0])
eps = 1e-10
friction_coef = 0.5
N = plane_normal.transpose() @ plane_normal


@ti.kernel
def compute_force(t: ti.i32):
    for i in range(NV):
        force[t, i] = gravity
        penalty_force = ti.Vector([0.0, 0.0, 0.0])
        normal_proj = (pos[t, i] - plane_origin).dot(plane_normal)
        if normal_proj < -1:
            T = ti.Matrix.identity(ti.f32, 3) - N
            t_ = T @ (vel[t, i])
            kn = 5e2
            kt = kn * dt
            d = normal_proj
            fn = plane_normal * kn * ti.max(-d, 0) ** 0.5
            ft = (
                -t_
                / (t_.norm() + eps)
                * friction_coef
                * ti.tanh(kt * t_.norm() / (friction_coef * fn.norm()))
            )
            # or
            # ft = -t_ / (t_.norm() + eps) * ti.min(kt * t_.norm(), friction_coef * fn.norm())
            penalty_force = fn + ft
            force[t, i] += penalty_force
            # assert ft.norm() <= friction_coef * fn.norm(), 'friction'


@ti.kernel
def advance(t: ti.i32):
    for i in range(NV):
        pos.grad[t, i] = ti.max(ti.min(pos.grad[t, i], 0.1), -0.1)
        acc = -pos.grad[t, i] / (rho * dx**2)
        vel[t + 1, i] = (vel[t, i] + dt * (acc + force[t, i])) * ti.exp(-dt * damping)
        for xx in ti.static(range(pos.n)):
            if abs(vel[t + 1, i][xx]) > 1000:
                print("acc=", acc)
                print("pos.grad[t, i]=", pos.grad[t, i])
                print("vel[t, i]=", vel[t, i])
                print("vel[t + 1, i][xx]=", vel[t + 1, i][xx])
                assert False

        pos[t + 1, i] = pos[t, i] + dt * vel[t + 1, i]

    for i in range(NV):
        eps = 1e-5
        if (
            abs(vel[t + 1, i][0]) < eps
            and abs(vel[t + 1, i][1]) < eps
            and abs(vel[t + 1, i][2]) < eps
        ):
            n_vertices_at_rest[None] += 1


@ti.kernel
def clear_grads(t: ti.i32):
    for xx in range(NV):
        pos.grad[t + 1, xx] = ti.Vector([0.0, 0.0, 0.0])
    for xx in ti.grouped(F):
        F.grad[xx] = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


window: ti.ui.Window = None
canvas: ti.ui.Canvas = None
scene: ti.ui.Scene = None
camera: ti.ui.Camera = None


def visualize(t):
    global window, canvas, scene, camera
    if window is None:
        window = ti.ui.Window("Soft ball", (600, 600), vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color((1, 1, 1))
        scene = ti.ui.Scene()
        camera = ti.ui.make_camera()
    aux_update_scene()

    render(t)


def render(t):
    scene.particles(pos_draw, radius=0.01, color=(0.95, 0.52, 0.18))
    scene.particles(target_ball_center, radius=ball_radius, color=(0.05, 0.12, 0.18))
    if include_obstacle:
        scene.particles(ball_pos, radius=ball_radius, color=(0.05, 0.92, 0.18))


def aux_update_scene():
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


def forward(do_visualize):
    init_pos()
    n_sequential_contacts[None] = 0

    final_step = n_timesteps - 1
    for t in range(n_timesteps - 1):
        handle_user_input()

        U.grad[None] = 1
        clear_grads(t)
        update_U(t)
        update_U.grad(t)

        compute_force(t)
        advance(t)

        upd_n_sequential_contacts()
        if n_sequential_contacts[None] > 15:
            print("Shape landed on the ground. Stop simulation")
            final_step = t
            break

        if do_visualize:
            copy(pos_draw, pos, timestep_to_copy=t)
            visualize(t)

    t = final_step - 1
    reset_fields()
    compute_x_avg(t)
    compute_loss(t)
    return final_step


@ti.kernel
def compute_loss(t: ti.i32):
    dist = (x_avg[None] - (target_ball_center[0])).norm()
    print("x_avg[None]=", x_avg[None])
    print("dist=", dist)
    loss[None] = dist**2


def reset_fields():
    loss[None] = 0.0
    x_avg[None] = [0.0, 0.0, 0.0]


@ti.kernel
def compute_x_avg(t: ti.i32):
    for i in range(NV):
        ti.atomic_add(x_avg[None], (1 / NV) * pos[t, i])


def upd_n_sequential_contacts():
    if n_vertices_at_rest[None] > N:
        print("One side collided with a plane")
        n_sequential_contacts[None] += 1
        n_vertices_at_rest[None] = 0
    else:
        n_sequential_contacts[None] = 0


printer = Printer()


def update_inits():
    cum_v_grad = 0.0
    lr = 0.5
    for i in range(3):
        init_v.grad[None][i] = ti.min(ti.max(init_v.grad[None][i], -1), 1)
        init_v[None][i] -= lr * init_v.grad[None][i]
        cum_v_grad += abs(init_v.grad[None][i])
        assert abs(init_v.grad[None][i]) < 100, "Exploding init_v.grad"
    printer.print_grad_stats(init_x=None, init_v=init_v)
    assert abs(cum_v_grad) > 0, "init_v.grad is zero"


def main():
    init_mesh()
    losses = []
    grads = []
    for iter in range(n_iters):
        with ti.ad.Tape(loss=loss, validation=False, clear_gradients=True):
            is_vis_ter = iter in [0, (n_iters - 1)]
            final_step = forward(
                do_visualize=(args.do_visualize and not args.do_optim)
                or (is_vis_ter and args.do_visualize)
            )

        if iter == 0:
            x_avg[None] = [0, 0, 0]
            compute_x_avg(final_step - 1)
            init_x_avg[None] = x_avg[None]
        losses.append(loss[None])
        grads.append(init_v.grad[None].norm())
        if args.do_optim:
            printer.print_iter_stats(iter, loss=loss[None], pos=x_avg[None])
            update_inits()
    if args.do_optim:
        printer.print_final_optim_stats(
            pos=x_avg[None], target_pos=target_ball_center[0]
        )
    if args.do_plot:
        plot_losses(losses, ylabel="Loss", fig_title="Soft ball. Loss")
        plot_losses(
            grads,
            ylabel="V.grad.norm()",
            fig_title="Soft ball. Velocity gradient (unclipped) norm",
        )


def init_mesh():
    import numpy as np

    vertices, faces = np.load("meshes/tetra_sphere_vertices.npy"), np.load(
        "meshes/tetra_sphere_faces.npy"
    )
    vertices[:, 1] = vertices[:, 1] - 0.5
    for i in range(NV):
        pos[0, i] = vertices[i]
    for i in range(NF):
        f2v[i] = faces[i]


def handle_user_input():
    ...


if __name__ == "__main__":
    main()
