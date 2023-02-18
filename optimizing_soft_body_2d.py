
import taichi as ti
from utils import copy, parse_common_sim_args, plot_losses

args = parse_common_sim_args()
ti.init(arch=ti.gpu, debug=True)

# extra
n_sequential_contacts = ti.field(dtype=ti.f32, shape=())
n_vertices_at_rest = ti.field(dtype=ti.f32, shape=())

# optim
n_iters = args.opt_steps
n_timesteps = args.sim_steps
init_v = ti.Vector.field(2, dtype=float, shape=(), needs_grad=True)
init_v[None] = [-0.8, 0.6]
init_x_avg = ti.Vector.field(2, dtype=float, shape=(), needs_grad=True)

target_ball_center = ti.Vector.field(2, dtype=float, shape=(1,))
target_ball_center[0] = [0.85, 0.25]
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
x_avg = ti.Vector.field(2, dtype=float, shape=(), needs_grad=True)

# physics
N = 4
dt = 1e-3
dx = 1 / N
rho = 4e1
NF = 2 * N**2  # number of faces
NV = (N + 1) ** 2  # number of vertices
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
h = 1
mu, lam = mu * h, lam * h
ball_pos, ball_radius = ti.Vector([0.2, 0.0]), 0.2
gravity = ti.Vector([0, -40])
damping = 12.5

pos = ti.Vector.field(2, float, (n_timesteps, NV), needs_grad=True)
pos_draw = ti.Vector.field(2, float, NV, needs_grad=False)
vel = ti.Vector.field(2, float, (n_timesteps, NV), needs_grad=True)
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)
F = ti.Matrix.field(2, 2, float, NF, needs_grad=True)  # deformation grad
V = ti.field(float, NF)
phi = ti.field(float, NF)  # potential energy of each face (Neo-Hookean)
U = ti.field(float, (), needs_grad=True)  # total potential energy


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[0, k] = ti.Vector([i, j]) / N * 0.25 + ti.Vector([0.15, 0.15])
        vel[0, k] = init_v[None]
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[0, ia], pos[0, ib], pos[0, ic]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i_inv.inverse()


@ti.kernel
def update_U(t: ti.i32):
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[t, ia], pos[t, ib], pos[t, ic]
        # print(a,b,c)
        V[i] = abs((a - c).cross(b - c))
        D_i = ti.Matrix.cols([a - c, b - c])
        F[i] = D_i @ B[i]
    for i in range(NF):
        F_i = F[i]
        log_J_i = ti.log(F_i.determinant())
        FF = F_i.transpose() @ F_i
        phi_nh = mu / 2 * (FF.trace() - 2) - mu * log_J_i + lam / 2 * log_J_i**2
        phi_i = phi_nh
        phi[i] = phi_i
        # other material models
        # phi_stvk = (
        #     lam / 2 * (0.5 * (FF - ti.one(FF))).trace()**2+
        #     mu * (FF**2).trace()
        # )
        # shape is collapsed to a negligible volume of moving particles
        # phi_i=phi_stvk
        # phi_stvk_vol = (
        #     lam / 2 * (FF.determinant()-1)**2 +
        #     mu * (FF**2).trace()
        # )
        # shape is collapsed, yet the effect is less compared to vanilla StVK
        # phi_i=phi_stvk_vol
        # e = 0.5*(F_i+F_i.transpose())-ti.one(F_i)
        # phi_lin = (
        #     lam / 2 * (e.trace())**2 +
        #     mu * (e**2).trace()
        # )
        # shape is overall consistent, albeit corners are sinked
        # phi_i=phi_lin
        # phi_corot = (
        #     lam / 2 * (FF.determinant()-1)**2 +
        #     mu * (FF**2).trace()
        # )
        # phi_i=phi_corot
        U[None] += V[i] * phi_i


@ti.kernel
def advance(t: ti.i32):
    for i in range(NV):
        acc = -pos.grad[t, i] / (rho * dx**2)
        vel[t + 1, i] = (vel[t, i] + dt * (acc + gravity)) * ti.exp(-dt * damping)
        for xx in ti.static(range(2)):
            if abs(vel[t + 1, i][xx]) > 1000:
                print("acc=", acc)
                print("pos.grad[t, i]=", pos.grad[t, i])
                print("vel[t, i]=", vel[t, i])
                print("vel[t + 1, i][xx]=", vel[t + 1, i][xx])
                assert False
    for i in range(NV):
        cond = (pos[t, i] < 0) & (vel[t + 1, i] < 0) | (pos[t, i] > 1) & (
            vel[t + 1, i] > 0
        )
        for j in ti.static(range(pos.n)):
            if cond[j]:
                vel[t + 1, i][j] += -vel[t + 1, i][j]  # global rule 2

        pos[t + 1, i] = pos[t, i] + dt * vel[t + 1, i]

    for i in range(NV):
        eps = 1e-5
        if abs(vel[t + 1, i][0]) < eps and abs(vel[t + 1, i][1]) < eps:
            n_vertices_at_rest[None] += 1


@ti.kernel
def clear_grads(t: ti.i32):
    for xx in range(NV):
        pos.grad[t + 1, xx] = ti.Vector([0.0, 0.0])
    for xx in ti.grouped(F):
        F.grad[xx] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])


class Printer:
    def print_iter_stats(self, i):
        print("#" * 10)
        print("Iter=", i, "Loss=", loss[None])
        print("x_avg[None]=", x_avg[None])
        print("#" * 10)

    def print_optim_stats(self):
        print("#" * 10)
        print("FINAL optimization results")
        print("(iter 0) x_avg[None]=", init_x_avg[None])
        print("x_avg[None]=", x_avg[None])
        print("target_ball_center[0]=", target_ball_center[0])

    def print_grad_stats(self):
        print("init_v[None]=", init_v[None])
        print("init_v.grad[None]=", init_v.grad[None])


@ti.kernel
def compute_loss(t: ti.i32):
    dist = (x_avg[None] - (target_ball_center[0])).norm()
    print("x_avg[None]=", x_avg[None])
    print("dist=", dist)
    loss[None] = dist**2


@ti.kernel
def compute_x_avg(t: ti.i32):
    for i in range(NV):
        ti.atomic_add(x_avg[None], (1 / NV) * pos[t, i])


printer = Printer()

gui = ti.GUI("FEM99", background_color=0xFFFFFF)


def render(t):
    for i in range(NV):
        gui.circle(pos_draw[i], radius=3, color=0x999)
    gui.circle(target_ball_center[0], radius=ball_radius * 50, color=0x999)
    # gui.circle(ball_pos, radius=ball_radius * 512, color=0x999)
    gui.show()


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

        advance(t)

        upd_n_sequential_contacts()
        if n_sequential_contacts[None] > 15:
            print("Shape landed on the ground. Stop simulation")
            final_step = t
            break

        if do_visualize:
            copy(pos_draw, pos, timestep_to_copy=t)
            render(t)

    t = final_step - 1
    reset_fields()
    compute_x_avg(t)
    compute_loss(t)
    return final_step


def upd_n_sequential_contacts():
    if n_vertices_at_rest[None] > N:
        print("One side collided with a plane")
        n_sequential_contacts[None] += 1
        n_vertices_at_rest[None] = 0
    else:
        n_sequential_contacts[None] = 0


def main():
    init_mesh()
    losses = []
    grads = []
    for iter in range(n_iters):
        with ti.ad.Tape(loss=loss, validation=False, clear_gradients=True):
            is_final_iter = iter == (n_iters - 1)
            do_visualize = (args.do_visualize and not args.do_optim) or (
                is_final_iter and args.do_visualize
            )

            final_step = forward(do_visualize=do_visualize)

        if iter == 0:
            x_avg[None] = [0, 0]
            compute_x_avg(final_step - 1)
            init_x_avg[None] = x_avg[None]
        printer.print_iter_stats(iter)
        losses.append(loss[None])
        grads.append(init_v.grad[None].norm())
        update_inits()
    printer.print_optim_stats()
    if args.do_plot:
        plot_losses(losses, ylabel="Loss", fig_title="Soft rectangle. Loss")
        plot_losses(
            grads,
            ylabel="V.grad.norm()",
            fig_title="Soft rectangle. Velocity gradient (unclipped) norm",
        )


def update_inits():
    cum_v_grad = 0.0
    lr = 10
    for i in range(2):
        init_v.grad[None][i] = ti.min(ti.max(init_v.grad[None][i], -2), 2)
        init_v[None][i] -= lr * init_v.grad[None][i]
        cum_v_grad += init_v.grad[None][i]
        assert abs(init_v.grad[None][i]) < 100, "Exploding init_v.grad"
    printer.print_grad_stats()
    assert abs(cum_v_grad) > 0, "init_v.grad is zero"


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]


def reset_fields():
    loss[None] = 0
    x_avg[None] = [0, 0]


def handle_user_input():
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == "r":
            init_pos()


if __name__ == "__main__":
    main()
