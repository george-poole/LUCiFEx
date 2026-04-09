from lucifex.sim import parallel_run

from .U20_simulation import create_simulation


def performance_report(
    t_exec: float,
    n_proc: int | None,
    n_stop: int,
    store: int,
    Nx: int,
    Ny: int,
) -> str:
    s = "\n".join(
        [
            f'Nx = {Nx}',
            f'Ny = {Ny}',
            f'store = {store}',
            f'n_stop = {n_stop}',
            f'n_proc = {n_proc}',
            f't_exec = {t_exec}',
        ]
    )
    return s


if __name__ == "__main__":
    STORE = 1
    N_PROC = 4
    N_STOP = 200
    STORE = 1
    NX = 200
    NY = 200
    DT = 0.01
    D_OPTS = (0.1, 1.0, 5.0, 10.0)

    create_sim = create_simulation(store_delta=STORE)

    parallel_run(
        create_sim, N_PROC, N_STOP, return_as='grid',
    )(NX, NY, DT)(d=D_OPTS)

