import argparse
import os
import time
import warnings
from multiprocessing import Pool
import numpy as np
import tqdm

# Solver-specific imports
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from concorde.tsp import TSPSolver as ConcordeSolver
from lkh import solve as lkh_solve  # Make sure the installed python-lkh supports parameters such as max_trials, runs and time_limit
# If you prefer tsplib95 + the LKH binary, see the optional implementation in the comments below
import tsplib95

warnings.filterwarnings("ignore")

def solve_tsp_ortools(nodes_coord: np.ndarray) -> list:
    """Solves a TSP instance using Google OR-Tools."""
    # OR-Tools requires integer distances, so coordinates are scaled up here
    coords_scaled = (nodes_coord * 10000).astype(int)
    data = {'locations': coords_scaled, 'num_vehicles': 1, 'depot': 0}
    manager = pywrapcp.RoutingIndexManager(len(data['locations']), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(np.linalg.norm(data['locations'][from_node] - data['locations'][to_node]))

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return tour
    return None


def solve_tsp_concorde(nodes_coord: np.ndarray) -> list:
    """Solves a TSP instance using Concorde."""
    # Scale [0,1) float coordinates to integers so that Concorde computes distances accurately
    scale = 1000000
    x_ints = (nodes_coord[:, 0] * scale).astype(int)
    y_ints = (nodes_coord[:, 1] * scale).astype(int)

    try:
        solver = ConcordeSolver.from_data(x_ints, y_ints, norm="EUC_2D")
        # Give Concorde a 60 s time limit; adjust according to the problem size
        solution = solver.solve(verbose=False, time_bound=60.0)
        if solution.success:
            # Return a 0-based tour list
            return solution.tour.tolist()
    except Exception as e:
        print(f"Concorde failed with error: {e}")
        return None
    return None


# def solve_tsp_lkh(nodes_coord: np.ndarray) -> list:
#     """Solves a TSP instance using LKH (python-lkh interface)."""
#     # Scale up to large integers as well
#     scale = 1000000
#     scaled_coords = (nodes_coord * scale).astype(int)

#     try:
#         # Pass more thorough search parameters: max_trials, runs and time_limit
#         # Pass them directly if your python-lkh version supports them; otherwise see the tsplib95-based .tsp approach below.
#         tour_indices = lkh_solve(
#             scaled_coords,
#             max_trials=1000,
#             runs=5,
#             time_limit=60.0
#         )
#         if tour_indices:
#             # Does lkh_solve return a 1-based or 0-based tour? Assumed 1-based here; subtract 1 depending on your version
#             # If your version returns 0-based tours, remove the subtraction below
#             return [i - 1 for i in tour_indices]
#     except Exception as e:
#         print(f"LKH failed with error: {e}")
#         return None
#     return None

def solve_tsp_lkh(nodes_coord: np.ndarray) -> list:
    """Solve TSP by creating a tsplib95 problem in memory and calling LKH."""
    scale = 1000000
    scaled = (nodes_coord * scale).astype(int)
    n = len(scaled)

    problem = tsplib95.models.StandardProblem()
    problem.name = "TSP"
    problem.type = "TSP"
    problem.dimension = n
    problem.edge_weight_type = "EUC_2D"
    problem.node_coords = {i + 1: tuple(scaled[i]) for i in range(n)}

    try:
        # lkh_solve returns a list containing the tour, e.g., [[1, 5, ...]]
        solution = lkh_solve(
            "LKH-3.0.13/LKH",
            problem=problem,
            max_trials=1000,
            runs=5,
            time_limit=10.0
        )
        # We need to access the first element to get the tour itself.
        # Then convert the 1-based indices to 0-based.
        return [i - 1 for i in solution[0]]  # <-- **CHANGE tour to solution[0]**
    except Exception as e:
        print(f"LKH failed with error: {e}")
        return None


# --- Data Processing Wrapper ---

SOLVER_MAP = {
    "ortools": solve_tsp_ortools,
    "concorde": solve_tsp_concorde,
    "lkh": solve_tsp_lkh,
    # If using tsplib95 + the LKH binary, replace with:
    # "lkh": lambda coords: solve_tsp_lkh_tsplib(coords, lkh_path="LKH-3.0.6/LKH")
}


def process_instance_wrapper(args):
    """
    Wrapper function for multiprocessing.
    It takes coordinates and a solver function, returns the solved tour coordinates.
    """
    instance_coords, solver_func = args
    num_nodes = len(instance_coords)

    # Solve the TSP to get the tour (a list of indices)
    tour = solver_func(instance_coords)

    # Validate and reorder coordinates
    if tour is not None and len(tour) == num_nodes:
        # Ensure the tour contains all nodes exactly once before reordering
        if np.array_equal(np.sort(tour), np.arange(num_nodes)):
            return instance_coords[tour, :]
    return None


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['generate', 'solve'],
                        help="'generate' to create map files, 'solve' to solve them.")
    parser.add_argument("--solver", type=str, default='ortools', choices=SOLVER_MAP.keys(),
                        help="Solver to use when mode is 'solve'.")
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count() or 1, 4),
                        help="Parallel worker count; should not exceed the number of physical CPU cores.")
    parser.add_argument("--batch_size", type=int, default=256)
    opts = parser.parse_args()

    # Define file paths based on parameters
    output_dir = "./data"

    os.makedirs(output_dir, exist_ok=True)
    map_filename = os.path.join(output_dir, f"tsp_maps_n{opts.num_nodes}_num{opts.num_samples}_s{opts.seed}.npz")

    # ===== MAP GENERATION MODE =====
    if opts.mode == 'generate':
        print("=" * 50)
        print(f"MODE: GENERATE. Creating {opts.num_samples} random maps.")
        print(f"Map file: {map_filename}")
        print("=" * 50)

        if os.path.exists(map_filename):
            print("Map file already exists. Skipping generation.")
        else:
            rng = np.random.RandomState(opts.seed)
            all_coords = rng.rand(opts.num_samples, opts.num_nodes, 2)
            np.savez_compressed(map_filename, maps=all_coords)
            print(f"Successfully generated and saved {opts.num_samples} maps.")

    # ===== SOLVING MODE =====
    elif opts.mode == 'solve':
        if not os.path.exists(map_filename):
            print(f"Error: Map file not found at {map_filename}")
            print("Please run the script with '--mode generate' first.")
            exit()

        solution_filename = os.path.join(
            output_dir,
            f"tsp_solutions_n{opts.num_nodes}_s{opts.seed}_solver_{opts.solver}.npz"
        )

        print("=" * 50)
        print(f"MODE: SOLVE. Using solver: {opts.solver.upper()}")
        print(f"Loading maps from: {map_filename}")
        print(f"Outputting solutions to: {solution_filename}")
        print("=" * 50)

        loaded_maps = np.load(map_filename)['maps']
        num_maps_to_solve = min(opts.num_samples, len(loaded_maps))
        maps_to_solve = loaded_maps[:num_maps_to_solve]

        start_time = time.time()
        solver_function = SOLVER_MAP[opts.solver]

        # Prepare tasks for multiprocessing
        tasks = [(maps_to_solve[i], solver_function) for i in range(num_maps_to_solve)]

        all_results = []
        # Cap parallelism at the number of CPU cores to avoid excessive context switching
        num_workers = min(opts.num_workers, os.cpu_count() or 1)
        # with Pool(num_workers) as pool:
        #     with tqdm.tqdm(total=len(tasks), desc=f"Solving with {opts.solver}") as pbar:
        #         for result in pool.imap_unordered(process_instance_wrapper, tasks):
        #             if result is not None:
        #                 all_results.append(result)
        #             pbar.update(1)
        print(f"Submitting {len(tasks)} tasks to {num_workers} workers...")
        with Pool(num_workers) as pool:
            # pool.map preserves the order of the results with respect to the input tasks
            # It waits for all tasks to finish and returns a list
            all_results_in_order = list(tqdm.tqdm(pool.imap(process_instance_wrapper, tasks), total=len(tasks), desc=f"Solving with {opts.solver}"))
        
        # Filter out failed results (None)
        all_results = [res for res in all_results_in_order if res is not None]

        if all_results:
            final_data = np.stack(all_results, axis=0)
            np.savez_compressed(solution_filename, locs=final_data)

            total_time = time.time() - start_time
            print("\n" + "=" * 50)
            print(f"Successfully solved and saved {final_data.shape[0]} samples.")
            print(f"Output file: {solution_filename}")
            print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print("=" * 50)
        else:
            print("\nNo valid solutions were generated.")