"""
Utilities in the grid2op and gym convertion.
"""

import logging

import grid2op


def calculate_action_topology_spaces(environments: list[str]) -> None:
    """
    Function prints the number of legal actions and topologies without symmetries,
    following Subrahamian et al. (2020)'s implementation and following a proposed implementation.
    """
    # environments = ["rte_case5_example", "rte_case14_realistic", "l2rpn_wcci_2022"]

    for env_name in environments:
        logging.info(env_name)
        env = grid2op.make(env_name, test=True)

        nr_substations = len(env.sub_info)

        logging.info("no symmetries")
        action_space = 0
        possible_topologies = 1
        for sub in range(nr_substations):
            nr_elements = len(
                env.observation_space.get_obj_substations(substation_id=sub)
            )
            alpha = 2 ** (nr_elements - 1)
            action_space += alpha
            possible_topologies *= max(alpha, 1)

        logging.info(f"actions {action_space}")
        logging.info(f"topologies {possible_topologies}")

        logging.info("medha")
        action_space = 0
        possible_topologies = 1
        for sub in range(nr_substations):
            nr_elements = len(
                env.observation_space.get_obj_substations(substation_id=sub)
            )
            nr_non_lines = sum(
                1
                for row in env.observation_space.get_obj_substations(substation_id=sub)
                if row[1] != -1 or row[2] != -1
            )
            alpha = 2 ** (nr_elements - 1)
            beta = nr_elements - (1 if nr_elements == 2 else 0)
            gamma = 2**nr_non_lines - 1 - nr_non_lines
            action_space += alpha - beta - gamma
            possible_topologies *= max(alpha - beta - gamma, 1)

        logging.info(f"actions {action_space}")
        logging.info(f"topologies {possible_topologies}")

        logging.info("TenneT")
        action_space = 0
        possible_topologies = 1
        for sub in range(nr_substations):
            nr_elements = len(
                env.observation_space.get_obj_substations(substation_id=sub)
            )
            nr_non_lines = sum(
                1
                for row in env.observation_space.get_obj_substations(substation_id=sub)
                if row[1] != -1 or row[2] != -1
            )
            nr_lines = nr_elements - nr_non_lines
            alpha = 2 ** (nr_elements - 1)
            beta = (
                2 ** (nr_non_lines) - 1
            )  # cases in which there is at least one non-line
            epsilon = (
                1 + nr_lines
            )  # cases in which there are less than two lines connected to a busbar
            theta = nr_lines  # cases in which there is only one line and no non-lines
            action_space += alpha - beta * epsilon - theta
            possible_topologies *= max(alpha - beta * epsilon - theta, 1)

        logging.info(f"actions {action_space}")
        logging.info(f"topologies {possible_topologies}")


# def load_config(file_path="config.yaml"):
#     with open(file_path, "r") as stream:
#         try:
#             config = yaml.safe_load(stream)
#             return config
#         except yaml.YAMLError as e:
#             print(f"Error loading YAML file {file_path}: {e}")
#             return None
