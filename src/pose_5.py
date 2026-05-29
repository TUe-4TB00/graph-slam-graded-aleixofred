import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X
import copy

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    print(result)
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None
    best_landmark = None
    best_sum = float('inf')
    best_return_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1]:
            g = copy.deepcopy(graph)
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)

            marginals = gtsam.Marginals(g, result)

            # Use trace to select the winner
            selection_metric = np.trace(marginals.marginalCovariance(L(1)))

            # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
            return_sum = (
                marginals.marginalCovariance(L(1)).sum() +
                marginals.marginalCovariance(L(2)).sum()
            )

            print(f"Pose {pose_key}, Landmark {landmark}: sum of marginals = {return_sum}")

            if selection_metric < best_sum:
                best_sum = selection_metric
                best_return_sum = return_sum
                best_pose = pose_key
                best_landmark = landmark

    # Rebuild the best result to return
    g = copy.deepcopy(graph)
    est = gtsam.Values(initial_estimate)
    pose_5 = pose_options[best_pose]
    g, est = add_pose(g, est, pose_5)
    result = optimize(g, est)
    g = add_landmark_measurement(g, result, pose_5, best_landmark)
    result = optimize(g, est)

    return best_pose, best_landmark, best_return_sum

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None
    best_landmark = None
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = copy.deepcopy(graph)
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)

            # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
            list_of_errors = []
            for i, (x_gt, y_gt, theta_gt) in enumerate([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]):
                pose = result.atPose2(X(i + 1))
                error = abs(pose.x() - x_gt) + abs(pose.y() - y_gt) + abs(pose.theta() - theta_gt)
                list_of_errors.append(error)

            # TODO: compute the sum of the errors and return it along with the best pose and landmark
            sum_of_errors = sum(list_of_errors)

            print(f"Pose {pose_key}, Landmark {landmark}: sum of errors = {sum_of_errors}")

            if sum_of_errors < best_sum:
                best_sum = sum_of_errors
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum