import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X
import copy

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate, pose_5):
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
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print(result)
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
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

            # Use .sum() as the value to return
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

            list_of_errors = [
                np.array([result.atPose2(X(1)).x(), result.atPose2(X(1)).y(), result.atPose2(X(1)).theta()]) - np.array([0, 0, 0]),
                np.array([result.atPose2(X(2)).x(), result.atPose2(X(2)).y(), result.atPose2(X(2)).theta()]) - np.array([2, 0, 0]),
                np.array([result.atPose2(X(3)).x(), result.atPose2(X(3)).y(), result.atPose2(X(3)).theta()]) - np.array([4, 0, 0]),
            ]

            sum_of_errors = sum(np.dot(e, e) for e in list_of_errors)

            print(f"Pose {pose_key}, Landmark {landmark}: sum of errors = {sum_of_errors}")

            if sum_of_errors < best_sum:
                best_sum = sum_of_errors
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum