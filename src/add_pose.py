import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate):
    # Odometry: rotate 45°, move 2m, rotate 45° more (all in robot local frame)
    # Local displacement: (sqrt(2), sqrt(2), 90°)
    odometry = gtsam.Pose2(math.sqrt(2), math.sqrt(2), math.pi / 2)
    
    # TODO: Add the odometry factor between X(3) and X(4)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # TODO: Initial estimate for X(4) in global frame
    # X(3) is at (4, 0, 0°), applying odometry gives approximately (4+√2, √2, 90°)
    x4_global = gtsam.Pose2(4 + math.sqrt(2), math.sqrt(2), math.pi / 2)
    initial_estimate.insert(X(4), x4_global)

    return graph, initial_estimate