import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_landmark_measurement(graph, initial_estimate, result):
    # Get optimized X(4) and L(2) positions from result
    x4 = result.atPose2(X(4))
    l2 = result.atPoint2(L(2))

    # Compute bearing and distance from X(4) to L(2)
    dx = l2[0] - x4.x()
    dy = l2[1] - x4.y()
    
    # Global angle to landmark
    global_angle = math.atan2(dy, dx)
    
    # Bearing relative to robot heading
    rotation = math.degrees(global_angle - x4.theta())
    
    # Euclidean distance
    distance = math.sqrt(dx**2 + dy**2)

    graph.add(gtsam.BearingRangeFactor2D(
        X(4), L(2),
        gtsam.Rot2.fromDegrees(rotation),
        distance,
        MEASUREMENT_NOISE
    ))
    return graph