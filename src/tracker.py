import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class HungarianTracker:

    def __init__(self, max_distance=25):
        self.max_distance = max_distance

    def match(self, previous, current):

        if len(previous) == 0 or len(current) == 0:
            return []

        distance_matrix = cdist(previous, current)

        rows, cols = linear_sum_assignment(distance_matrix)

        matches = []

        for r, c in zip(rows, cols):

            if distance_matrix[r, c] <= self.max_distance:

                matches.append(
                    {
                        "previous": r,
                        "current": c,
                        "distance": distance_matrix[r, c],
                    }
                )

        return matches

    def track(self, detections):

        all_matches = []

        for frame in range(len(detections) - 1):

            matches = self.match(
                detections[frame],
                detections[frame + 1],
            )

            all_matches.append(matches)

        return all_matches