import itertools
import sys
from geopy import distance as geopy_dist

def held_karp_tsp(distance_mat):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        distance_mat: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = len(distance_mat)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (distance_mat[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + distance_mat[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + distance_mat[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))

def nearest_neighbour_tsp(distance_mat, center_idx):
    
    def nextStore(current_idx, visited, distance_mat):
        min_idx = -1
        min_distance = -1
        for i in range(len(distance_mat)):
            if i not in visited:
                distance = distance_mat[current_idx][i]
                if distance < min_distance or min_idx == -1:
                    min_idx = i
                    min_distance = distance

        return min_idx, min_distance


    path = [center_idx]
    visited = [center_idx]
    cost = 0

    current_idx = center_idx
    while current_idx != -1:
        current_idx, min_distance = nextStore(current_idx, visited, distance_mat)

        path.append(current_idx)
        visited.append(current_idx)

        if min_distance != -1:
            cost += min_distance

    cost += distance_mat[path[-1]][center_idx]
    path = path[:-1]
    
    return cost, path

def nearest_center(coordinates):
    n = len(coordinates)

    center = [0, 0]
    for coordinate in coordinates:
        center[0] += coordinate[0]
        center[1] += coordinate[1]
    center[0] /= n
    center[1] /= n
    
    minDiff = -1
    minIndex = -1
    for i in range(n):
        diff = geopy_dist.geodesic(center, coordinates[i])
        if diff < minDiff or minIndex < 0:
            minDiff = diff
            minIndex = i
    
    return minIndex