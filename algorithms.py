import itertools
from geopy import distance as geopy_dist

def held_karp_tsp(distance_mat): 
    # O(n^2 * 2^n)
    # Worst case scenario: more than ~15 stores, time complexity will be 'super' exponentially large
    # Best case scenario: under 15
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
    # O(n^2)
    # worst-case scenario: no
    # best-case scenario: no
    
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
    # O(n)
    # worst-case scenario: no
    # best-case scenario: no

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


def KMPSearch(pat, txt):
    # O(n)
    # worst-case scenario: no repetitiveness in text or pattern , e.g., abcdef, instead of e.g., ababcabc
    # best-case scenario: e.g., ababcabc
    def computeLPSArray(pat, M, lps):
        len = 0  # length of the previous longest prefix suffix

        lps[0]  # lps[0] is always 0
        i = 1

        # the loop calculates lps[i] for i = 1 to M-1
        while i < M:
            if pat[i] == pat[len]:
                len += 1
                lps[i] = len
                i += 1
            else:
                # This is tricky. Consider the example.
                # AAACAAAA and i = 7. The idea is similar
                # to search step.
                if len != 0:
                    len = lps[len-1]

                    # Also, note that we do not increment i here
                else:
                    lps[i] = 0
                    i += 1

    visited= False
    count = 0
    # global count
    M = len(pat)
    N = len(txt)

    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0]*M
    j = 0  # index for pat[]

    # Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, M, lps)

    if M == N:
        i = 0  # index for txt[]
        while i < N:
            if pat[j] == txt[i]:
                i += 1
                j += 1

            if j == M:
                #print ("Found pattern at index", str(i-j))
                j = lps[j-1]
                count = count+1
                visited=True

            # mismatch after j matches
            elif i < N and pat[j] != txt[i]:
                # Do not match lps[0..lps[j-1]] characters,
                # they will match anyway
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1
    
    return count, visited

class TrieNode:
        def __init__(self):
            self.children = {}
            self.endOfString = False
            
class Trie:    
    def __init__(self):
        self.root = TrieNode()

    def insertString(self, word):
        current = self.root
        for i in word:
            ch = i
            node = current.children.get(ch)
            if node == None:
                node = TrieNode()
                current.children.update({ch:node})
            current = node
        current.endOfString = True

    def searchString(self, word):
        count=0
        is_word_unfinished = False
        currentNode = self.root
        for i in word:
            node = currentNode.children.get(i)
            
            if node == None:
                is_word_unfinished = True
                break
                
            currentNode = node
        
        if is_word_unfinished:
            count = 0
        elif currentNode.endOfString == True:
            count += 1

        return count  