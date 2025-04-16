import numpy as np

# do not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
# __________end of block__________


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    # YOUR CODE HERE
    distances = np.zeros((des1.shape[0], des2.shape[0]))
    for i in range(des1.shape[0]):
      for j in range(des2.shape[0]):
        distances[i, j] = np.linalg.norm(des1[i] - des2[j])

    matches = []
    for i in range(des1.shape[0]):
      temp_ind = np.argmin(distances[i])
      if np.argmin(distances[:, temp_ind]) == i:
        matches.append(DummyMatch(i, temp_ind, distances[i, temp_ind]))
    return sorted(matches, key=lambda x: x.distance)
