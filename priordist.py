import numpy as np

class PriorDistributionCell:
    # Generates prior distribution for a given set of responses against a given set of stimuli,
    # with response mapping done at individual cell level
    # Spike counts are assumed to be bined into 4 categories of firing activity (bottom 25%, 25-50%, 50-75%, top 25%)
    num_cats = 4

    def map_response_distribution(responses: np.array, dist=None) -> np.array:
        # Takes in (n x l) 2-D array of (l cells) population responses across (n number of) observations
        # Returns a 2-D array mapping of each cell's binned responses to its number of occurences for all cells
        # Format of mapping is dist[cell, cat] = occurence count, each row corresponds to the mapping of one cell
        if dist is None:
            dist = np.zeros((responses.shape[1], PriorDistributionCell.num_cats), dtype=np.int32)
        for (obs, cell), cat in np.ndenumerate(responses):
            dist[int(cell), int(cat)] += 1
        return dist

    def get_occurences_per_stimulus(responses_per_stimulus: np.array) -> np.array:
        # Returns the number of occurences of each stimulus in the dataset
        return np.array(list(map(lambda arr: arr.shape[0], responses_per_stimulus)))

    def get_distribution_total(dist: np.array) -> np.array:
        # Returns the total number of observations/occurences per cell in the given distribution
        return np.sum(dist, axis=1)[0]

    def __init__(self, responses: np.array, responses_per_stimulus: np.array):
        self.r_dist = PriorDistributionCell.map_response_distribution(responses)
        self.s_dist = PriorDistributionCell.get_occurences_per_stimulus(responses_per_stimulus)
        self.r_s_dist = list()
        for stimulus in responses_per_stimulus:
            self.r_s_dist.append(PriorDistributionCell.map_response_distribution(stimulus))

        self.r_total = PriorDistributionCell.get_distribution_total(self.r_dist)
        self.s_total = np.sum(self.s_dist)
        self.r_s_total = list()
        for stimulus in self.r_s_dist:
            self.r_s_total.append(PriorDistributionCell.get_distribution_total(stimulus))

    def P_r(self, cell: int, resp: int) -> float:
        # Returns probability of observing a given response for a given cell
        return self.r_dist[cell, resp] / self.r_total

    def P_s(self, stim: int) -> float:
        # Returns probability of ocuurence of a given stimulus
        return self.s_dist[stim] / self.s_total

    def P_r_s(self, cell: int, resp: int, stim: int) -> float:
        # Returns probability of observing a given response for a given cell, given a specific stimulus
        return self.r_s_dist[stim][cell, resp] / self.r_s_total[stim]
