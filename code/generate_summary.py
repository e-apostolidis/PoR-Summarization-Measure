import numpy as np
from knapsack_implementation import knapSack

''' 
  Function that generates a binary summary for each video of a split using the knapsack algorithm and the computed importance scores. 

  Inputs:
    split_shot_bound: shot boundaries for the videos of the split - list of length n_videos - each element contains a numpy array of shape (n_shots, 2)
    split_scores: importance scores for the videos of the split - list of length n_videos - each element contains a numpy array of shape (n_frames, 1)
  Outputs:
    split_summaries: binary summaries for the videos of the split - list of length n_videos - each element contains a numpy array of shape (n_frames, 1)

'''

def generate_summary(split_shot_bound, split_scores): 
    split_summaries = []
    for video_index in range(len(split_scores)):
        shot_bound = split_shot_bound[video_index] # [number_of_shots, 2] - the boundaries refer to the initial number of frames (before the subsampling)
        scores = split_scores[video_index]
	
    	# Compute shot-level importance scores by taking the average importance scores of all frames in the shot
        shot_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1]-shot[0]+1)
            shot_scores.append((scores[shot[0]:shot[1]+1].mean()).item())
	
        # Select the best shots using the knapsack implementation
        final_max_length = int((shot[1]+1)*0.15)

        selected = knapSack(final_max_length, shot_lengths, shot_scores, len(shot_lengths))
		
        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(shot[1]+1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1]+1] = 1
	
        split_summaries.append(summary)
		
    return split_summaries

