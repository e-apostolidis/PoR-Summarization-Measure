import h5py
import json
import csv
import numpy as np
import argparse
import os
from generate_summary import generate_summary
from evaluation_fscore import evaluate_summary_fscore

''' 
  Function that computes random performance over a set of videos (split). 

  Inputs:
    split_shot_bound: shot boundaries for the videos of the split - list of length n_videos - each element contains a numpy array of shape (n_shots, 2)
    split_nframes: number of frames for the videos of the split - list of length n_videos
    split_user_summary: multiple user summaries for the videos of the split - list of length n_videos - each element contains a numpy array of shape (n_users, n_frames)
    eval_method: method for combining the F-Scores for each comparison with user summaries - values: 'avg' or 'max'
  Outputs:
    performance of a random summarizer for this split

'''

def compute_RP(split_shot_bound, split_nframes, split_user_summary, eval_method):
    # For 100 seeds
    f_score_seeds = []
    for seed in range(100):
        np.random.seed(seed)
        split_rand_seq = []
        f_score_split = []
        for i in range(len(split_shot_bound)): # runs for all videos
            random_sequence = np.random.rand(split_nframes[i])
            split_rand_seq.append(random_sequence)
            
        split_rand_summaries = generate_summary(split_shot_bound, split_rand_seq)

        # Compare the random summary with the ground truth one, for each video
        for video_index in range(len(split_rand_summaries)):
            rand_summary = split_rand_summaries[video_index]
            user_summary = split_user_summary[video_index]
            f_score = evaluate_summary_fscore(rand_summary, user_summary, eval_method)
            f_score_split.append(f_score)
        
        f_score_split_mean = np.mean(f_score_split)
        f_score_seeds.append(f_score_split_mean)
        
    return np.mean(f_score_seeds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="path to data directory") # e.g. '../data/'
    parser.add_argument('--splits_filename', type=str, required=True, help="name of splits' file") # e.g. 'summe_splits.json'
    parser.add_argument('--split_id', type=int, required=True, help="path to splits' file") # e.g. 2 (starting from 0)
    parser.add_argument('--h5_filename', type=str, required=True, help="name of h5 (data) file") # e.g. 'eccv16_dataset_summe_google_pool5.h5'
    parser.add_argument('--save_dir', type=str, required=True, help="path to results directory") # e.g. '../results/'
    parser.add_argument('--results_filename', type=str, required=True, help="name of the file, where the results will be saved (.csv)") # e.g. 'summe_random.csv'
    parser.add_argument('--eval_method', type=str, required=True, help="evaluation method ('avg' or 'max')")
    args = parser.parse_args()

    splits_path = os.path.join(args.data_dir, args.splits_filename)
    h5_path = os.path.join(args.data_dir, args.h5_filename)
    results_path = os.path.join(args.save_dir,  args.results_filename)

    # Read the names of the videos of the selected split
    video_names = []
    with open(splits_path) as f:
        data = json.loads(f.read())
        split = data[args.split_id]
        for video_name in split['test_keys']:
            video_names.append(video_name)

    # Read the relevant data
    split_user_summary = []
    split_shot_bound =[]
    split_nframes = []
    with h5py.File(h5_path, 'r') as hdf:
        for video_name in video_names:

            user_summary = np.array( hdf.get(video_name+'/user_summary') )
            sb = np.array( hdf.get(video_name+'/change_points') )
            n_frames = np.array( hdf.get(video_name+'/n_frames') )

            split_user_summary.append(user_summary)
            split_shot_bound.append(sb)
            split_nframes.append(n_frames)
    
    RP = compute_RP(split_shot_bound, split_nframes, split_user_summary, args.eval_method)

    # Save in a csv file: (split_{split_id}, RP)
    csv_file = open(results_path, 'a')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['split_'+str(args.split_id), str(RP)])
    csv_file.close()