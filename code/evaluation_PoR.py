import os
import csv
import json
import numpy as np
import h5py
import argparse
from evaluation_fscore import evaluate_summary_fscore
from random_per import compute_RP

''' Computes PoR for a specific data split '''

def evaluate_summary_PoR(data_dir, h5_filename, split_id, input_rp, input_fscore, save_dir, results_filename, rp=None, fscore=None, summaries_path=None, splits_filename=None, eval_method=None):

    # 1) Compute the performance of a random summarizer (F) for the selected data split
    if args.input_rp:
        F = args.rp
    else:
        # Read the names of the videos of the selected split
        if args.input_fscore:
            splits_path = os.path.join(args.data_dir, args.splits_filename)
            video_names = []
            with open(splits_path) as f:
                data = json.loads(f.read())
                split = data[args.split_id]
                for video_name in split['test_keys']:
                    video_names.append(video_name)

        else:
            with open(args.summaries_path) as f:
                data = json.loads(f.read())
                video_names = list(data.keys())

        # Read the relevant data
        split_user_summary = []
        split_shot_bound =[]
        split_nframes = []
        with h5py.File(os.path.join(args.data_dir, args.h5_filename), 'r') as hdf:
            for video_name in video_names:

                user_summary = np.array( hdf.get(video_name+'/user_summary') )
                sb = np.array( hdf.get(video_name+'/change_points') )
                n_frames = np.array( hdf.get(video_name+'/n_frames') )

                split_user_summary.append(user_summary)
                split_shot_bound.append(sb)
                split_nframes.append(n_frames)

        F = compute_RP(split_shot_bound, split_nframes, split_user_summary, args.eval_method)


    # 2) Compute the F-Score (S) for each generated summary of the selected data split
    if args.input_fscore:
        S = args.fscore
    else:
        hdf = h5py.File(os.path.join(args.data_dir, args.h5_filename), 'r')
        split_fscores = []
        with open(args.summaries_path) as f:
            data = json.loads(f.read())
            video_names = list(data.keys())

            for video_name in video_names:
                summary = np.asarray(data[video_name])
                user_summary = np.array( hdf.get(video_name+'/user_summary') )
                fscore = evaluate_summary_fscore(summary, user_summary, args.eval_method)
                split_fscores.append(fscore)

        hdf.close()
        S = np.mean(split_fscores)

    # 3) Use the computed F-Scores to compute the PoR metric as: PoR = S / F * 100
    PoR = S / F * 100
    print("PoR for split", args.split_id, "is", PoR)

    # save in a csv file: (split_{split_id}, RP)
    results_path = os.path.join(args.save_dir, args.results_filename)
    csv_file = open(results_path, 'a')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['split_'+str(args.split_id), str(PoR)])
    csv_file.close()

    return PoR

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="path to data directory") # e.g. '../data/'
    parser.add_argument('--h5_filename', type=str, required=True, help="name of h5 (data) file") # e.g. 'eccv16_dataset_summe_google_pool5.h5'
    parser.add_argument('--split_id', type=int, required=True, help="split index") # e.g. 2 (starting from 0)

    parser.add_argument('--input_rp', type=str2bool, required=True, help="whether the random performance has already been extracted ('True' or 'False')")
    parser.add_argument('--rp', type=float, required=False, help="F-Score value for this split (if input_rp=True)") # e.g. 42.4

    parser.add_argument('--input_fscore', type=str2bool, required=True, help="whether the F-Scores have already been extracted ('True' or 'False')")
    parser.add_argument('--fscore', type=float, required=False, help="F-Score value for this split (if input_fscore=True)") # e.g. 45.6
    parser.add_argument('--summaries_path', type=str, required=False, help="path to the summaries' file (if input_fscore=False)") # e.g. path_to/automatic_summaries.json

    parser.add_argument('--splits_filename', type=str, required=False, help="name of splits' file (if input_rp=False & input_fscore=True)") # e.g. 'summe_splits.json'
    parser.add_argument('--eval_method', type=str, required=False, help="evaluation method ('avg' or 'max') - if input_rp=False or input_fscore=False")

    parser.add_argument('--save_dir', type=str, required=True, help="path to results directory") # e.g. '../results/'
    parser.add_argument('--results_filename', type=str, required=True, help="name of the file, where the results will be saved (.csv)") # e.g. 'summe_PoR.csv'
    args = parser.parse_args()

    evaluate_summary_PoR(args.data_dir, args.h5_filename, args.split_id, args.input_rp, args.input_fscore, args.save_dir, args.results_filename, args.rp, args.fscore, args.summaries_path, args.splits_filename, args.eval_method)
