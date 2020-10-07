import numpy as np

''' 
  Function that evaluates the predicted summary using F-Score. 

  Inputs:
    predicted_summary: numpy array of shape (n_frames)
    user_summary: numpy array of shape (n_users, n_frames)
    eval_method: method for combining the F-Scores for each comparison with user summaries - values: 'avg' or 'max'
  Outputs:
    max or average F-Score between users, depending on the selected 'eval_method'

'''

def evaluate_summary_fscore(predicted_summary, user_summary, eval_method):
    max_len = max(len(predicted_summary),user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        # Compute precision, recall, f-score
        precision = sum(overlapped)/sum(S)
        recall = sum(overlapped)/sum(G)
        if (precision+recall==0):
            f_scores.append(0)
        else:
            f_scores.append(2*precision*recall*100/(precision+recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)
