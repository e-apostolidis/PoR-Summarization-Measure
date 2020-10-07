# Performance over Random: A Robust Evaluation Protocol for Video Summarization Methods

## PyTorch Implementation of PoR-Summarization-Measure
- From **"Performance over Random: A Robust Evaluation Protocol for Video Summarization Methods"** (28th ACM International Conference on Multimedia (MM '20), October 12-16, 2020, Seattle, WA, USA)
- Written by Evlampios Apostolidis, Eleni Adamantidou, Alexandros I. Metsai, Vasileios Mezaris and Ioannis Patras
- This software can be used for evaluating automatically generated video summaries using the PoR evaluation protocol.

## Main dependencies
- Python  3.6

## Data
Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the "data" folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:
<pre>
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of subsampled frames in original video
    /n_steps                  number of subsampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
</pre>
Original videos and annotations for each dataset are also available in the authors' project webpages:
- TVSum dataset: https://github.com/yalesong/tvsum
- SumMe dataset: https://gyglim.github.io/me/vsum/index.html#benchmark

The 50 randomly created splits that were used for the experiments presented in the paper can be found in the "data/splits" directory. In each split 80% of the data is used for training and 20% for testing.

## Evaluation
To evaluate the automatically generated summaries of a data split, run:
<pre>
python evaluation_PoR.py
  --data_dir: Path to data directory (here, '../data/')
  --h5_filename: Name of the aforementioned h5 file.
  --split_id: Index of the selected data split.
  --input_rp: Whether the random performance has already been extracted ('True' or 'False').
  --rp: F-Score value for the random performance of the split (if input_rp=True)
  --input_fscore: Whether the F-Score values for the generated summaries have already been extracted.
  --fscore: F-Score value for this split (if input_fscore=True)
  --summaries_path: Path to the generated binary summaries' file (if input_fscore=False)
  --splits_filename: Name of the splits' file (if input_rp=False & input_fscore=True)
  --eval_method: Evaluation Method ('avg' or 'max') - if input_rp=False or input_fscore=False
  --save_dir: Path to results directory (here, '../results/')
  --results_filename: name of the file where the results will be saved (.csv)
</pre>

Not all of the above arguments are required to run the code. It is specified in the paranthesis above when an argument is required or not. This functionality gives freedom to the user to use the whole pipeline or to provide either the random performance or the fscore of the generated summaries, that have already been computed.

Alternatively, you can integrate in your own code the function "evaluate_summary_PoR(...)" included in this file if you want to evaluate multiple data splits.

## Calculation of random performance
To calculate only the performance of a random summarizer for a data split, run:
<pre>
python random_per.py
  --data_dir: Path to data directory (here, '../data/')
  --h5_filename: Name of the aforementioned h5 file.
  --split_id: Index of the selected data split.
  --splits_filename: Name of the splits' file (if input_rp=False & input_fscore=True)
  --eval_method: Evaluation Method ('avg' or 'max') - if input_rp=False or input_fscore=False
  --save_dir: Path to results directory (here, '../results/')
  --results_filename: name of the file where the results will be saved (.csv)
</pre>

## Citation
If you find this code useful in your work, please cite the following publication:

E. Apostolidis, E. Adamantidou, A. I. Metsai, V. Mezaris, I. Patras. **"Performance over Random: A Robust Evaluation Protocol for Video Summarization Methods"**. Proc. 28th Int. Conference on Multimedia (MMM '20), October 12-16, 2020, Seattle, WA, USA

DOI: https://doi.org/10.1145/3394171.3413632

## License
Copyright (c) 2020, Evlampios Apostolidis, Eleni Adamantidou, Alexandros I. Metsai, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Acknowledgement
This work was supported by the European Union Horizon 2020 research and innovation programme under grant agreement H2020-780656 ReTV. The work of Ioannis Patras has been supported by EPSRC under grant No. EP/R026424/1.
