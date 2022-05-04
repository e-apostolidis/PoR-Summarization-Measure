# Performance over Random: A Robust Evaluation Protocol for Video Summarization Methods

## Python Implementation of the PoR Evaluation Measure
- From **"Performance over Random: A Robust Evaluation Protocol for Video Summarization Methods"**, Proc. of the 28th ACM Int. Conf. on Multimedia (MM '20), October 12-16, 2020, Seattle, WA, USA)
- Written by Evlampios Apostolidis, Eleni Adamantidou, Alexandros I. Metsai, Vasileios Mezaris and Ioannis Patras
- This software can be used for evaluating the summaries of a video summarization method using the PoR evaluation protocol.

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

The 50 randomly-created data splits that were used in our experiments can be found in the "data/splits" directory. In each split 80% of the data is used for training and the remaining 20% for testing.

## Evaluation
To evaluate the performance of a summarization algorithm using the PoR evaluation measure, the output of the algorithm should follow the format of the binary (json) file available [here](https://github.com/e-apostolidis/PoR-Summarization-Measure/blob/master/data/example_binary_summary.json). In this file, the sequence of binary (0 or 1) values for each test video indicates whether a video frame was selected (value = 1) or not (value = 0) to be inlcuded in the summary. Given this file, the evaluation is performed by running:
<pre>
python evaluation_PoR.py
  --data_dir: Path to data directory (here, '../data/')
  --h5_filename: Name of the h5 file with the video features and annotations of the used dataset
  --split_id: Index of the selected data split
  --input_rp: Boolean value ('True' or 'False') that indicates whether the random performance has been already computed for the test videos of the used data split
  --rp: F-Score value (in percentages) of the computed random performance (Optional use, if input_rp=True)
  --input_fscore: Boolean value ('True' or 'False') that indicates whether the performance of a video summarization method has been already computed for the test videos of the used data split
  --fscore: F-Score value (in percentages) of the computed summarization performance (Optional use, if input_rp=True)
  --summaries_path: Path to the binary (json) file with the data about the automatically-generated summary (Optional use, if input_fscore=False)
  --splits_filename: Name of the json file with the data splits, that can be found in '../data/splits/'
  --eval_method: String that indicates how the computed F-Score values for the different user summaries of a test video will be used; it can be either 'avg' (for the TVSum dataset) or 'max' (for the SumMe dataset)
  --save_dir: Path to the directory where the results of the evaluation will be stored (here, '../results/')
  --results_filename: Name of the file where the results will be stored (.csv)
</pre>

Please note that some arguments are optional for running the code. This functionality gives freedom to the user, which is able to either use the entire evaluation pipeline or to provide precomputed scores about the performance of the random summarizer and/or a summarization method.

Alternatively, if the user wants to evaluate multiple different data splits using the PoR evaluation measure, he/she can integrate in his/her own code the function "evaluate_summary_PoR(...)" that is included in the "evaluation_PoR.py" file.

## Calculation of random performance
To calculate only the performance of a random summarizer for the test videos of a data split, run:
<pre>
python random_per.py
  --data_dir: Path to data directory (here, '../data/')
  --h5_filename: Name of the h5 file with the video features and annotations of the used dataset
  --split_id: Index of the selected data split
  --splits_filename: Name of the json file with the data splits, that can be found in '../data/splits/'
  --eval_method: String that indicates how the computed F-Score values for the different user summaries of a test video will be used; it can be either 'avg' (for the TVSum dataset) or 'max' (for the SumMe dataset)
  --save_dir: Path to the directory where the results of the evaluation will be stored (here, '../results/')
  --results_filename: Name of the file where the results will be stored (.csv)
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
