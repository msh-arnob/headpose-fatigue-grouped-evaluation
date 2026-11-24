# Head-pose angles and fatigue detection under grouped evaluation

Code, results and figures for the paper

> Md Sharmon Hossain Arnob, Susmita Biswas Bristy, Judith N. Njoku and Diksha Shukla.
> **Do Absolute Head-Pose Angles Help Fatigue Detection on Unseen Recordings and Drivers?**
> WACV 2027 Workshops (HOUR), 2027.

The study trains the same classifiers on eye aspect ratio (EAR), mouth aspect ratio (MAR) and absolute head-pose angles from UTA-RLDD, and changes only how the data are split: random frames, held-out recordings or held-out drivers. Head pose raises Random Forest balanced accuracy from 54.46% to 88.12% under random-frame splitting, but gives no improvement once complete recordings or drivers are held out. The notebook reproduces every table, figure and number in the paper and its supplementary material.

## Repository contents

| Path | Contents |
|---|---|
| `headpose_fatigue_grouped_evaluation.ipynb` | Main notebook, run with saved outputs. Sections follow the paper. |
| `superseded_analyses.ipynb` | Earlier analyses and figure versions, kept for the record. Not used in the paper. |
| `results/` | Result tables (CSV) used in the paper and supplement |
| `results/histories/` | Bi-LSTM training and validation loss for every epoch |
| `results/predictions/` | Test-set predictions on the 12 held-out drivers |
| `figures/` | The 11 figures produced by the notebook |
| `requirements.txt` | Package versions used to produce the results |

The datasets, the extracted feature files (`data/`) and the trained models (`models/`) are not included; see [Data](#data).

## Where each result comes from

Section numbers refer to the headings in `headpose_fatigue_grouped_evaluation.ipynb`.

**Main paper**

| Paper item | Notebook section | File |
|---|---|---|
| Table 1, split composition | 4.1 | `results/split_composition.csv` |
| Table 2, Random Forest under three splits | 5.1 | `results/rf_three_splits.csv` |
| Table 3, Bi-LSTM checkpoint selection | 6.1, 6.3 | `results/bilstm_held_out_drivers.csv`, `results/bilstm_checkpoint_selection.csv` |
| Table 4, static classifiers on held-out drivers and NTHU-DDD | 5.3 | `results/static_held_out_drivers.csv`, `results/static_nthu_ddd.csv` |
| All-drowsy reference on NTHU-DDD | 5.4 | `results/nthu_ddd_full_set_with_reference.csv` |
| Bootstrap intervals | 7.8 | `results/bootstrap_intervals.csv`, `results/bootstrap_differences.csv` |
| Figure 1, study overview | Drawn separately; section 3.1 makes the landmark panel | |
| Figure 2, drop from random to subject-held-out evaluation | 7.1 | `figures/drop_random_to_subject_held_out.png` |
| Figure 3, impurity and permutation importance | 7.2 | `figures/feature_importance_impurity_vs_permutation.png`, `results/permutation_importance.csv` |
| Figure 4, Bi-LSTM loss curves | 7.3 | `figures/bilstm_loss_curves.png` |
| Figure 5, held-out balanced accuracy by model | 7.5 | `figures/held_out_balanced_accuracy_by_model.png` |
| Figure 6, held-out confusion matrices | 7.6 | `figures/confusion_held_out_full_set.png` |

**Supplementary material**

| Supplement item | Notebook section | File |
|---|---|---|
| Tables S1 and S2, all static classifiers | 5.3 | `results/static_held_out_drivers.csv`, `results/static_nthu_ddd.csv` |
| Table S3, Bi-LSTM on held-out drivers | 6.1 | `results/bilstm_held_out_drivers.csv` |
| Table S4, repeated subject-held-out partitions | 8.2 | `results/repeated_subject_partitions.csv` |
| Table S5, repeated recording-held-out partitions | 8.3 | `results/repeated_recording_partitions.csv` |
| Table S6, random-frame split with angle encodings | 8.4 | `results/random_frame_encodings.csv` |
| Table S7, driver IDs in each partition | 4.2, 8.2 | `results/driver_lists.csv` |
| Tables S8 and S9, variance shares and identity probe | 9.1 | `results/between_group_variance.csv`, `results/identity_probe.csv` |
| Table S10, recording-held-out accuracy by training labels | 9.2 | `results/recording_held_out_by_training_labels.csv` |
| Figure S1, balanced accuracy per held-out driver | 7.9 | `figures/per_driver_balanced_accuracy.png`, `results/per_driver_balanced_accuracy.csv` |
| Figure S2, confusion matrices for Logistic Regression and Decision Tree | 7.6 | `figures/confusion_held_out_full_set_lr_dt.png` |
| Figure S3, confusion matrices on NTHU-DDD | 7.7 | `figures/confusion_nthu_ddd_full_set.png` |
| Figure S4, Bi-LSTM loss curves for the other feature sets | 7.4 | `figures/bilstm_loss_curves_other_feature_sets.png` |
| Figure S5, head-pose angles by class and recording | 9.3 | `figures/head_pose_by_class_and_recording.png` |

Section 7.9 also draws `figures/per_driver_spread.png`, another view of the per-driver results that is not in the paper.

## Data

Neither dataset can be redistributed here. Obtain them from their owners and place them as shown below.

**UTA Real-Life Drowsiness Dataset (UTA-RLDD).** Available from the dataset page: https://sites.google.com/view/utarldd/home. Please cite Ghoddoosian, Galib and Athitsos (CVPR Workshops 2019) when using it. Only some participants agreed to have their faces published, so do not publish frames without checking the dataset's consent list.

Our extracted data contain 609,068 frames (every fifth frame) from 178 video files of 60 participants. Five of the 180 intended participant-state videos are not present, and three participant states (drivers 32 and 49 drowsy, driver 42 drowsy) are stored as two files each; each file is treated as one recording. A video whose file name contains `_5` (low vigilance) or `_10` (drowsy) is labelled drowsy; `_0` is alert.

**NTHU Driver Drowsiness Detection dataset (NTHU-DDD).** We used a frame version shared on Kaggle as `nthuddd2` (66,521 images: 36,030 drowsy and 30,491 not drowsy), which is no longer online. Its contents are documented in this public notebook: https://www.kaggle.com/code/huthang/normalize-dataset-drowsiness. The original dataset was introduced by Weng, Lai and Lai, "Driver drowsiness detection via a hierarchical temporal deep belief network", ACCV 2016 Workshops; please cite it and obtain the data from its authors. Face extraction succeeded on 65,859 of the images (36,021 drowsy, 29,838 alert).

**Folder layout**

```
data/
  UTA-RLDD/                  UTA-RLDD videos (any subfolders)
  nthuddd2/drowsy/           NTHU-DDD drowsy images
  nthuddd2/notdrowsy/        NTHU-DDD not-drowsy images
  uta_rldd_features.csv      written by section 3.3
  nthu_ddd_features.csv      written by section 3.2
models/                      written when training is switched on
```

## How to run

1. Install Python 3.9 and the packages in `requirements.txt` (`pip install -r requirements.txt`).
2. Place the datasets in `data/` as shown above.
3. Open the notebook and set the two switches in the Configuration cell:
   * `RUN_EXTRACTION = True` extracts the facial features from the videos and images and writes the two feature files. This takes many hours and is needed once.
   * `RUN_TRAINING = True` retrains every classifier and Bi-LSTM and overwrites the files in `results/`. This takes several hours.
   * With both set to `False`, the notebook reads the feature files and the saved results and rebuilds every table and figure in about 10 to 15 minutes.
4. Run all cells in order.

The notebook checks itself as it runs: the Random Forest balanced accuracy on held-out drivers must be 50.70% (Full Set) and 53.35% (EAR + MAR), the Decision Tree Full Set must be 51.75%, the random-frame results of sections 5.1 and 5.2 must agree, and a refitted Random Forest must reproduce the saved predictions.

## Reproducibility notes

* All splits and models use seed 42. The repeated recording-held-out partitions use seeds 1 to 5.
* The results were produced with the versions in `requirements.txt` (Python 3.9.7 on Windows). Random Forest results can change slightly with other scikit-learn versions; for example, scikit-learn 1.9 gives 53.42% instead of 53.35% for EAR + MAR on held-out drivers, which stops the check above.
* Bi-LSTM training on a GPU is not bit-for-bit reproducible. Section 6.3 reruns two configurations and reports the difference from section 6.1; the paper discusses this.

## Citation

```bibtex
@inproceedings{arnob2027headpose,
  title     = {Do Absolute Head-Pose Angles Help Fatigue Detection on Unseen Recordings and Drivers?},
  author    = {Arnob, Md Sharmon Hossain and Bristy, Susmita Biswas and Njoku, Judith N. and Shukla, Diksha},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year      = {2027}
}
```

## License

The code and results in this repository are released under the MIT License (see `LICENSE`). The datasets remain under the terms set by their owners.

## Contact

Md Sharmon Hossain Arnob, University of Wyoming: marnob@uwyo.edu
