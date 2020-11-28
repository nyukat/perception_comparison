# Differences between human and machine perception in medical diagnosis
This repository accompanies our paper [Differences between human and machine perception in medical diagnosis](https://arxiv.org/abs/????.?????). In the paper, 
we propose a framework for comparing human and machine perception in medical diagnosis, and demonstrate it with a case 
study in breast cancer screening. This repository contains the data and code necessary to reproduce the results from our 
case study. There are three components:
1. `probabilistic_inference.py`: We collected predictions from radiologists and DNNs on screening mammograms perturbed 
with Gaussian low-pass filtering (Figure 1a--b). The predictions are provided in `data/observed_predictions`, see 
below for details. We apply probabilistic modeling to these predictions in order to isolate the effect that 
low-pass filtering has on their predictions (Figure 1d). 
2. `perturbation_study_analysis.py`: We sample from the probabilistic model and compare radiologists and DNNs 
with respect to predictive confidence and class separability (Figure 1e--f).
3. `annotation_study_analysis.py`: Radiologists annotated regions of interest (ROIs) indicating the most suspicious 
regions of images (Figure 1g). We then collected predictions from DNNs where we separately applied low-pass filtering to 
the ROI interior (Figure 1h), ROI exterior (Figure 1i), and to the entire image (Figure 1j). These predictions are provided 
in `data/observed_predictions`, see below for details. We then examine how these low-pass filtering schemes affect the 
DNNs' class separability.  

![](data/framework.svg)

Additionally, we include our implementation of Gaussian low-pass filtering in `fourier_filter.py`.

### Setup

Add this directory to `PYTHONPATH`, and install the following:
* Python 3.*
* Gin-config
* Imageio
* Matplotlib
* NumPy
* Pandas
* PyStan
* SciPy

PyStan requires a C++14 compatible compiler. If gcc-related errors are encountered, see the detailed installation 
instructions in <https://pystan.readthedocs.io/en/latest/installation_beginner.html>. These errors can likely be 
resolved with:
* Linux: `conda install gcc_linux-64 gxx_linux-64 -c anaconda`
* Mac: `conda install clang_osx-64 clangxx_osx-64 -c anaconda`

### Data
The radiologist and DNN predictions from our two reader studies are stored as NumPy arrays, and are located in 
`data/observed_predictions`. For DNNs, we include predictions for two architectures: [GMIC](https://arxiv.org/abs/2002.07613) 
and [DMV](https://ieeexplore.ieee.org/document/8861376).

For our perturbation study, we have predictions for (i) radiologists (`radiologists.pkl`), (ii) DNNs trained on unperturbed 
data (`unperturbed.pkl`), and (iii) DNNs trained on low-pass filtered data (`filtered.pkl`). The radiologists' predictions 
have the shape `(10, 9, 720, 2)`, where 10 is the number of radiologists, 9 is the number of filtering severities, 720 is 
the number of screening mammography exams, and 2 is the number of breasts. The DNNs' predictions have a similar shape of 
`(5, 9, 720, 2)`, where 5 is the number of DNN training seeds, and all other dimensions are the same as the radiologists. 
Each set of predictions comes with a corresponding mask, which is a binary NumPy array with the same shape as the predictions. 
Each element in the mask is set to 1 if we have a prediction for that corresponding index. This is necessary since the 
radiologists' predictions are sparse, while the DNNs' predictions are dense. See our paper for details regarding this issue.

For our annotation study, we have DNN predictions where low-pass filtering is applied to the (i) ROI interior 
(`roi_interior.pkl`), (ii) ROI exterior (`roi_exterior.pkl`), and to the (iii) entire image (these are a subset of 
predictions from the perturbation study). These predictions have the shape `(7, 5, 9, 120, 2)`, where 7 is the number of 
radiologists who annotated the images, 5 is the number of DNN training seeds, 9 is the number of severities, 120 is the 
number of screening mammography exams, and 2 is the number of breasts.

### Usage
**Reproducing our results**

Here are the steps for reproducing our results on comparing radiologists and DNNs for the GMIC architecture. See 
`demo.ipynb` for these steps applied sequentially. To perform the analyses with the DMV architecture, replace all 
instances of `gmic` with `dmv`. The results of the analyses are saved in the `results` directory by default, but this 
behavior can be modified in the configuration files in `cfg`.

The first step is to perform probabilistic inference and generate posterior samples:
1. Radiologists: `python code/probabilistic_inference.py cfg/probabilistic_inference/radiologists.gin`
2. DNNs: `python code/probabilistic_inference.py cfg/probabilistic_inference/dnns/gmic/unperturbed.gin`
3. DNNs (trained w/ filtered data): `python code/probabilistic_inference.py cfg/probabilistic_inference/dnns/gmic/filtered.gin`

Next, we compare how low-pass filtering affects the predictive confidence and class separability of radiologists 
and DNNs, performing separate analyses for two subgroups:
1. Microcalcifications: `python code/perturbation_study_analysis.py cfg/perturbation_study_analysis/gmic/microcalcifications.gin`
2. Soft tissue lesions: `python code/perturbation_study_analysis.py cfg/perturbation_study_analysis/gmic/soft_tissue_lesions.gin` 

Finally, we compare radiologists and DNNs with respect to the regions of an image deemed most suspicious. We 
perform separate analyses for two subgroups, but this time in a single call.

`python annotation_study_analysis.py cfg/annotation_study_analysis/gmic.gin`

**Gaussian low-pass filtering**

See `fourier_filter.ipynb` to see our implementation of Gaussian low-pass filtering applied to an example screening 
mammogram.

### License
This repository is licensed under the terms of the GNU AGPLv3 license.

## Reference

If you found this code useful, please cite our paper:

**Differences between human and machine perception in medical diagnosis**\
Taro Makino, Stanisław Jastrzębski, Witold Oleszkiewicz, Celin Chacko, Robin Ehrenpreis, Naziya Samreen, Chloe Chhor, Eric Kim, Jiyon Lee, Kristine Pysarenko, Beatriu Reig, Hildegard Toth, Divya Awal, Linda Du, Alice Kim, James Park, Daniel K. Sodickson, Laura Heacock, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras\
2020

    @article{Makino2020Differences, 
        title = {Differences between human and machine perception in medical diagnosis},
        author = {Taro Makino and Stanisław Jastrzębski and Witold Oleszkiewicz and Celin Chacko and Robin Ehrenpreis and Naziya Samreen and Chloe Chhor and Eric Kim and Jiyon Lee and Kristine Pysarenko and Beatriu Reig and Hildegard Toth and Divya Awal and Linda Du and Alice Kim and James Park and Daniel K. Sodickson and Laura Heacock and Linda Moy and Kyunghyun Cho and Krzysztof J. Geras}, 
        journal = {arXiv:????.?????},
        year = {2020}
    }
