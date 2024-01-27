# Modeling the Trade-off of Privacy Preservation and Activity Recognition on Low-Resolution Images

> Yuntao Wang\*, Zirui Cheng\*, Xin Yi†, Yan Kong, Xueyang Wang, Xuhai Xu, Yukang Yan, Chun Yu, Shwetak Patel, Yuanchun Shi

This is an official code repository for our paper **Modeling the Trade-off of Privacy Preservation and Activity Recognition on Low-Roslution Images**.

In this paper, we proposed a framework for modeling the trade-off of privacy preservation and activity recognition on low-resolution images. Inspired by previous work, we define the following model to understand the trade-off:
$$
S(r) = L_T\left(f_T(f_r(\mathcal{X})), g_T(\mathcal{X})\right) - 
\lambda \sum_{i=1}^{n} \omega_i L_{P_i}\left(f_{P_i}(f_r(\mathcal{X})), g_{P_i}(\mathcal{X})\right)
$$
In the proposed model, $f_r(\mathcal{X})$ denotes the captured dataset at the resolution $r$. $f_T(\cdot)$ and $f_{P_i}(\cdot)$ denote recognition functions. $L_T(\cdot)$ and $L_{P_i}(\cdot)$ denote evaluation functions. $\omega_i$ denotes the importance weight of each privacy feature. $\lambda$ is a factor measuring the sensitivity ratio between privacy preservation and activity recognition.

We designed a pipeline to implement the model. In our paper, we used the PA-HMDB dataset as a demonstration. First, we collected the importance weights through a user study. Based on the results, we annotated the dataset with privacy feature labels. Second, we investigated the effect of image resolution ($r$) on the recognition performance of both humans and machines by conducting user studies and experimenting with vision models. In the end, we calculated the results for modeling the trade-off of privacy preservation and activity recognition.

<img src="./documentation/framework.pdf">

In this repository, we presented the source code for part of the implementation pipeline. For details of the whole process (e.g., user studies) please see our [paper](https://dl.acm.org/doi/abs/10.1145/3544548.3581425).

## Usage

### Dataset

We provided the image dataset from PA-HMDB in `./dataset`. We included the sampled frames and annotations from the original PA-HMDB dataset. For details on the processing procedure please see our paper.

### Inference

We provided the code for inferring from vision models with our dataset. For referenced repositories please see our paper.

+ For activity recognition, please see `./activity_recognition`. We provided the code for finetuning ResNet, EfficientNet, and VisionTransformers on our dataset.

+ For facial identification, please see `./facial_identification`. We provided the code for testing the ArcFace model on our dataset.

+ For nudity recognition, please see `./nudity_recognition`. We provided the code for testing the NudeNet model on our dataset.

+ For property and object detection, please see `./object_detection`. We provided the code for testing the DETR model on our dataset.

+ For relationship classification, please see `./relationship_classification`. We provided the code for testing the Graph Reasoning Model on our dataset.

## License

Our repository is released under the MIT license. Please see the LICENSE file for more information.

## Cite

```latex
@inproceedings{wang_modeling_2023,
	address = {New York, NY, USA},
	series = {{CHI} '23},
	title = {Modeling the {Trade}-off of {Privacy} {Preservation} and {Activity} {Recognition} on {Low}-{Resolution} {Images}},
	isbn = {978-1-4503-9421-5},
	url = {https://dl.acm.org/doi/10.1145/3544548.3581425},
	doi = {10.1145/3544548.3581425},
	abstract = {A computer vision system using low-resolution image sensors can provide intelligent services (e.g., activity recognition) but preserve unnecessary visual privacy information from the hardware level. However, preserving visual privacy and enabling accurate machine recognition have adversarial needs on image resolution. Modeling the trade-off of privacy preservation and machine recognition performance can guide future privacy-preserving computer vision systems using low-resolution image sensors. In this paper, using the at-home activity of daily livings (ADLs) as the scenario, we first obtained the most important visual privacy features through a user survey. Then we quantified and analyzed the effects of image resolution on human and machine recognition performance in activity recognition and privacy awareness tasks. We also investigated how modern image super-resolution techniques influence these effects. Based on the results, we proposed a method for modeling the trade-off of privacy preservation and activity recognition on low-resolution images.},
	booktitle = {Proceedings of the 2023 {CHI} {Conference} on {Human} {Factors} in {Computing} {Systems}},
	publisher = {Association for Computing Machinery},
	author = {Wang, Yuntao and Cheng, Zirui and Yi, Xin and Kong, Yan and Wang, Xueyang and Xu, Xuhai and Yan, Yukang and Yu, Chun and Patel, Shwetak and Shi, Yuanchun},
	year = {2023},
	keywords = {Privacy, activities of daily living, ADLs, low-resolution image, privacy preserving, visual privacy},
	pages = {1--15},
}

```

