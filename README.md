SAR2Optical
==============================

This project aims to test the effectiveness of transcoding Synthetic Aperture Radar (SAR) imagery to optical ones using the conditional generative adversarial network (cGAN) technique, and its practicality in waterbody mapping applications using SAR imagery. The main objective is to explore whether this approach can solve the challenge of the similarity of backscattering properties between waterbodies and certain sandy land covers in arid regions in SAR imagery. This similarity creates a perplexing situation for image interpretation. Traditional visual analysis methods often struggle to differentiate between these two land cover types accurately. This leads to ambiguity in identifying and delineating waterbodies and sandy areas, constraining the precision of various applications like environmental monitoring, land use planning, and disaster management.

This project leverages the power of a cutting-edge technique called Conditional Generative Adversarial Network (cGAN) to transform Synthetic Aperture Radar (SAR) images into visually interpretable optical representations. A cGAN is a type of deep learning architecture consisting of two neural networks, a generator, and a discriminator, working together to generate realistic data from a given input. The generator network learns to translate the complex patterns and features present in SAR images into corresponding optical characteristics. Through an adversarial training process, the discriminator evaluates the generated optical images, providing feedback to the generator, which refines its output iteratively. This dynamic interplay between the generator and discriminator results in the generation of realistic and visually coherent optical images from SAR inputs.

## Table of Contents
- [**Study Area**](#study_area)
- [**Experiment 1:Transcoding Sentinel-1 SAR Image VV Band to NIR Band of Sentinel-2**](#vv_b8)
   - [**Dataset**](#b8_dataset)

### Study Area <a name="study_area"></a>

The study area consists of seven distinct water bodies situated within Egypt, spanning across its vast desert landscape. The choice of these waterbodies is deliberate. Their strategic location amidst sandy terrain provides an ideal and complex setting for the research problem.

<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/32a0c77f-b514-40d7-8386-c2ded27852e7" width = "750" height = "600" />

 
### Experiment 1: Transcoding Sentinel-1 SAR Image VV Band to NIR Band of Sentinel-2 <a name ="vv_b8"></a>

The first experiment focuses on transcoding Sentinel-1 (SAR) imagery from the VV (vertical-vertical) polarization band to the Near-Infrared (NIR) band of Sentinel-2 optical imagery. The goal is to explore the feasibility of utilizing NIR data, which is sensitive to water content, to enhance the discrimination between waterbodies and sand landcover in arid regions. 


**Our Source: Sentinel-1 SAR VV Band (Toushka Lakes, Southern Egypt), Acquisition Date: 06-12-2021**
<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/11d51ae8-2734-4925-8589-f31bfbd93a89" width = "800" height = "600" />

**Our Target: Sentinel-2 NIR Band (Toushka Lakes, Southern Egypt), Acquisition Date: 04-12-2021**
<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/ab0a7f63-cf29-4a8b-9a45-127e40a324c4" width = "800" height = "600" />

#### 1.1 Dataset <a name ="b8_dataset"></a>: 
In this experiment, a pair of SAR-Optical datasets were created from the Sentinel-1 SAR VV band  and the NIR band of Sentinel-2 satellite imagery. The training dataset comprises paired images covering the designated study area, acquired in 2020, with a high spatial resolution of 30 meters. The testing dataset consists of images captured over the same study area in 2021. The primary goal is to assess the model's capability in identifying temporal changes, particularly focusing on waterbody expansion or shrinkage between the two years.














--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
