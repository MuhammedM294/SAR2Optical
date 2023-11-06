SAR2Optical
==============================

This project aims to test the effectiveness of transcoding Synthetic Aperture Radar (SAR) imagery to optical ones using the conditional generative adversarial network (cGAN) technique, and its practicality in waterbody mapping applications using SAR imagery. The main objective is to explore whether this approach can solve the challenge of the similarity of backscattering properties between waterbodies and certain sandy land covers in arid regions in SAR imagery. This similarity creates a perplexing situation for image interpretation. Traditional visual analysis methods often struggle to differentiate between these two land cover types accurately. This leads to ambiguity in identifying and delineating waterbodies and sandy areas, constraining the precision of various applications like environmental monitoring, land use planning, and disaster management.

This project leverages the power of a cutting-edge technique called Conditional Generative Adversarial Network (cGAN) to transform Synthetic Aperture Radar (SAR) images into visually interpretable optical representations. A cGAN is a type of deep learning architecture consisting of two neural networks, a generator, and a discriminator, working together to generate realistic data from a given input. The generator network learns to translate the complex patterns and features present in SAR images into corresponding optical characteristics. Through an adversarial training process, the discriminator evaluates the generated optical images, providing feedback to the generator, which refines its output iteratively. This dynamic interplay between the generator and discriminator results in the generation of realistic and visually coherent optical images from SAR inputs.

## Table of Contents
- [**Study Area**](#study_area)
- [**Experiment 1: Transcoding Sentinel-1 SAR Image VV Band to NIR Band of Sentinel-2**](#vv_b8)
   - [**Dataset**](#b8_dataset)
   - [**Dataset Samples**](#dataset_samples)
   - [**Training Configuration**](#train_config)
   - [**Results Samples**](#b8_result)
      - [**Temporal Variations**](#temp)
      - [**New Study Area**](#new_study)
- [**Experiment 2: Transcoding Sentinel-1 SAR Image VV Band + DEM to NIR Band of Sentinel-2 + Water Mask**](#vv_b8_dem)
  - [**Dataset**](#b8_dem_dataset)
  - [**Training Configuration**](#train_dem_config)
  - [**Results Samples**](#b8__dem_result)
      - [**Temporal Variations**](#temp_dem)
      - [**New Study Area**](#new_study_dem)
  
### Study Area <a name="study_area"></a>

The study area consists of seven distinct water bodies situated within Egypt, spanning across its vast desert landscape. The choice of these waterbodies is deliberate. Their strategic location amidst sandy terrain provides an ideal and complex setting for the research problem.

<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/32a0c77f-b514-40d7-8386-c2ded27852e7" width = "750" height = "600" />

 
### Experiment 1: Transcoding Sentinel-1 SAR Image VV Band to NIR Band of Sentinel-2 <a name ="vv_b8"></a>

The first experiment focuses on transcoding Sentinel-1 (SAR) imagery from the VV (vertical-vertical) polarization band to the Near-Infrared (NIR) band of Sentinel-2 optical imagery. The goal is to explore the feasibility of utilizing NIR data, which is sensitive to water content, to enhance the discrimination between waterbodies and sand landcover in arid regions. 



#### 1.1 Dataset <a name ="b8_dataset"></a>: 
In this experiment, a pair of SAR-Optical datasets were created from the Sentinel-1 SAR VV band  and the NIR band of Sentinel-2 satellite imagery. The training dataset comprises paired images covering the designated study area, acquired in 2020, with a high spatial resolution of 30 meters. The testing consists of two main categories:
1. New study areas: This category includes images from study areas not part of the training dataset to evaluate the model's ability to generalize and produce accurate results in unfamiliar landscapes.
2. Temporal Variations: This included images from the same study area captured in different years to examine how well the model handles changes over time within a specific location. It's a crucial test to ensure the model's consistency and adaptability across various temporal contexts.

#### 1.2 Dataset Samples<a name ="dataset_samples"></a>:
**Source:** Sentinel-1 SAR VV Band (Toushka Lakes, Southern Egypt), Acquisition Date: 06-12-2021
<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/11d51ae8-2734-4925-8589-f31bfbd93a89" width = "800" height = "600" />

**Target:** Sentinel-2 NIR Band (Toushka Lakes, Southern Egypt), Acquisition Date: 04-12-2021
<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/ab0a7f63-cf29-4a8b-9a45-127e40a324c4" width = "800" height = "600" />

#### 1.3 Training Configuration <a name ="train_config"></a>:

The generator network [architecture](https://github.com/MuhammedM294/SAR2Optical/blob/ff48411b85650c46562398f09700244d220a7fbb/src/models/model.py#L49) utilized UNet-style blocks, combining downsampling ([UNetDown](https://github.com/MuhammedM294/SAR2Optical/blob/ff48411b85650c46562398f09700244d220a7fbb/src/models/model.py#L16)) and upsampling ([UNetUp](https://github.com/MuhammedM294/SAR2Optical/blob/ff48411b85650c46562398f09700244d220a7fbb/src/models/model.py#L30)) layers. The downsampling layers incorporated 2D convolutional operations, Leaky Rectified Linear Unit (LeakyReLU) activation with a slope of 0.2, optional dropout for regularization, and instance normalization for stability during training. Upsampling layers utilized transpose convolution operations, ReLU activation, and instance normalization. Skip connections were established between corresponding layers in the encoder and decoder sections, enhancing the flow of gradient information during backpropagation.

The [discriminator](https://github.com/MuhammedM294/SAR2Optical/blob/ff48411b85650c46562398f09700244d220a7fbb/src/models/model.py#L95) network consists of multiple convolutional blocks, each performing downsampling operations. These blocks use 2D convolutional layers, leaky rectified linear unit (LeakyReLU) activation, and optional instance normalization.

| **Component**             | **Architecture**                                                   |
|---------------------------|---------------------------------------------------------------------|
| **Input Channels**        | 1 (Generator), 2 (Discriminator)                                    |
| **Output Channels**       | 1 (for both Generator and Discriminator)                                                  |
| **Patch Size**       | 512 x 512                                                               |
| **Training Epochs**       | 50                                                                  |
| **GPU**                   | NVIDIA GeForce GTX 1660 Ti                                        |
| **Generator**             | UNet Architecture                                                  |
| **Downsampling Layers**   | 2D Convolution, LeakyReLU (slope: 0.2), Optional Dropout, Instance Normalization |
| **Upsampling Layers**     | Transpose Convolution, ReLU Activation, Instance Normalization      |
| **Discriminator**         | Convolutional Blocks with LeakyReLU (slope: 0.2) and Instance Normalization |
| **Optimizer (Generator)** | Adam (lr=0.0002, betas=(0.5, 0.999))                                |
| **Optimizer (Discriminator)** | Adam (lr=0.0002, betas=(0.5, 0.999))                            |




#### 1.4 Results Samples (Patch Level)<a name ="b8_result"></a>:
##### 1.4.1 Temporal Variations Catagory<a name ="temp"></a>:

*Wadi El Rayan Lakes* (Southwest of Cairo, Egypt)

*Sentinel-2*, Acquisition Date: 09-12-2021

<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/0fbcf516-451f-4376-9f4d-7ebccab7520c" width = "750" height = "500" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/50797395-5945-4c67-b4bf-d658fc3a2159" width = "900" height = "300" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/1f32fc8c-eef6-42f6-9e35-0f61395b8355" width = "900" height = "300" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/f534dca0-91f5-4abe-a0e9-2cb3c18bfc36" width = "900" height = "300" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/f8a6c5ca-14b5-4533-9f6a-670bc102a470" width = "900" height = "300" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/a2cba8d0-8215-41dc-9f07-335c57f3511e" width = "900" height = "300" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/2cceeab6-c18f-4b42-81d0-bd558389ce79" width = "900" height = "300" />



In the showcased samples, the generator model effectively translates waterbodies from SAR imagery to the optical NIR band. While successful in this task, challenges persist, particularly in distinguishing certain land cover types, such as sandy areas, which exhibit backscattering patterns similar to water bodies.

##### 1.4.2 **New Study Area Catagory**<a name ="new_study"></a>:

*These lakes are located along Egypt's western border shared with Libya*

*Sentinel-2*, Acquisition Date: 03-12-2021

<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/1d8b4ac4-53f8-466f-841f-d126b8b8baa5" width = "750" height = "500" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/d739a0ff-f649-4b09-ae77-1e537be965ae" width = "900" height = "300" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/1d989bd8-e1f6-4096-9d59-a666e659aa68" width = "900" height = "300" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/03a0dc1e-eec7-485f-87ab-bf8a64dbaddb" width = "900" height = "300" />


In this testing dataset category, the model exhibited poor performance in distinguishing both waterbodies and other land cover types. Challenges still persist in accurately classifying these categories.




### Experiment 2: Transcoding Sentinel-1 SAR Image VV Band + DEM to NIR Band of Sentinel-2 + Water Mask <a name ="vv_b8_dem"></a>

In this experiment, the generator architecture is enhanced by integrating digital elevation data with the VV band of SAR images as inputs. The output includes the NIR band optical data and a water mask extracted from the VV band of the SAR image. This modification represents a significant step forward, combining multiple data sources to improve the accuracy and richness of our model's output.

#### 2.1 Dataset <a name ="b8_dem_dataset"></a>

For this experiment, the same dataset is utilized as in the first study along with incorporating digital elevation models specific to each study area. By integrating these elevation data alongside the VV band of SAR images, our model generated outputs comprising NIR band optical data and water masks extracted from the SAR imagery. 

#### 2.2 Training Configuration <a name ="train_dem_config"></a>

| **Component**             | **Architecture**                                                   |
|---------------------------|---------------------------------------------------------------------|
| **Input Channels**        | 2 (Generator), 4 (Discriminator)                                    |
| **Output Channels**       | 2 (Generator), 1  Discriminator)                                            |
| **Patch Size**       | 512 x 512                                                               |
| **Training Epochs**       | 100                                                                |
| **GPU**                   | NVIDIA GeForce GTX 1660 Ti                                        |
| **Generator**             | UNet Architecture                                                  |
| **Downsampling Layers**   | 2D Convolution, LeakyReLU (slope: 0.2), Optional Dropout, Instance Normalization |
| **Upsampling Layers**     | Transpose Convolution, ReLU Activation, Instance Normalization      |
| **Discriminator**         | Convolutional Blocks with LeakyReLU (slope: 0.2) and Instance Normalization |
| **Optimizer (Generator)** | Adam (lr=0.0002, betas=(0.5, 0.999))                                |
| **Optimizer (Discriminator)** | Adam (lr=0.0002, betas=(0.5, 0.999))                            |

#### 2.3 Results Samples (Patch Level)<a name ="b8__dem_result"></a>:
##### 2.3.1 Temporal Variations Catagory<a name ="temp_dem"></a>:

*Toushka Lakes, Southern Egypt*

Sentinel-2, Acquistation Date 04-12-2021

<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/3cb60e98-038d-4c28-b9a0-a11d4012f40a" width = "750" height = "500" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/85f8b8ee-c2c0-441f-9545-7db3e6bda4ee" width = "1200" height = "200" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/6b01b28a-9e22-4898-bfea-c0869b5b67ab" width = "1200" height = "200" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/82a07522-66bc-4c96-9c4a-633639823943" width = "1200" height = "200" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/5e45021e-cb43-4d5f-a238-ca8375c99568" width = "1200" height = "200" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/e12b55b0-430f-4c54-9429-273ea145a18c" width = "1200" height = "200" />


<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/ee8d4259-ad32-4b57-af00-eb089e4f1442" width = "1200" height = "200" />

Including the digital elevation model in the second dataset and training the model for 100 epochs, double the epochs of the first experiment, unfortunately, does not achieve  significant improvements. While the model effectively classifies most water and land cover types, it struggles with challenging land cover like sand, which shares similar backscatter properties with water bodies. Despite this limitation, the visually accurate generated water masks underscore the superiority of the supervised learning approach over the self-supervised approach (cGAN) utilized in this project.


##### 2.3.2 **New Study Area Catagory**<a name ="new_study_dem"></a>:






--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
