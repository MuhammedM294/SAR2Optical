SAR2Optical
==============================

This project aims to test the effectiveness of transcoding Synthetic Aperture Radar (SAR) imagery to optical ones, and its practicality in waterbody mapping applications. The main objective is to solve the challenge of the similarity of backscattering properties between waterbodies and certain sandy land covers in arid regions in SAR imagery. This similarity creates a perplexing situation for image interpretation. Traditional visual analysis methods often struggle to differentiate between these two land cover types accurately. This leads to ambiguity in identifying and delineating waterbodies and sandy areas, constraining the precision of various applications like environmental monitoring, land use planning, and disaster management.

## Table of Contents
- [**Study Area**](#study_area)
- 

### Study Area <a name="study_area"></a>

The study area consists of seven distinct water bodies situated within Egypt, spanning across its vast desert landscape. The choice of these waterbodies is deliberate. Their strategic location amidst sandy terrain provides an ideal and complex setting for the research problem.

<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/32a0c77f-b514-40d7-8386-c2ded27852e7" width = "750" height = "600" />

 
### Experiment 1: Transcoding Sentinel-1 SAR Image VV Band to NIR Band of Sentinel-2

The first experiment focuses on transcoding Sentinel-1 (SAR) imagery from the VV (vertical-vertical) polarization band to the Near-Infrared (NIR) band of Sentinel-2 optical imagery. The goal is to explore the feasibility of utilizing NIR data, which is sensitive to water content, to enhance the discrimination between waterbodies and sand landcover in arid regions. 


**Sentinel-1 SAR VV Band (Toushka Lakes, Southern Egypt), Acquisition Date: 06-12-2021**
<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/11d51ae8-2734-4925-8589-f31bfbd93a89" width = "800" height = "600" />

**Sentinel-2 NIR Band (Toushka Lakes, Southern Egypt), Acquisition Date: 04-12-2021**
<img src= "https://github.com/MuhammedM294/SAR2Optical/assets/89984604/ab0a7f63-cf29-4a8b-9a45-127e40a324c4" width = "800" height = "600" />














--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
