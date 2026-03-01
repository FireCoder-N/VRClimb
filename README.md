# VRClimb
## Modelling and Simulation of Rock Climbing in a Virtual Environment

![GitHub Stars](https://img.shields.io/github/stars/FireCoder-N/VRClimb?style=flat-square)
![Thesis PDF](https://img.shields.io/badge/Thesis-PDF-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![Unreal Engine](https://img.shields.io/badge/Unreal_Engine-313131?style=flat-square&logo=unreal-engine&logoColor=white)


<img src="https://github.com/FireCoder-N/VRClimb/blob/main/logo.png?raw=true " alt="VRClimb Logo" width="500"/>

**VRClimb** is an Extended Reality (XR) rock climbing application, initially developed
as a diploma thesis project.

The system focuses on the creation of an immersive environment, accurate reconstruction of a indoor climbing wall via computer vision, realistic climbing
mechanics and real-time mapping of user movement in the virtual environment.

------------------------------------------------------------------------

`Game Development` `Human-Computer Interaction` `Virtual Reality` `Unreal Engine` `XR Interaction` `Rock Climbing`
`C++` `Python` `YOLOv8` `MiDaS` `OpenCV` `HTC Vive` 

------------------------------------------------------------------------

## 📌 Overview

VRClimb is designed to enhance indoor rock climbing using VR system in order to simulate the aesthetically pleasing, immersive experience of outdoor rock climbing.

VRClimb integrates:

- Classical computer vision (OpenCV)
- Deep learning object detection (YOLO)
- Depth sensing & point cloud processing (ZED Stereo Camera)
- Deep learning depth estimation (MiDaS)
- Homography, panorama stitching and image processing
- Pointcloud manipulation and mesh Generation
- A beautiful, realistic and immersive virtual environment (Unreal Engine)
- XR capabilities and movement mapping in a digital twin
- Analysis of Human Kinesiology in order to create an artificial Climbing Instructor

### 🚀 Quick Start
- [Quick-start Guide (English)](https://github.com/FireCoder-N/VRClimb/blob/main/13.Documentation/quickstart_en.md)
- [Quick-start Guide (Greek)](https://github.com/FireCoder-N/VRClimb/blob/main/13.Documentation/quickstart_el.md) or
    1. [Εquipment Setup (Εκκίνηση Εξοπλισμού)](https://github.com/FireCoder-N/VRClimb/tree/main/13.Documentation#%CE%B5%CE%BA%CE%BA%CE%AF%CE%BD%CE%B7%CF%83%CE%B7-%CE%B5%CE%BE%CE%BF%CF%80%CE%BB%CE%B9%CF%83%CE%BC%CE%BF%CF%8D)
    2. [Application Start and Play (Αναρρίχηση στο εικονικό περιβάλλον)](https://github.com/FireCoder-N/VRClimb/tree/main/13.Documentation#%CE%B1%CE%BD%CE%B1%CF%81%CF%81%CE%AF%CF%87%CE%B7%CF%83%CE%B7-%CF%83%CF%84%CE%BF-%CE%AD%CF%84%CE%BF%CE%B9%CE%BC%CE%BF-%CE%B5%CE%B9%CE%BA%CE%BF%CE%BD%CE%B9%CE%BA%CF%8C-%CF%80%CE%B5%CF%81%CE%B9%CE%B2%CE%AC%CE%BB%CE%BB%CE%BF%CE%BD)
- [Detailed Guide (Greek)](https://github.com/FireCoder-N/VRClimb/blob/main/13.Documentation/README.md)

### 📁 Repository Structure

The project evolves through structured experiment folders, culminating in a final demonstrator.

01–03. OpenCV image processing & background removal  
05–06. YOLO training & application  
07–08. ZED depth camera & point cloud experiments  
09–10. Homography projections & panorama stitching  
11. FinishedProject (final demonstrator build)  
12. Results  
13. Documentation  
14. ProjectShowcase 

------------------------------------------------------------------------

## 🎥 Project Showcase

![Gameplay Screenshot](https://github.com/FireCoder-N/VRClimb/blob/main/14.ProjectShowcase/HighresScreenshot00033.png?raw=true)

![Cinematic Scenery](https://github.com/FireCoder-N/VRClimb/blob/main/14.ProjectShowcase/HighresScreenshot00010.png?raw=true)

Additional media and presentation material can be found in folder [`14.ProjectShowcase`](https://github.com/FireCoder-N/VRClimb/blob/main/14.ProjectShowcase)

------------------------------------------------------------------------

## 📚 Documentation

Comprehensive documentation is available in folder [`13.Documentation`](https://github.com/FireCoder-N/VRClimb/blob/main/13.Documentation) as well on the wiki page. The README guide (written in Greek) includes step by step instructions for setting up, using and improving the project.

In order to book a trial for the project inside the VVR-group premises, where it was developed, you have to contact Professor [Konstantinos Moustakas](https://www.ece.upatras.gr/index.php/el/faculty/moustakas) at the Electrical and Computer Engineering Department of the University of Patras.

### 🎓 Thesis Information

This project was initially developed as my Diploma Thesis at the University
of Patras.

##### Thesis Title:
**Modelling and Simulation of Rock Climbing in a Virtual Environment**

##### Thesis contents:
-   Theoretical background on Rock Climbing, VR systems, AI, Computer Vision and Computer Graphics
-   Implementation details
-   Analysis of the User Experience
-   Conclusions and future work

##### Thesis Document:
The full text is available [here](https://nemertes.library.upatras.gr/items/7d1b8cc3-ada4-4f26-b048-3c3ce71b72ba) hosted by the library of the University of Patras.

##### Suggested citation:
> Manolis, D. N. (2025). Modelling and Simulation of Rock Climbing in a Virtual Environment. Diploma thesis, University of Patras.

------------------------------------------------------------------------

## 🏗️ System Architecture

The system is implemented in 4 distinct, but interconnected layers:

- **Environment Module** -- Creation of a Virtual Environment and Scene Setup
- **Climbing Surface Construction Module** -- Digitial recreation of phyisical rock climbing wall
- **VR Interaction Module** -- Handles tracker input and manages user movement from physical to virtual
- **Kinesiology Module** -- Virtual Educator ans Trainer for Rock Climbing exercises

![Architecture Diagram](https://github.com/FireCoder-N/VRClimb/blob/main/13.Documentation/architecture.png?raw=true)

You can find more information about the first three modules in the thesis document as well as the [presentation file](https://github.com/FireCoder-N/VRClimb/blob/main/14.ProjectShowcase/Rock%20Climbing%20in%20VR.pptx), which strictly follows this layered structure.

Among those layers, the recreation of the climbing wall is the most technical one, and can consequently be divided into several steps.

![Wall Reconstruction diagram](https://github.com/FireCoder-N/VRClimb/blob/main/13.Documentation/Wall_reconstruction.png?raw=true)

The kinesiology module is documented in the documentation folder, under the readme guide. A list of all the papers consulted can be found in the [Kinesiology](https://github.com/FireCoder-N/VRClimb/blob/main/13.Documentation/Kinesiology.md) file of the same folder.

### 🛠️ Technologies & Equipment

- Unreal Engine
- Quixel Bridge / FAB Asset Marketplace
- Blender
- C++
- Python
- OpenCV
- Open3d
- MidaS (Monocular Depth Estimation)
- YOLOv8 (Computer vision model for detection, classification, segmentation, etc)
- SteamVR / OpenXR / OpenVR
- Vive Pro Head-Mounted Display (HMDs)
- Vive Trackers (2.0)

For more information you may once again consult the thesis document.

------------------------------------------------------------------------

## 📋 User Evaluation

The system's usability, accuarcy and appeal was evaluated through a structured user questionaire, inspired by the SUS and UEQ questionaires.

Below are presened some prominent results of the evaluation.

![Learnability](https://github.com/FireCoder-N/VRClimb/blob/main/12.Results/c743f874-dab0-4973-9fd9-9132ed7a6de4.png?raw=true)

![Recommendation](https://github.com/FireCoder-N/VRClimb/blob/main/12.Results/362adde5-f64a-499f-b849-c3b0fabb4625.png?raw=true)

Detailed statistical analysis is available in the thesis document. A visual overview of the responses can be found in the [presentation file](https://github.com/FireCoder-N/VRClimb/blob/main/14.ProjectShowcase/Rock%20Climbing%20in%20VR.pptx) as well as the [Questionaire Responses file](https://github.com/FireCoder-N/VRClimb/blob/main/12.Results/response_summary.pdf).

------------------------------------------------------------------------

## 📄 License

This project was developed for academic and research purposes.

This rpository is released under the GNU General Public License (GPL v3). Please refer to the [license file](https://github.com/FireCoder-N/VRClimb/blob/main/LICENSE) for specific licensing details.


### 👤 Author

Developed by FireCoder-N (Dimitrios-Nesotras Manolis)
<br>
Diploma Thesis Project -- University of Patras

**I would love to hear your thoughts and ideas on my project either by contacting me directly or via the [Discussions page](https://github.com/FireCoder-N/VRClimb/discussions)**

------------------------------------------------------------------------

*Last updated: 2026-03-01*
