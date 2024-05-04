<!--
# CS-766-Project-Webpage
Course webpage for UW Madison's CS 766 – Computer Vision course.
-->
### People and Links
*[Nicholas Russell](https://github.com/russell-nick), University of Wisconsin-Madison, Computer Sciences* <br> <br>
See [here](assets/resources/CS 766 Presentation.pdf){:target="_blank"} for the final presentation, [here](https://github.com/russell-nick/CS-766-Image-Inpainting-Project){:target="_blank"} for the project code, and [here](https://colab.research.google.com/drive/1fLbRK4v8gAwTNMn0_JT8cW2twOn1feD5?usp=sharing){:target="_blank"} for the project demo.

### Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Previous Work](#previous-work)
4. [Methodology](#methodology)
5. [Initial Results](#initial-results)
6. [Experiments](#experiments)
7. [Main Results](#main-results)
8. [Evaluation](#evaluation)
9. [Automatic Object Removal](#automatic-object-removal)
10. [Conclusion](#conclusion)
11. [References](#references)
12. [Supplementary Material](#supplementary-material)

## Introduction
Image inpainting is the task of realistically filling or reconstructing parts of an image that are damaged or missing. Traditional methods seek to solve this problem by using sequential algorithms to fill missing patches, but learning-based methods with CNNs and GANs have become much more popular and generally provide significantly better results. Context Encoders was a very influencial image inpainting model that produces semantically similar results with sharp features. However, it can still generate noisy patches that clearly do not blend into the scene. We seek to build on this by improving the learning-based Context Encoder model with a joint reconstruction loss specifically designed for visual appearance, along with integrating an object detection model to develop a system for automatically removing undesired objects or properties within an image.

## Motivation
As technology advances and the world becomes more digital, there are increasingly many applications for image inpainting. One can use image inpainting to edit photos and remove artifacts, reconstruct damaged regions of physical photographs or paintings, remove unwanted objects from a scene, or recover missing information from occluded regions. This is not only useful to large groups such as a museum recovering damaged artwork or text, but the average person can take advantage of this to enhance image quality and aesthetics in unideal scenarios. An automatic object removal system also allows users to remove undesired image properties without needing any technical expertise or knowledge of image editing software.

## Previous Work
The two main categories of image inpainting techniques are non-learning based or learning based. Popular non-learning based methods include interpolation-based, partial differential equation (PDE)-based, and patch-based inpainting techniques. PatchMatch [1] was a very influential patch-based method that searches for similar patches in the available part of the image, which provided high quality reconstructions while also being efficient.

However, these image inpainting methods require either similar pixels, structures, or patches to be contained within the input image, or required extra information about any holes or missing regions. If the missing region is large, has an arbitrary shape, or contains new or non-repetitive patterns, this constraint can be hard to satisfy. There exist other non-learning based methods that use internet-based retrieval or massive databases of images to search for semantically similar patches [2] to address this issue, but these methods fail when the input image is too different compared to those in the database.

Learning-based results have been much more prominent in recent years and provide solutions to these non-learning based problems. Context Encoders by Pathak *et al.* [5] is a General Adversarial Network (GAN)-based approach for semantic image inpainting that is conditioned to generate missing regions of an image based on the context of the image and surrounding structure. This method uses Deep Convolutional GANs (DCGANs) [6] as segments of the network architecture and also used a novel loss function that has an adversarial weight, along with the reconstruction loss, to produce sharper features. This will be used as our baseline model and more background information is given below.

Shortly after this, Yeh *et al.* [9] proposed a novel image inpainting method using DCGANs that searches for the closest encoding of the corrupted image in the latent space, which does not require the masks for training and can be applied for arbitrarily structured missing regions during inference.

Newer learning-based methods include using coarse-to-fine frameworks with contextual attention layers (DeepFill) [10], fast Fourier convolutions (LaMa) [8], and cascaded modulation with Fourier convolution blocks (CM-GAN) [12].

### Context Encoders Background
The Context Encoders model seeks to reconstruct more realistic images by attempting to understand the context of the image based on the surrounding structure. The basic architecture uses an encoder and decoder pipeline as shown below. The encoder takes an input image with missing regions and produces a latent feature representation of that image, which the decoder uses to produce the missing image content.
<p align="center">
<img width="800" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Context Encoders/context_encoder_architecture.png">
  <center><em>Context Encoders Architecture [5] </em></center>
</p>

An important aspect of Context Encoders is the loss function. Instead of training the model with standard pixel-wise reconstruction loss, an adversarial loss factor is added to produce much sharper features. Context Encoders uses the loss function $$L = \lambda_{adv}\mathcal{L}_{adv} + \lambda_{rec}\mathcal{L}_{rec}$$ with Binary Cross-Entropy (BCE) Adversarial and L2 Reconstruction losses.

<p align="center">
<img width="600" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Context Encoders/Context Encoders Example.png">
  <center><em>Context Encoders Example [5]</em></center>
</p>

Although the adversarial loss adds sharp features, these features are often incoherent or have semantically little to do with the input image (as shown below from [5]). Combining this with a reconstruction loss that offers good structure but smooth or blurry images results in a an image with both good structure and coherent, sharp features.
<div align="center">
  <figure style="display:inline-block;padding-right:15px">
      <img width="250" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Context Encoders/context_encoder_masked.png">
      <figcaption style="text-align:center;">Input</figcaption>
  </figure>
  <figure style="display:inline-block;">
      <img width="250" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Context Encoders/context_encoder_L2.png">
      <figcaption style="text-align:center;">L2</figcaption>
  </figure> 
</div>
<div align="center">
  <figure style="display:inline-block;padding-right:15px">
      <img width="250" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Context Encoders/context_encoder_adv.png">
      <figcaption style="text-align:center;">Adversarial</figcaption>
  </figure>
  <figure style="display:inline-block;">
      <img width="250" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Context Encoders/context_encoder_joint.png">
      <figcaption style="text-align:center;">Joint</figcaption>
  </figure>
</div>

## Methodology
Our goal is to improve the visual apperance of regions generated by Context Encoders and to integrate object detection and segmentation with image inpainting techniques to allow for the automatic removal of many object types in images.

To detect objects, segment objects, and create masks, we are using the YOLOv8 model [3], which is the newest version of the state-of-the-art You Only Look Once (YOLO) model [7]. This model offers real-time object detection with high accuracy. Given an input image and a list of object types of remove, our integrated model will use YOLOv8 to create masks of the specified objects, which will be the pixels to “remove” for the image inpainting process.

We use Context Encoders [5] as the baseline image inpainting models, which was chosen due to its influence and impact on image inpainting tasks.

Below is a list of experiments and modifications that we made:
- Experimented with reconstruction metrics and implemented different joint reconstruction losses to achieve better visual results (exact details given in [Experiments](#experiments).)
- Modified the model for automatic object removal:
  - Modified the baseline Context Encoders model to support evaluation with arbitrary masks.
  - Integrated YOLOv8 with Context Encoders to generate masks to automatically remove objects from the scene.


### Datasets
For the image inpainting task, we used the MiniPlaces dataset, which is a subset of MIT’s Places dataset [13] that only contains 100,000 128x128 images from 100 scene categories.
<p align="center">
<img width="800" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/teaser.jpg">
  <center><em>MiniPlaces Example Images</em></center>
</p>

We used a pre-trained YOLOv8 model for object segmentation and mask extraction due to resource constraints. This model was trained on Microsoft's COCO dataset [4], a large-scale object detection, segmentation, and captioning dataset with 2.5 million labeled object instances in 328k images. Microsoft COCO supports the detection of 81 object types. 

### Evaluation Metrics
Since the image inpainting model is generating pixels to replace missing areas of an image, we want the output to be as realistic and convincing as possible. Therefore, the main method of evaluation will be through comparing the visual results by displaying the input images, the masked images, and the output images. However, we would still like to measure some performance quantitatively and use certain metrics to shape the way our model learns to fill missing regions. 

The standard L1 (MAE) and L2 (MSE) losses are used:
<p align="center">
  <img style="padding-right:40px;" width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L1 Loss.png">
  <img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L2 Loss.png">
</p>

In addition to these, two standard quantitative metrics for measuring image inpainting performance are given below: <br>
- Peak signal-to-noise ratio (PSNR): Measures the quality of image reconstruction.
  <p align="center">
    <img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/PSNR.png">
  </p>
- Structural Similarity Index (SSIM):  Measures the visual similarity between images.
  <p align="center">
    <img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/SSIM.png">
  </p>
  
## Initial Results
The first experiment was changing the reconstruction loss from L2 to L1 since it has been reported that this helps to obtain better results for image generation [11]. Then the adversarial loss was changed to L2. These two changes produce a joint loss function that generates smoother and coherent regions, at the downside of being blurrier on images with more detail. 

The Context Encoders baseline and this experimental L2 adversarial and L1 reconstruction loss model were trained for 40 epochs on the MiniPlaces dataset. To compare the visual results between the baseline model and the modified losses model, 64 images were manually selected from the MiniPlaces validation set on a variety of different scenes. 

<!---The results can be seen in Figures 3 and 4. Since L2 and L1 were used as reconstruction losses, we also measured the L2 and L1 loss of each reconstructed image (shown in Figure 5).-->

<p align="center">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/original_grid.png">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/masked_grid.png">
  <center><em>The left side shows 64 chosen validation images and the right shows the masked input.</em></center>
</p>

<p align="center">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/bce_inpainted_images_grid.png">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_images_grid.png">
  <center><em>Left: Baseline Context Encoders model. Right: L2 Adversarial + L1 Reconstruction model</em></center>
</p>

A few examples are chosen to demonstrate some of the patchiness / artifacts present in the baseline Context Encoders model and to show the importance of using an adversarial loss with no smoothing effect in order to keep the sharp features produced by the main Context Encoders loss function.

<p align="center">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00000064.jpg">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00000119.jpg">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00000064.jpg">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00000119.jpg">
  <center><em>Top: Base Context Encoders. Bottom: $\mathcal{L}_{adv}=\mathcal{L}_{L2},\mathcal{L}_{rec}=\mathcal{L}_{L1}$.</em></center>
</p>

<p align="center">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00000618.jpg">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00003823.jpg">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00000618.jpg">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00003823.jpg">
  <center><em>Top: Base Context Encoders. Bottom: $\mathcal{L}_{adv}=\mathcal{L}_{L2},\mathcal{L}_{rec}=\mathcal{L}_{L1}$.</em></center>
</p>

<p align="center">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00004299.jpg">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00004560.jpg">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00004299.jpg">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00004560.jpg">
  <center><em>Top: Base Context Encoders. Bottom: $\mathcal{L}_{adv}=\mathcal{L}_{L2},\mathcal{L}_{rec}=\mathcal{L}_{L1}$.</em></center>
</p>


This initial result gives motivation for the below experiments which produce our main results.

## Experiments
We want to bring back sharp edges, but create smoother results with less noise than some of the patches created by the baseline Context Encoders model. Our proposed solution is to take inspiration from Context Encoder's joint adversarial and reconstruction loss function and implement a joint reconstruction loss that is designed for visual appearance: 
<p>$$\lambda_{rec1}\mathcal{L}_{rec1}+\lambda_{rec2}\mathcal{L}_{rec2}.$$ </p>

Specifically, we will use a factor of SSIM, which measures not only the structural similarity between images, but also measures the difference in luminance and contrast. Therefore, our proposed loss function is 
<p>$$\mathcal{L}=\lambda_{adv}\mathcal{L}_{adv}+\lambda_{rec}(\lambda_{L1}\mathcal{L}_{L1}+\lambda_{SSIM}(1-\mathcal{L}_{SSIM})).$$ </p>

Note that the factor of $$(1-\mathcal{L}_{SSIM})$$ is used since we want to maximize SSIM. We still use Binary Cross-Entropy adversarial loss with $\lambda_{adv} = 0.005$ and $\lambda_{rec} = 0.995$ as suggested by Pathak *et al.* [5].

## Main Results
Now we demonstrate the effectiveness of our proposed solution. From what was tested, $\lambda_{L1}=0.5, \lambda_{SSIM}=0.5$ gives the best results, but different factors of SSIM could potentially perform better (probably with SSIM slightly less than 0.5, but still significant enough). First we show a side-by-side comparison of the Context Encoders model and our variation with the joint reconstruction loss designed for visual appearances.

<p align="center">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/bce_inpainted_images_grid.png">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_images_grid.png">
  <center><em>Left: Baseline Context Encoders model. Right: $\lambda_{L1}=0.5, \lambda_{SSIM}=0.5$ model</em></center>
</p>

As before, a few examples are chosen to demonstrate the visual improvements that our proposed solution provides. Our proposed model with $\lambda_{L1}=0.5, \lambda_{SSIM}=0.5$ is highlighted, as it has significant visual improvements in the majority of images.

<p align="center">
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/baseline_context_encoders.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00004047.jpg">
  <span style="display:block;float:right;"> L1: 0.0774 <br> L2: 0.0386 <br> PSNR: 14.1375 <br> SSIM: 0.7189</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L2_L1_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00004047.jpg">
  <span style="display:block;float:right;"> L1: 0.0698 <br> L2: 0.0347 <br> PSNR: 14.5935 <br> SSIM: 0.7403</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.3L1_0.7SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.3L1 + 0.7SSIM/inpainted_00004047.jpg">
  <span style="display:block;float:right;"> L1: 0.0734 <br> L2: 0.0396 <br> PSNR: 14.0252 <br> SSIM: 0.7287</span> <br>
<img style="border: 3px solid lawngreen" width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.5L1_0.5SSIM_Loss.png">
  <img style="padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_00004047.jpg">
  <span style="display:block;float:right;"> L1: 0.0744 <br> L2: 0.0400 <br> PSNR: 13.9811 <br> SSIM: 0.7345</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.005L1_0.995SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.005L1 + 0.995SSIM/inpainted_00004047.jpg">
  <span style="display:block;float:right;"> L1: 0.0844 <br> L2: 0.0552 <br> PSNR: 12.5826 <br> SSIM: 0.7232</span> <br>
  <br>
</p>

<p align="center">
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/baseline_context_encoders.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00000119.jpg">
  <span style="display:block;float:right;"> L1: 0.0551 <br> L2: 0.0199 <br> PSNR: 17.0017 <br> SSIM: 0.7063</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L2_L1_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00000119.jpg">
  <span style="display:block;float:right;"> L1: 0.0511 <br> L2: 0.0180 <br> PSNR: 17.4431 <br> SSIM: 0.7164</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.3L1_0.7SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.3L1 + 0.7SSIM/inpainted_00000119.jpg">
  <span style="display:block;float:right;"> L1: 0.0562 <br> L2: 0.0221 <br> PSNR: 16.5612 <br> SSIM: 0.7139</span> <br>
<img style="border: 3px solid lawngreen" width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.5L1_0.5SSIM_Loss.png">
  <img style="padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_00000119.jpg">
  <span style="display:block;float:right;"> L1: 0.0560 <br> L2: 0.0213 <br> PSNR: 16.7069 <br> SSIM: 0.7128</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.005L1_0.995SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.005L1 + 0.995SSIM/inpainted_00000119.jpg">
  <span style="display:block;float:right;"> L1: 0.0661 <br> L2: 0.0298 <br> PSNR: 15.2601 <br> SSIM: 0.7060</span> <br>
  <br>
</p>

<p align="center">
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/baseline_context_encoders.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00003823.jpg">
  <span style="display:block;float:right;"> L1: 0.0509 <br> L2: 0.0209 <br> PSNR: 16.8047 <br> SSIM: 0.7701</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L2_L1_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00003823.jpg">
  <span style="display:block;float:right;"> L1: 0.0415 <br> L2: 0.0181 <br> PSNR: 17.4288 <br> SSIM: 0.8060</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.3L1_0.7SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.3L1 + 0.7SSIM/inpainted_00003823.jpg">
  <span style="display:block;float:right;"> L1: 0.0507 <br> L2: 0.0224 <br> PSNR: 16.4965 <br> SSIM: 0.7920</span> <br>
<img style="border: 3px solid lawngreen" width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.5L1_0.5SSIM_Loss.png">
  <img style="padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_00003823.jpg">
  <span style="display:block;float:right;"> L1: 0.0448 <br> L2: 0.0205 <br> PSNR: 16.8840 <br> SSIM: 0.8065</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.005L1_0.995SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.005L1 + 0.995SSIM/inpainted_00003823.jpg">
  <span style="display:block;float:right;"> L1: 0.0522 <br> L2: 0.0290 <br> PSNR: 15.3750 <br> SSIM: 0.7975</span> <br>
  <br>
</p>

<p align="center">
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/baseline_context_encoders.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00002919.jpg">
  <span style="display:block;float:right;"> L1: 0.0691 <br> L2: 0.0296 <br> PSNR: 15.2889 <br> SSIM: 0.7111</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L2_L1_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00002919.jpg">
  <span style="display:block;float:right;"> L1: 0.0648 <br> L2: 0.0268 <br> PSNR: 15.7234 <br> SSIM: 0.7194</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.3L1_0.7SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.3L1 + 0.7SSIM/inpainted_00002919.jpg">
  <span style="display:block;float:right;"> L1: 0.0640 <br> L2: 0.0279 <br> PSNR: 15.5403 <br> SSIM: 0.7256</span> <br>
<img style="border: 3px solid lawngreen" width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.5L1_0.5SSIM_Loss.png">
  <img style="padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_00002919.jpg">
  <span style="display:block;float:right;"> L1: 0.0669 <br> L2: 0.0289 <br> PSNR: 15.3979 <br> SSIM: 0.7197</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.005L1_0.995SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.005L1 + 0.995SSIM/inpainted_00002919.jpg">
  <span style="display:block;float:right;"> L1: 0.0789 <br> L2: 0.0415 <br> PSNR: 13.8219 <br> SSIM: 0.7187</span> <br>
  <br>
</p>

<p align="center">
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/baseline_context_encoders.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00004299.jpg">
  <span style="display:block;float:right;"> L1: 0.0652 <br> L2: 0.0277 <br> PSNR: 15.5769 <br> SSIM: 0.7183</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L2_L1_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00004299.jpg">
  <span style="display:block;float:right;"> L1: 0.0596 <br> L2: 0.0244 <br> PSNR: 16.1256 <br> SSIM: 0.7237</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.3L1_0.7SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.3L1 + 0.7SSIM/inpainted_00004299.jpg">
  <span style="display:block;float:right;"> L1: 0.0613 <br> L2: 0.0277 <br> PSNR: 15.5784 <br> SSIM: 0.7278</span> <br>
<img style="border: 3px solid lawngreen" width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.5L1_0.5SSIM_Loss.png">
  <img style="padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_00004299.jpg">
  <span style="display:block;float:right;"> L1: 0.0607 <br> L2: 0.0260 <br> PSNR: 15.8559 <br> SSIM: 0.7234</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.005L1_0.995SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.005L1 + 0.995SSIM/inpainted_00004299.jpg">
  <span style="display:block;float:right;"> L1: 0.0662 <br> L2: 0.0313 <br> PSNR: 15.0510 <br> SSIM: 0.7267</span> <br>
  <br>
</p>

<p align="center">
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/baseline_context_encoders.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00004560.jpg">
  <span style="display:block;float:right;"> L1: 0.0740 <br> L2: 0.0358 <br> PSNR: 14.4571 <br> SSIM: 0.7388</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L2_L1_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00004560.jpg">
  <span style="display:block;float:right;"> L1: 0.0688 <br> L2: 0.0389 <br> PSNR: 14.1060 <br> SSIM: 0.7421</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.3L1_0.7SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.3L1 + 0.7SSIM/inpainted_00004560.jpg">
  <span style="display:block;float:right;"> L1: 0.0724 <br> L2: 0.0417 <br> PSNR: 13.8002 <br> SSIM: 0.7382</span> <br>
<img style="border: 3px solid lawngreen" width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.5L1_0.5SSIM_Loss.png">
  <img style="padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_00004560.jpg">
  <span style="display:block;float:right;"> L1: 0.0734 <br> L2: 0.0464 <br> PSNR: 13.3350 <br> SSIM: 0.7479</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.005L1_0.995SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.005L1 + 0.995SSIM/inpainted_00004560.jpg">
  <span style="display:block;float:right;"> L1: 0.0811 <br> L2: 0.0553 <br> PSNR: 12.5762 <br> SSIM: 0.7232</span> <br>
  <br>
</p>

<p align="center">
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/baseline_context_encoders.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00000280.jpg">
  <span style="display:block;float:right;"> L1: 0.0783 <br> L2: 0.0360 <br> PSNR: 14.4411 <br> SSIM: 0.7140</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L2_L1_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00000280.jpg">
  <span style="display:block;float:right;"> L1: 0.0846 <br> L2: 0.0468 <br> PSNR: 13.2932 <br> SSIM: 0.7185</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.3L1_0.7SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.3L1 + 0.7SSIM/inpainted_00000280.jpg">
  <span style="display:block;float:right;"> L1: 0.0941 <br> L2: 0.0572 <br> PSNR: 12.4256 <br> SSIM: 0.7129</span> <br>
<img style="border: 3px solid lawngreen" width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.5L1_0.5SSIM_Loss.png">
  <img style="padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_00000280.jpg">
  <span style="display:block;float:right;"> L1: 0.0859 <br> L2: 0.0474 <br> PSNR: 13.2443 <br> SSIM: 0.7118</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.005L1_0.995SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.005L1 + 0.995SSIM/inpainted_00000280.jpg">
  <span style="display:block;float:right;"> L1: 0.1020 <br> L2: 0.0716 <br> PSNR: 11.4505 <br> SSIM: 0.7136</span> <br>
  <br>
</p>

<p align="center">
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/baseline_context_encoders.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/inpainted_00006461.jpg">
  <span style="display:block;float:right;"> L1: 0.0954 <br> L2: 0.0521 <br> PSNR: 12.8282 <br> SSIM: 0.7108</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/L2_L1_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_00006461.jpg">
  <span style="display:block;float:right;"> L1: 0.0918 <br> L2: 0.0516 <br> PSNR: 12.8775 <br> SSIM: 0.7213</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.3L1_0.7SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.3L1 + 0.7SSIM/inpainted_00006461.jpg">
  <span style="display:block;float:right;"> L1: 0.0962 <br> L2: 0.0609 <br> PSNR: 12.1527 <br> SSIM: 0.7269</span> <br>
<img style="border: 3px solid lawngreen" width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.5L1_0.5SSIM_Loss.png">
  <img style="padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_00006461.jpg">
  <span style="display:block;float:right;"> L1: 0.0915 <br> L2: 0.0543 <br> PSNR: 12.6535 <br> SSIM: 0.7365</span> <br>
<img width="200" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Loss Functions/0.005L1_0.995SSIM_Loss.png">
  <img style="padding-left:6px;padding-right:20px;" width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.005L1 + 0.995SSIM/inpainted_00006461.jpg">
  <span style="display:block;float:right;"> L1: 0.1137 <br> L2: 0.0873 <br> PSNR: 10.5899 <br> SSIM: 0.7164</span> <br>
  <center><em>Very high $\lambda_{SSIM}$ in addition to adversarial loss often results in weird results or artifacts.</em></center>
</p>

## Evaluation
To evaluate the general performance of our models, every model was evaluated on the MiniPlaces validation dataset containing 10,000 128x128 images from 100 different scene categories. Each model was trained for 40 epochs. All methods, except the second ($$\mathcal{L}_{adv}=\mathcal{L}_{L2}, \mathcal{L}_{rec}=\mathcal{L}_{L1}$$), use BCE adversarial loss. 
<!--- We also tested whether replacing L1 reconstruction loss with L2 -->

<p align="center">
<img width="700" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Model Losses Table Original.png">
  <center><em> Table of losses per model</em></center>
</p>

From the above table, we can see that the L2 Adversarial + L1 Reconstruction model performs best quantitatively. However, this model does not produce the most visually appealing images and usually results in blurry results without sharp features. We can see that our proposed joint reconstruction model with $\lambda_{L1}=0.5, \lambda_{SSIM}=0.5$ produces not only significant visual improvements, but also slight quantitative improvements. The change of SSIM scores from 0.7385 to 0.7512 gives a 1.72% improvement over the baseline, but the PSNR score decreases by 2.08%. Therefore, this method has a tradeoff: increased SSIM performance but decreased PSNR performance. However, the SSIM score is preferred in our case, as it produces more visually appealing and convincing results. 

Note that all models are only trained for 40 epochs while they are trained for 500 by Pathak *et al.* in [5]. It would be interesting to see how significant both the visual and quantitative differences are if all models are trained until convergence since the visual results are significant for training 12.5x less than required for convergence. Training for 40 epochs on the MiniPlaces dataset takes approximately 9 hours on a NVIDIA T4 GPU on Google Colab.

## Automatic Object Removal
### Mask Extraction
To automatically remove objects from a scene, we must first detect the objects we want to remove and extract their masks. This is done by using YOLOv8 and combining the masks for all the object types we want to remove.
<p align="center">
<img width="275" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Object Removal/car_scene.jpg">
<img width="275" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Object Removal/predictions.jpg">
<img width="275" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Object Removal/objects-to-remove-masks.jpg">
  <center><em>Extracting Mask of People and Cars.</em></center>
</p>

### Automatic Object Removal Examples
A few examples of automatically detecting and removing objects from images are given below. The model being used is the proposed model with $\lambda_{L1}=0.5, \lambda_{SSIM}=0.5$. Note that this model is only trained for 40 epochs, which is 12.5x less than in [5], so the model has not converged. Therefore, the object outlines can still be seen in the images.

<p align="center">
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Object Removal/inpainted_beach.jpg"><br>
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Object Removal/inpainted_zoo.jpg"><br>
<img width="400" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Object Removal/inpainted_cliff.jpg"><br>
  <center><em>Automatically Removing Objects.</em></center>
</p>
The first two images were taken from the internet and the last is a personal image from "Potato Chip Rock" near San Diego, CA.

### Object Removal Limitations
Currently, removing objects with arbitrary masks sometimes gives strange artifacts or inpaints a near solid color (especially on small objects). However, inpainting a rectangular region on the same image still produces results as expected. A potential solution for this would be to implement random masking when training the model. The image inpainting model is currently trained on random rectangular regions of a fixed size.

<p align="center">
<img width="500" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Object Removal/inpainted_IMG_5101.jpg"><br>
<img width="500" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Object Removal/inpainted_IMG_5101_rectangular.jpg"><br>
  <center><em>Limitations of object removal (3rd image is the model's entire generated output).</em></center>
</p>


## Conclusion
In this project, we implemented the baseline Context Encoders model, made model changes to improve visual results, and integrated with an object detection model to automatically detect and remove objects without having to manually draw masks. To modify the baseline Context Encoders model to support arbitrary masks in evaluation, an extra upsampling layer was added and the training and evaluation code was modified to extract arbitrary masks.

The main improvement to the Context Encoders model was the following loss function: 
<p>$$\mathcal{L}=\lambda_{adv}\mathcal{L}_{adv}+\lambda_{rec}(\lambda_{L1}\mathcal{L}_{L1}+\lambda_{SSIM}(1-\mathcal{L}_{SSIM})).$$ </p>
This resulted in significant visual improvements in most sample images and gave a slight 1.72% increase in SSIM scores across the validation dataset (this might be more significant if models are trained to convergence). 

## References
[1] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman. “PatchMatch: A Randomized
Correspondence Algorithm for Structural Image Editing”. In: *ACM Transactions on Graphics (Proc. SIG-
GRAPH)* 28.3 (Aug. 2009).

[2] James Hays and Alexei A Efros. “Scene Completion Using Millions of Photographs”. In: *ACM Transactions on Graphics (SIGGRAPH 2007)* 26.3 (2007).

[3] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. *Ultralytics YOLO.* Version 8.0.0. Jan. 2023. url: https://github.com/ultralytics/ultralytics

[4] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll ́ar,
and C. Lawrence Zitnick. “Microsoft COCO: Common Objects in Context”. In: *Computer Vision – ECCV
2014.* Ed. by David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars. Cham: Springer International
Publishing, 2014, pp. 740–755. isbn: 978-3-319-10602-1. 

[5] Deepak Pathak, Philipp Kr ̈ahenb ̈uhl, Jeff Donahue, Trevor Darrell, and Alexei Efros. “Context Encoders: Feature Learning by Inpainting”. In: *Computer Vision and Pattern Recognition (CVPR).* 2016. 

[6] Alec Radford, Luke Metz, and Soumith Chintala. *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* 2016. arXiv: 1511.06434 [cs.LG].

[7] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. “You Only Look Once: Unified, Real-Time Object Detection”. In: *IEEE Conference on Computer Vision and Pattern Recognition (CVPR).* 2016,
pp. 779–788. doi: 10.1109/CVPR.2016.

[8] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. “Resolution-robust Large Mask
Inpainting with Fourier Convolutions”. In: *arXiv preprint arXiv:2109.07161* (2021).

[9] Raymond A. Yeh$^{\*}$, Chen Chen$^{\*}$, Teck Yian Lim, Schwing Alexander G., Mark Hasegawa-Johnson, and Minh N. Do. “Semantic Image Inpainting with Deep Generative Models”. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.* $^{\*}$ equal contribution. 2017.

[10] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. “Generative Image Inpainting with Contextual Attention”. In: *arXiv preprint arXiv:1801.07892* (2018).

[11] Hang Zhao, Orazio Gallo, Iuri Frosio, and Jan Kautz. “Loss Functions for Image Restoration With Neural Networks”. In: *IEEE Transactions on Computational Imaging 3.1* (2017), pp. 47–57. doi: 10.1109/TCI. 2016.2644865.

[12] Haitian Zheng, Zhe Lin, Jingwan Lu, Scott Cohen, Eli Shechtman, Connelly Barnes, Jianming Zhang, Ning
Xu, Sohrab Amirghodsi, and Jiebo Luo. “CM-GAN: Image Inpainting with Cascaded Modulation GAN and Object-Aware Training”. In: *arXiv preprint arXiv:2203.11947* (2022).

[13] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. “Places: A 10 million
Image Database for Scene Recognition”. In: *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2017)

## Supplementary Material
Extra images are provided below in larger versions than above:

<p align="center">
<img width="800" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/Baseline/bce_inpainted_images_grid.png">
  <center><em>Baseline Context Encoders Results</em></center>
</p>

<p align="center">
<img width="800" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/L2 Adv + L1 Recon/inpainted_images_grid.png">
  <center><em>$\mathcal{L}_{adv}=\mathcal{L}_{L2},\mathcal{L}_{rec}=\mathcal{L}_{L1}$ Results</em></center>
</p>

<p align="center">
<img width="800" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.3L1 + 0.7SSIM/inpainted_images_grid.png">
  <center><em>$\mathcal{L}_{rec}=0.3\mathcal{L}_{L1}+0.7\mathcal{L}_{SSIM}$ Results</em></center>
</p>

<p align="center">
<img width="800" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.5L1 + 0.5SSIM/inpainted_images_grid.png">
  <center><em><b>$\mathcal{L}_{rec}=0.5\mathcal{L}_{L1}+0.5\mathcal{L}_{SSIM}$ Results</b></em></center>
</p>

<p align="center">
<img width="800" src="https://raw.githubusercontent.com/russell-nick/CS-766-Image-Inpainting-Project/gh-pages/assets/images/Inpainting_Outputs/0.005L1 + 0.995SSIM/inpainted_images_grid.png">
  <center><em>$\mathcal{L}_{rec}=0.005\mathcal{L}_{L1}+0.995\mathcal{L}_{SSIM}$ Results</em></center>
</p>
