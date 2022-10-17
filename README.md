# Autoencoder for Fairness

The Jupyter Notebook ```autoencoder_mpl.ipynb``` contains the implementation of an <i>Autoencoder Neural Network</i> for the creation of latent spaces representing two types of human body class: <b>class α</b> shapes that possess the most common appearance for the human body and <b>class β</b> shapes, bodies with a limb amputation (e.g. leg or arm).
The network was trained on a dataset containing 1000 instances of both classes included in ```dataset``` directory.

The model architecture comprehends two main AE mirroring one another and two MLP connecting the latent spaces of each AE. AE<sub>α</sub> has an Encoder E<sub>α</sub> with two linear layers followed by tanh activation and a Decoder D<sub>α</sub> with one linear layer followed also by tanh activation. Dimensions are n × 3 → 512 → 256 → 256 → n × 3. We have the same structure for AE<sub>β</sub>, with an Encoder E<sub>β</sub> and a Decoder D<sub>β</sub>. The MLP N<sub>βα</sub> is a network mapping latent space v<sub>β</sub> from AE<sub>β</sub> to latent space v<sub>α</sub> in AE<sub>α</sub>. It has five linear layers followed by SELU activation and a batch normalization layer. Dimensions are f → 64 → 128 → 256 → 128 → 64 → f. MLP N<sub>αβ</sub> similarly has the same structure.

<p align="center"><img width="736" alt="194407227-d4567984-c0dc-4a5a-93c0-883c44dd535a" src="https://user-images.githubusercontent.com/34343511/196158992-71e6f8a0-caee-484f-b4ac-602cefdf2c48.png"></p>

The described network is part of my CS Master's thesis at Sapienza University titled <i>Fairness in Geometry Processing</i>.

## Thesis project

The thesis context is <b>Fair Machine Learning</b>, the study of correcting bias respect to sensitive variables in automated decision processes based on ML models.
Generally current human body model generation methods create human bodies compliant with the standard person capabilities and we have very little material on bodies considered a deviation from the norm. The objective is to work on geometric methods that favor a representation of all human bodies in their diversity.

In particular we focused on the body modeling aspect of Virtual Humans and its creation given by <i>statistical body models</i>. Statistical body models are geometric models that describe human pose and body shape in a unified framework by leveraging an encoding for mesh surfaces; this technique fastly produces a human 3D model with a quite satisfactory level of detail. 
We chose [<b>SMPL</b>](https://smpl.is.tue.mpg.de) as our statistical model.

We generated a 3D dataset with two types of classes: α shapes have the most common appearance for a human body; β shapes reproduce the appearance of a person that no longer possesses a limb. It was important for having a realistic appearance that we did not have a clean cut near the point of amputation, but instead a "smooth" deformation.
Specifically we applied a <b><i>Conformalized Mean Curvature Flow</i></b> and we took [<b>mkazhdan code</b>](https://github.com/mkazhdan/ConformalizedMCF) as a reference, so convergence problems like extreme expansion of the shape were avoided. For further details on this part [go here](https://github.com/luismautone/human-body-mcf).

<p align="center"><img width="1141" alt="Schermata 2022-10-17 alle 12 06 27" src="https://user-images.githubusercontent.com/34343511/196150981-12eeeb9d-7508-406e-a1d1-67a85d75d3ab.png"></p>
<p align="center"><img width="1283" alt="Schermata 2022-10-17 alle 12 06 33" src="https://user-images.githubusercontent.com/34343511/196151017-009fda55-5920-480d-8680-ce8027895ecc.png"></p>

Next we trained an Autoencoder Neural Network on the created dataset in order to obtain a latent space containing the representation for the classes.
Before establishing the details for our model, we performed experiments on a classic AE which had the purpose to create a single latent space for both type of shapes. From the fairness point of view this approach seemed to be a good choice, since  it would have generated a global knowledge of both types of given shapes. Nevertheless this approach presented problems in the shape reconstruction, as highlighted above.

Our model, respect to a classic AE, reconstructs shapes with accuracy and realism: the decoded shape is very much similar to the original input; moreover the model gives coeherent results for the mapping of shapes, since the reconstructed shapes even after this operation are pretty similar in appearance to the correspondent input shape. We arrive to this solution after two separate training on different typologies of models.

### MSE Evaluation

<p align="center"><img width="745" alt="Schermata 2022-10-17 alle 12 56 49" src="https://user-images.githubusercontent.com/34343511/196160500-12507e79-ee18-469f-a182-7c4db7161c04.png"></p>

A single AE trained on both datasets has a worse behavior than our model. Our proposal works pretty fine for reconstructing the shapes and also for mapping. This suggests that it exists a good connection between the latent spaces and there is a good knowledge about the similarity between shapes belonging to different spaces.

### Experiments: Decoding and Mapping

<p align="center"><img width="813" alt="Schermata 2022-10-17 alle 12 58 48" src="https://user-images.githubusercontent.com/34343511/196160801-e2950de0-0e9c-4b90-9818-ff7082b941db.png"></p>

We can notice that the reconstruction of the proposed model is nearly identical to the input shape, proving that our method is working properly and it learned in an effective way. The classic AE, instead, does not provide a mesh with the same identity template of the input and it does not present a likely appearance in the limb reconstruction.
We can also notice that our method is giving visually good results even for the mapping between latent spaces: the latent representation of the class α is mapped to the opposite latent space and its decoding shows a body shape very similar to the input shape β, and viceversa for the other class.

Our model represents a better working solution respect to the classic AE, because producing body shapes with not likely results does not contribute at all to a fair appearance of body models. Our results in decoding and mapping let instead protect the uniqueness in the shapes since the identity of a body is preserved and it is guaranteed a closer representation of the body to reality.

### Experiments: Interpolation

We performed shape interpolation between models from a test dataset and we obtained their mapping to the opposite latent space. An interpolation is given by the following expression:

$$k * v_α + (1−k) * v_β$$

where $k ∈ [0, 1]$, $v_α$ is the latent representation of an input shape of class α and $v<sub>β</sub>$ is the
latent representation of an input shape of class β.

<p align="center"><img width="651" alt="Schermata 2022-10-17 alle 13 02 34" src="https://user-images.githubusercontent.com/34343511/196161494-2b0c6ea1-28a2-4ce7-92be-7d5e4a496683.png"></p>

We have observed from experiments that a classic AE does not permit to move properly from one shape to another in a space with large variations: the interpolated results are unrealistic and uninteresting.
In our model space it is possible instead to move in an Euclidean way: we have an expressive latent space that may find application in a generative model. From a fairness point of view, the expressiveness would facilitate the creation of new types of data and thus it would favor a more inclusive representation within the human body generation methods.

### Latent Space Visualization

We used Principal Component Analysis (PCA) on latent spaces in order to enhance a visual representation of them, plotting the obtained components on a graph. Latent variables are obtained from a given dataset of 400 test shapes (200 from class α, 200 from class β), not seen at training time.

Here: the comparison visualization of latent space v<sub>α</sub> and visualization of the mapping of latent space v<sub>β</sub>; the comparison of visualization of v<sub>β</sub> and visualization of the mapping of v<sub>α</sub>.

<p align="center"><img width="1068" alt="Schermata 2022-10-17 alle 13 38 36" src="https://user-images.githubusercontent.com/34343511/196167798-9a074667-886d-4714-93b4-d95c53e4f140.png"></p>

We can see that, in the space visualizations of our model, the distribution is respected quite well, even if there is not a perfect correspondence between points. In particular, we notice that the difficulty in the matching is mainly present in the contour areas but this behavior does not happen in the center. There are empty spaces both for the latent space generated by the encoder and for that generated by the mapping.

Here the plot of latent space for the classic AE, where we are comparing the display of latent space for α shapes and that for β shapes.

<p align="center"><img width="499" alt="Schermata 2022-10-17 alle 13 43 42" src="https://user-images.githubusercontent.com/34343511/196168797-f7a08d58-f1db-4ec7-b5ef-d9480013cc14.png">
</p>

At the right-most part of the plot, almost half the data from class β are far away from class α data: the network makes a more marked division between the two classes of shapes, shapes of different classes are more different between each other.

Our model has created instead a latent space where there is no evident separation between classes. This aspect testify that our proposal is indeed fair, since shapes from different classes, when living in the same space, are more similar between each other.
