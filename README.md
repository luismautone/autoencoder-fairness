# Autoencoder for Fairness

The Jupyter Notebook ```autoencoder_mpl.ipynb``` contains the implementation of an <i>Autoencoder Neural Network</i> for the creation of latent spaces representing two types of human body class: <b>class α</b> shapes that possess the most common appearance for the human body and <b>class β</b> shapes, bodies with a limb amputation (e.g. leg or arm).
The network was trained on a dataset containing 1000 instances of both classes (included in ```dataset``` directory) and the trained model was saved as ```ae_model.pt```.

The model architecture comprehends two main AE mirroring one another and two MLP connecting the latent spaces of each AE. AE<sub>α</sub> has an Encoder E<sub>α</sub> with two linear layers followed by tanh activation and a Decoder D<sub>α</sub> with one linear layer followed also by tanh activation. Dimensions are n × 3 → 512 → 256 → 256 → n × 3. We have the same structure for AE<sub>β</sub>, with an Encoder E<sub>β</sub> and a Decoder D<sub>β</sub>. The MLP N<sub>βα</sub> is a network mapping latent space v<sub>β</sub> from AE<sub>β</sub> to latent space v<sub>α</sub> in AE<sub>α</sub>. It has five linear layers followed by SELU activation and a batch normalization layer. Dimensions are f → 64 → 128 → 256 → 128 → 64 → f. MLP N<sub>αβ</sub> similarly has the same structure.

<p align="center"><img width="736" alt="194407227-d4567984-c0dc-4a5a-93c0-883c44dd535a" src="https://user-images.githubusercontent.com/34343511/196158992-71e6f8a0-caee-484f-b4ac-602cefdf2c48.png"></p>

The described network is part of my CS Master's thesis at Sapienza University titled <i>Fairness in Geometry Processing</i>.

## Thesis project

The thesis context is <b>Fair Machine Learning</b>, the study of correcting bias respect to sensitive variables in automated decision processes based on ML models.
Generally current human body model generation methods create human bodies compliant with the standard person capabilities and we have very little material on bodies considered a deviation from the norm. The objective is to work on geometric methods that favor a representation of all human bodies in their diversity.

In particular we focused on the body modeling aspect of Virtual Humans and its creation given by <i>statistical body models</i>. Statistical body models are geometric models that describe human pose and body shape in a unified framework by leveraging an encoding for mesh surfaces; this technique fastly produces a human 3D model with a quite satisfactory level of detail. 
We chose [<b>SMPL</b>](https://smpl.is.tue.mpg.de) as our statistical model.

We generated a 3D dataset with two types of classes: α shapes have the most common appearance for a human body; β shapes reproduce the appearance of a person that no longer possesses a limb. It was important for having a realistic appearance that we did not have a clean cut near the point of amputation, but instead a "smooth" deformation.
Specifically we applied a <b><i>Conformalized Mean Curvature Flow</i></b> and we took [<b>mkazhdan code</b>](https://github.com/mkazhdan/ConformalizedMCF) as a reference, so convergence problems like extreme expansion of the shape were avoided. For further details on this part [go here](https://github.com/luismautone/MCF-FairnessGeometryProcessing).

<p align="center"><img width="1141" alt="Schermata 2022-10-17 alle 12 06 27" src="https://user-images.githubusercontent.com/34343511/196150981-12eeeb9d-7508-406e-a1d1-67a85d75d3ab.png"></p>
<p align="center"><img width="1283" alt="Schermata 2022-10-17 alle 12 06 33" src="https://user-images.githubusercontent.com/34343511/196151017-009fda55-5920-480d-8680-ce8027895ecc.png"></p>

Before establishing the details for our model, we performed experiments on a classic AE which had the purpose to create a single latent space that included representations for both type of classes. The AE was trained on a single dataset including both classes α and β. The intent in this case was to propose a latent space that included both types of shapes. From the fairness point of view this approach seemed to be a good choice, since it would have produced an embedding of a set of shapes different for some aspects but in any case common to many others; it would have generated a global knowledge of both types of given shapes. Nevertheless this approach presented problems in the shape reconstruction: the decoding of the shape was very similar among different shapes of various sizes, compromising the shape identity; also the shape presented a limb of intermediate sizes that affected the realism of the body model.
Our model, respect to a classic AE, reconstructs shapes with accuracy and realism: the decoded shape is very much similar to the original input; moreover the model gives coeherent results for the mapping of shapes, since the reconstructed shapes even after this operation are pretty similar in appearance to the correspondent input shape. We arrive to this solution after two separate training on different typologies of models: data are treated equally, but at the same time also their differences are taken into consideration.

### Experiments

A single AE trained on both datasets has a worse behavior than our model. Our proposal works pretty fine for reconstructing the shapes and also for mapping. This suggests that it exists a good connection between the latent spaces and there is a good knowledge about the similarity between shapes belonging to different spaces.

<p align="center"><img width="745" alt="Schermata 2022-10-17 alle 12 56 49" src="https://user-images.githubusercontent.com/34343511/196160500-12507e79-ee18-469f-a182-7c4db7161c04.png"></p>

We can notice that the reconstruction of the proposed model is nearly identical to the input shape, proving that our method is working properly and it learned in an effective way. The classic AE, instead, does not give satisfactory results: as we can see it does not provide a mesh with the same identity template of the input and it does not present a likely appearance in the limb reconstruction.
We can also notice that our method is giving visually good results even for the mapping between latent spaces: the latent representation of the class α is mapped to the opposite latent space and its decoding shows a body shape very similar to the input shape β, and viceversa for the other class.

<p align="center"><img width="813" alt="Schermata 2022-10-17 alle 12 58 48" src="https://user-images.githubusercontent.com/34343511/196160801-e2950de0-0e9c-4b90-9818-ff7082b941db.png"></p>

Our model represents a better working solution respect to the classic AE, because producing body shapes with not likely results does not contribute at all to a fair appearance of body models. Our results in decoding and mapping let instead protect the uniqueness in the shapes since the identity of a body is preserved and it is guaranteed a closer representation of the body to reality.
Many applications could be positively influenced by these results. Programs in areas such as medicine, simulations, virtual humans that contains already body models would have the possibility to easily switch from one form of representation to another, thus managing to accelerate a process of inclusion in these fields.

We performed shape interpolation between models from a test dataset and we obtained their mapping to the opposite latent space. An interpolation is given by the following expression:

$$k*v<sub>α</sub> +(1−k)*v<sub>β</sub>$$, where

k ∈ [0, 1], v<sub>α</sub> is the latent representation of an input shape of class α and v<sub>β</sub> is the
latent representation of an input shape of class β.

<img width="651" alt="Schermata 2022-10-17 alle 13 02 34" src="https://user-images.githubusercontent.com/34343511/196161494-2b0c6ea1-28a2-4ce7-92be-7d5e4a496683.png">

We have observed from experiments that a classic AE does not permit to move properly from one shape to another in a space with large variations: the interpolated results are unrealistic and uninteresting.
In our model space it is possible instead to move in an Euclidean way: we have an expressive latent space that may find application in a generative model. From a fairness point of view, the expressiveness would facilitate the creation of new types of data and thus it would favor a more inclusive representation within the human body generation methods. For example, a new advance in this sense could lead to an even more widespread use of generative methods as integration tools into 3D modeling softwares.
