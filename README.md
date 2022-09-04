# Multi-level-variaitonal-auto-encoder-for-speech-disentanglement (Master's thesis)
The base paper of ML-VAE https://arxiv.org/abs/1705.08841 is extended to the speech dataset.


# Introduction:
Deep generative models achieved great success in unsupervised learning. The idea behind
generative models is to capture the inner probabilistic distribution that generates a class of
observation in latent space to generate the observation

# Motivation:
Data representation plays a vital role in machine learning algorithms. For that reason, most
of the effort goes into feature engineering. Disentangled representation to learn a latent embedding that
describes the data’s underlying structure that enhances the robustness, interpretability, and
generalization to unseen observation on downstream tasks. The learned representation of the
data can be used as an input to a classifier, as well, as it captures the posterior distribution of
the variation factor for the input data in the case of probabilistic models

# Deep probabilistic approach 
The thesis work aims to disentangle speech into two representations: Style and content
representation. Style represents the global characteristics of the speech signals, and the
content represents the local characteristics of the speech signals. The global characteristics
explain the speakers’ information. In contrast, the local characteristics explain the speech
signal’s linguistic content and prosody-related information.

## Variational Auto Encoder:
Some probabilistic assumptions are considered to learn the factor of
variation in the latent representation. First, the latent space z is sampled from the prior
distribution p(z), and second, the observation X is sampled from the conditional distribution
p(X|z). The approximate posterior distribution qϕ(z |X) is parameterised with neural networks. The
weight of the network is updated, and variational parameters ϕ are shared across the data
points, which is referred to as amortized inference that is in contrast to traditional inference
methods where the variational parameters are not shared across the data points

## System:
Given a set of acoustic features $\mathbf{X}= (X_{1},X_{2},...X_{k})$ with each observation $X_{k} \in \mathbb{R}^{F \times T}$, 
the ML-VAE technique is employed on a speech signals to extract the speaker invariant information from the content representation and speaker identity 
from the style representation. The model of speech disentanglement consists of style and content encoders that map the acoustic feature information 
into the latent embedding, and the decoder reconstructs the features from the combined style and content embedding, as shown in the figure 

![model](https://user-images.githubusercontent.com/57464195/188325378-df563bea-811a-4a37-ac41-eb684dc0ef00.png)

### Content representation:
The speaker invariant representation contains multiple information such as environmental
conditions, linguistic content, etc. The inference model is an encoder that maps the speaker
invariant information into the content representation using VAE. While VAE performs
amortised inference, that is, the acoustic features of the observation parameterise the posterior
distribution of the content representation, all the observations share a single set of parameters
ϕ_c that represents the speaker invariant information. The illustration of the content
representation is shown in the figure, where the reparameterisation trick is applied to the
output of the encoder network.

![content](https://user-images.githubusercontent.com/57464195/188325442-be9a2cbc-79d4-4af7-9de4-0ec3c3ae470c.png)

### Style representation:
To obtain the speaker embedding, https://arxiv.org/abs/1705.08841 approached a deep probabilistic method. There
are many possibilities for grouping operations for a speech signal because of multiple factors
of variations. In this work, the group operation is performed on speaker identity. In the case
of style representation, the group wise reparameterisation trick is performed for each distinct
group of observation as shown in the figure. The group-wise reparameterisation trick uses
a backpropagation algorithm to learn the speaker information as a generative factor
![style](https://user-images.githubusercontent.com/57464195/188325517-d3de4acb-4e18-4018-bdc7-f36b3469f5f8.png)



# Evaluation
## t-SNE:
t-SNE plot is used to analyse the style embedding
of speech data qualitatively. The result, as shown in the figure 6.5 is promising and shows
that the style embedding is clustered based on speaker utterances. In the plot, four different
speakers are randomly chosen from the ’test-clean-100’ dataset, and each data point represents
different utterances produced by the speakers. There are 280 utterance segments, and every
speaker has 70 utterance segments, and each segment is four seconds long. These results
deduce that the ML-VAE learns the speaker information from the style representation from
the group supervision as expected. However, the results should be further investigated to
understand the style embedding.

![ML_VAE](https://user-images.githubusercontent.com/57464195/188325693-55821aab-850e-4419-9c55-a33471733372.png)

## Evalutation on Error Rate:
The speaker’s invariant information is supposed to be inferred from the content embedding
so that speaker information is discarded, resulting in poor speaker classification and better
phoneme classification on the downstream tasks. The best checkpoint with minimum overall
loss chosen from the trained ML-VAE model is used to perform the downstream classifications.
Once the downstream training tasks are completed, the error rates. are
calculated. The result of ML-VAE model in  **Phoneme Error Rate** achieved **34.3%**
and the **Speaker Error Rate** of **56.83%**. The style embedding from the trained ML-VAE model is used to
calculate the **Equal Error Rate (EER)** on the ”test-clean” dataset and achieved **12.87%**.
