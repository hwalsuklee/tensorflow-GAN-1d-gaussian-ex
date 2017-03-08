# Generative Adversarial Network for approximating a 1D Gaussian distribution
Tensorflow implementation of Generative Adversarial Network for approximating a 1D Gaussian distribution.

## Results

Desirable result of GAN is to have decision boundary (db) of one-half and probability distribution function (pdf) of generated data similar to the original pdf.

Through the following images, you can see that all of results for various 1D Gaussian distributions are well shaped.  

On the other hand, code in two references does not give stable results when changing mean, sigma, or seed.  
Please check it out for yourself. 

<table align='center'>
<tr align='center'>
<td></td>
<td> mean = -1 </td>
<td> mean = +1 </td>
</tr>
<tr>
<td> stdev = 0.7 </td>
<td><img src = 'results/GAN_1d_gaussian_mu_-1_sigma_0.7.png' height = '300px'>
<td><img src = 'results/GAN_1d_gaussian_mu_1_sigma_0.7.png' height = '300px'>
</tr>
<tr>
<td> stdev = 1.0 </td>
<td><img src = 'results/GAN_1d_gaussian_mu_-1_sigma_1.png' height = '300px'>
<td><img src = 'results/GAN_1d_gaussian_mu_1_sigma_1.png' height = '300px'>
</tr>
<tr>
<td> stdev = 2.0 </td>
<td><img src = 'results/GAN_1d_gaussian_mu_-1_sigma_2.png' height = '300px'>
<td><img src = 'results/GAN_1d_gaussian_mu_1_sigma_2.png' height = '300px'>
</tr>
</table>

## References

The implementation is based on the projects:

[1] Project by Eric Jang : [BLOG](http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html), [CODE](https://github.com/ericjang/genadv_tutorial)  
[2] Project by John Glober : [BLOG](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/), [CODE](https://github.com/AYLIEN/gan-intro)

## Implementation Details
### Network architecture
Fully-connected neural network with 1 hidden layer.
<table align='center'>
<tr align='center'>
<td></td>
<td> Generator </td>
<td> Discriminator </td>
</tr>
<tr>
<td> Input layer </td>
<td> 1 node </td>
<td> 1 node </td>
</tr>
<tr>
<td> Hidden layer </td>
<td> 32 nodes + relu </td>
<td> 32 nodes + relu  </td>
</tr>
<tr>
<td> Output layer </td>
<td> 1 node </td>
<td> 1 node + sigmoid  </td>
</tr>
</table>

### Pre-training
Discriminator is pre-trained with pdf of the orginal data in both two references.  
In real situation, we don't know about the pdf of the original data. Actually, that is what we want know.  
*Estimated* pdf of the original data is used to pre-train discriminator in this implementation.

## Acknowledgements
This implementation has been tested with Tensorflow r0.12 on Windows 10 and Ubuntu 14.04.
