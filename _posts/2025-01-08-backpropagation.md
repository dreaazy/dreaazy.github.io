---
title:  "Backpropagation"
date:   2025-08-02
categories: 
  - blog
tags:
  - jekyll
  - images

---


The backpropagation is main algorithm used for training neural network via gradient descend.
To understand deeply the topics in this essay require a basic knowledge of calculus and linear algebra.

What does it mean training a network so that it could learn and generalize to solve our problem?
Training a network means adjusting his weight and biases according on how the model perform better for our current problem.
To achieve this result we aim to reduce the function of loss.
This is what gradient descend stands for.
Finding the local minimum of our function of loss to minimize the error.
So the gradient descend is just a routine of optimization of our model.

Backpropagation can make training a network with gradient descend as much as ten million times faster, relative to a naive implementation.

Backpropagation is mainly used for training neural networks by efficiently computing gradients needed for optimization. However, its underlying principle of reverse-mode automatic differentiation has found applications in other fields such as scientific computing, control theory, computer graphics, and engineering design. In these areas, backpropagation helps optimize complex systems by enabling fast and accurate calculation of sensitivities and parameter gradients.

In this essay I'm going to use sources like the excellent free book on neural network written by Micheal Nilsen, you will find all the resources at the end of the essay.

## Computational graph

Computational graphs are very useful in computer science, they allow you to track and compute complex mathematical expressions by breaking them down into a sequence of simple operations. Each node in a computational graph represents an operation (like addition, multiplication, or a more complex function), and each edge carries data, often in the form of tensors or scalars.

For example consider the expression:
\\(e = (a+b) * (b+1)\\)

We can add a intermediate step:
\\(c = a+b\\) and \\(d = b+1\\).
And get the result \\(e = c * d\\).
<img
src="/assets/Backpropagation-media/95d9ad30fd5932138b603f7a38133d73835120aa.svg"
class="wikilink" alt="./resources/cg_1.svg" style="width: 100%; max-width: 600px;" />

To evaluate the expression we can set the input variable to certain values, for example a=2, b=1.

<img
src="/assets/Backpropagation-media/0b4c50deb00024de16f8bf39938eef77600c9d48.svg"
class="wikilink" alt="./resources/cg_2.svg" style="width: 100%; max-width: 600px;" />

To know how \\(e\\) varies changes \\(a\\) or \\(b\\) we have to compute partial derivatives in the computational graph.
For evaluating this graph we need the sum rule and the product rule.

$$
\frac{\partial }{\partial a}(a+b) = \frac{\partial a}{\partial a} + \frac{\partial b}{\partial a} = 1
$$

$$
\frac{\partial }{\partial u}(uv) = \frac{\partial u}{\partial u}\cdot v+\frac{\partial v}{\partial u}\cdot u
$$


<img
src="/assets/Backpropagation-media/0d06683576b3c735410bd8341dc24b0da52ca10b.svg"
class="wikilink" alt="./resources/cg_3.svg" style="width: 100%; max-width: 600px;" />
The key idea to understand the efficiency of backpropagation and follow the next part of the essay is to grasp the difference between **forward mode differentiation** and **backward mode differentiation**.
**Forward-mode differentiation** starts at an input to the graph and moves towards the end. At every node, it sums all the paths feeding in. Each of those paths represents one way in which the input affects that node. By **adding them up**, we get the total way in which the node is affected by the input, it's derivative.
**Reverse-mode differentiation**, on the other hand, starts at an output of the graph and moves towards the beginning. At each node, **it merges all paths** which originated at that node.

Let's take for example we have a function of three variable:

$$f(x,y,z) = z$$

The total differential for a function of three variable is:

$$df = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial y}dy+\frac{\partial f}{\partial z}$$

If in turn x,y,z depends on a variable t, we can apply the **chain rule** and the total differential becomes:

$$\frac{\partial f}{\partial t} = \frac{\partial f}{\partial x}\cdot\frac{\partial x}{\partial t} + \frac{\partial f}{\partial y}\cdot\frac{\partial y}{\partial t}+\frac{\partial f}{\partial z}\cdot\frac{\partial z}{\partial t}$$

To understand how \\(a\\) changes according to nodes that are not directly connected to it we just applying the chain rule.
Let's analyze how \\(e\\) changes if \\(a\\) changes:

$$\frac{\partial e}{\partial a} = \frac{\partial e}{\partial c}\cdot \frac{\partial c}{\partial a} = 2\cdot 1=2$$

Let's analyze how \\(e\\) changes if \\(b\\) changes, in this case **we have to sum the contribution for multiple paths.**

$$\frac{\partial e}{\partial b} = \frac{\partial e}{\partial c}\cdot \frac{\partial c}{\partial b} + \frac{\partial e}{\partial d}\cdot \frac{\partial d}{\partial b}  = 2 \cdot 1 + 3 \cdot 1 = 5$$

Now let's make a concrete example using the previous expression.  
Understanding this will help you appreciate **why backpropagation is so efficient** when training neural networks.

To see how the output \\(e\\) changes when the input $b$ changes, we could use a **forward-mode differentiation** approach:  
We would start from the node where \\(b=1\\) and **follow all paths forward** through the graph, **summing the contributions** that lead to \\(e\\).

However, we would have to repeat this process separately for **each input** (like a, b, etc.) --- which can be slow when there are many inputs, as in neural networks.

<figure>

<img src="/assets/Backpropagation-media/8bfdf7f5612baeb02bbcea9a31d0a465c334cf53.svg" alt="Description" style="width: 100%; max-width: 600px;">
<figcaption>

Figure 1: Feedforward differentiation starting from the node b.
</figcaption>

</figure>

With this approach we would have to do the same thing for the \\(a\\) node, and then sum up the contributions of all the derivatives of all the input nodes.

Using a **backward differentiation** approach (reverse-mode), we start from the final output and propagate gradients **backward** through the graph.  
This allows us to compute the **partial derivatives of the output with respect to all input variables in a single backward pass**, thanks to **factoring and reusing common paths** via the chain rule.
<figure>

<img src="/assets/Backpropagation-media/1b13aa38787f99388b86100a809639240e9fd24a.svg" alt="Description" 
style="width: 100%; max-width: 600px;" >
<figcaption>

Figure 2: Backward differentiation starting from the node e we are able to get the derivatives of all the input nodes.
</figcaption>

</figure>

This example show how if we have a function with multiple inputs and few outputs a function like:
$$ \large
f:\mathbb{R}^n\to \mathbb{R}$$

Is **much more efficient** using backward differentiation to understand how the output changes of the input changes.
If for example we have 1,000,000 inputs, forward differentiation would require 1,000,000 passes through the network, while backpropagation differentiation would require just 2 passes through the network (**one for evaluating the expression at each node and one for calculating the derivatives**).
So **reverse-mode is \~500,000× faster** in practice for this case!

## Structure of a neural network and notation

This is a simple schema of a neural network, it's composed of one layer of input, one hidden layer and one output layer.
<img src="/assets/Backpropagation-media/simplenet.svg" class="wikilink" alt="simplenet.svg" style="width: 100%; max-width: 600px;" />

### notation

Simple photo that follows show the notation on neurons connected between each other.
Two neurons are connected by a weight, the value of a neuron is given by \\(a_{j}^l\\) where l represents the current layer while \\(j\\) stands for which neuron in the layer we are referring to.
Each neuron has a bias (\\(b_{t}^l\\)), the meaning of the notation if the same as the previous.

<figure style="text-align: center; margin: 1em auto;">
  <img src="/assets/Backpropagation-media/notation.svg" alt="notation.svg" style="width: 100%; max-width: 600px;" />
  <figcaption>notation.svg</figcaption>
</figure>

In a neural network each neuron is connected with all the other neurons in the next layer, and the strength of each connection is characterized by a **weight**. In addition to these weighted connections, each neuron also has a **bias** term, which acts like an adjustable constant input that allows the neuron to shift its activation function, helping the network better fit the data by enabling the output to be offset independently of the inputs. The neuron computes a weighted sum of its inputs, adds the bias, and then passes the result through an **activation function** --- a nonlinear transformation such as ReLU, sigmoid, or tanh. This nonlinearity is crucial, as it allows the network to learn and approximate complex, non-linear mappings between inputs and outputs. Without activation functions, no matter how many layers the network has, it would behave like a linear model.

This photos shows the steps to evaluate the level of activation of one neuron, \\(z_{j}^1\\) is an intermediate step, while \\(a_{j}^1\\) is the final level of activation.

<img src="/assets/Backpropagation-media/activation.svg" class="wikilink" alt="activation.svg" style="width: 100%; max-width: 600px; "  />

Each neuron of the \\(\ell\\) layer takes as input the activation level of all the other neurons in the \\(\ell-1\\)  layer of the network.
We can calculate the activation of the neuron \\(j\\) in the layer \\(\ell\\) as:

 
$$ \large
a_{j}^{\ell} = \sigma \left(\sum_{k} w_{jk}^{\ell} a_{k}^{\ell -1} + b_{j}^{\ell}\right) 

\tag{1}
$$

I will be much easier to manage if we would translate this in vector form, it's simple to show that the last expression is equal to:

$$ \large
a^{\ell}=\sigma\left( W^{\ell}a^{\ell-1} + b^{\ell} \right)
$$

And our intermediate step will be:

$$\large z^{\ell} = W^{\ell}a^{\ell-1} + b^{\ell}$$

## Cost function

When training a neural network, we begin by defining a **loss function** \\(l(h(x),y)\\) that quantifies how far the model's prediction \\(h(x)\\) is from the true output \\(y\\) for a **single data point**. This single-point loss is exactly what we use to compute gradients via **backpropagation**, updating the model's weights to reduce the error. However, in practice, we don't just have one example --- we are given a dataset of samples \\(\{(X_{1},Y_{2}),…,(X_{n},Yn)\}\\), which are drawn from some **unknown distribution** \\(P(x,y)\\). Our goal would ideally be to minimize the **expected loss** over this distribution:

$$ \large
\mathbb{E}_{(x,y) \sim P} \left[ \ell(h(x), y) \right] = \int \ell(h(x),y) dP(x,y)$$

But since $P(x,y)$ is unknown, we cannot compute this expectation directly. Instead, we approximate it using the **empirical average** over our dataset:

 
$$ \large
\hat{C} = \frac{1}{n} \sum_{x} \ell(h(X_{i}),Y_{i})

\tag{2}
$$

This empirical loss is what we minimize in practice. And since it's a sum of differentiable single-sample losses, we can still apply **backpropagation**, either over the entire sum (in batch gradient descent) or over small subsets (in stochastic or mini-batch gradient descent). Thus, the justification for using backpropagation over many samples stems from the fact that the empirical loss is an **estimator of the expected loss** --- and the best tool we have for reducing it with gradient-based methods.

This cost function takes as input **all the weights and biases** of the network and measures **how far off the network's predictions are** compared to the actual target outputs.

The choice on which cost function to choose varies according to the problem we are trying to solve.
For this essay I'm going to take in consideration the simplest cost function that is the **Mean squared error**:

$$\large
C(w,b)=\frac{1}{n}\sum_{x} ||y(x)− a^L(x)||^2

\tag{3}
$$

Where \\(y(x)\\) is the desired output, \\(a\\) is the output we can, it obviously depends on the weights and biases of the network, while \\(x\\) stands for the training samples.

Indeed the cost for a simple training example is:
 
$$ \large
C_{x} = \frac{1}{2} ||y-a^L||^2

\tag{4}
$$

What backpropagation allow us to do is to calculate \\(\large \frac{\partial C_{X}}{\partial w}\\) and \\(\large \frac{\partial C_{x}}{\partial b}\\) and then finding \\(\large \frac{\partial C}{\partial w}\\) and \\(\large \frac{\partial C}{\partial b}\\) averaging over the training samples.

I said earlier that the cost function take as input all the weight and biases of the network, I need to precise better that.
The cost function take as input **the last layer of the network**, or also called the **"prediction**" of the network, the prediction depends on all the weights and biases of the network.
The cost function measures **how close or far is the prediction compared to the actual result**, backpropagation through gradient descend, is an algorithm to make this prediction closes as possible.
<img src="/assets/Backpropagation-media/cf_1.svg" class="wikilink" alt="cf_1.svg" style="width: 100%; max-width: 600px;" />

## Backpropagation derivation

Once understood the problem we are trying to solve, let's try to address the problem in a more mathematical way.
The heart of backpropagation is understand how the cost function varies, if the matrix of weights of the first layer varies.
Or in a mathematical way:

$$ 
\large
\frac{\partial C \left( a^L\left( a^{L-1}\right) \dots\left( a^l\left(w^l \right)\right)\right)}{\partial w^l}

$$

- \\(C\\) is a **scalar loss function** (like MSE or cross-entropy)

- \\(w^l \in \mathbb{R}^{n\times m}\\) is the **weight matrix** for layer \\(l\\)

- The output \\(C\\) depends on the activations, which depend on the weights from **all previous layers**, including \\(w^l\\)
  ##### What we want to do is to apply the chain rule layer by layer through the computational graph.

We have a scalar function \\(C\\) that depends on a vector \\(a^L\\) **that depends on a vector that depends on a vector**.... so on until we have a vector that depends on a matrix \\(w^l\\).

Since the cost \\(C\\) is a scalar, and the weights \\(w^l\\) are a matrix, we're computing the **derivative of a scalar function with respect to a matrix**.

But this scalar depends on a composition of many functions --- vectors and matrices interacting across layers. So to compute this derivative, we'll need to use:

- Derivative of a scalar with respect to vector (gradient)

- Derivative of a scalar with respect to matrix

- Derivative of a vector with respect to a vector (Jacobian)

- Chain rule for matrix calculus

### Derivative of a scalar function with respect to a vector

Let \\(C:\mathbb{R}^n\to \mathbb{R}\\), with \\(v=(x_1,x_{2},\dots,x_{n})\\).

$$\frac{\partial C}{\partial v} = \nabla C =\begin{pmatrix}
\frac{\partial C}{\partial x_{1}} \\
\frac{\partial C}{\partial x_{2}} \\
\vdots \\
\frac{\partial C}{\partial x_{n}}
\end{pmatrix}$$

Example: suppose \\(v=a^L\\) , \\(C=C(a^L)\\), then:

$$\frac{\partial C}{\partial a^L} = \begin{pmatrix}
\frac{\partial C}{\partial a_{1}^L} \\
\frac{\partial C}{\partial a_{2}^L} \\
\vdots \\
\frac{\partial C}{\partial a_{n}^L}
\end{pmatrix}$$

So: 
$$
\large \frac{\partial C}{\partial a^L} = \nabla_{a_{L}} C\in \mathbb{R}^{n\times1}
$$

The derivative of a scalar function with respect to a vector is just the gradient of that function, **this tells us how much the cost changes when each component of the vector \\(a^L\\) changes**.

### Vector function of a vector variable

Let \\(f:\mathbb{R}^n\to \mathbb{R}^m\\),\\(x=(x_{1},x_{2}\dots,x_{n})\\), then:

$$
f(x) = \begin{pmatrix}
f_{1}(x) \\
f_{2}(x) \\
\vdots \\
f_{m}(x)


\end{pmatrix}
$$

Each \\(f_{i}\\) is a scalar function of the same vector \\(x \in \mathbb{R}^n\\).
Then the derivative is the **Jacobian matrix**:

$$
\frac{\partial f}{\partial x} = J=
\begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}

$$

This generalizes the gradient it tells **how each component of the output vector changes according to every input component**.
Each row is the gradient of one component function \\(f_{i}(x)\\). So the Jacobian is a stack of gradients.

### Derivative of a scalar function with respect to a matrix

$$\large 
f:\mathbb{R}^{n \times m}\to \mathbb{R}$$

So \\(f\\) is a scalar function of a matrix.
The derivative is defined as:

$$

\frac{\partial f}{\partial W}
= \begin{pmatrix}
\frac{\partial f}{\partial w_{11}} & \cdots & \frac{\partial f}{\partial w_{1m}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial w_{n1}} & \cdots & \frac{\partial f}{\partial w_{nm}} \\
\end{pmatrix}
\in \mathbb{R}^{n \times m}

$$

We differentiate the scalar function with respect to each component of the matrix --- and the result is again a matrix.

Turning back to our final objective, that is to compute \\(\large \frac{\partial C}{\partial w^l}\\).
This falls under the case of a **scalar function of a matrix**, and produces a matrix of partial derivatives.
However, to compute it via the chain rule, we often encounter intermediate expressions such as the derivative of a **vector with respect to a matrix**.
### This brings us to study vector-valued functions of matrix variables:

$$
\large f:\mathbb{R}^{n\times m}\to \mathbb{R}^k
$$

This function takes as input a matrix and gives out a vector, so the function would look something like:

$$f(W)=\begin{pmatrix}
f_{1}(W) \\
f_{2}(W) \\
\vdots \\
f_{k}(W)
\end{pmatrix}$$

What is \\(\huge \frac{\partial C}{\partial W}\\)?

Each component function $f_{i}(W)$ is a scalar function of the matrix $W$. So we can take its derivative just like before:
$$ \large 
\frac{\partial f_{i}}{\partial W}\in R^{n\times m}$$

Now since we have $k$ of them we just stuck these matrixes the one after the other.
That gives us a cube-like object with shape:
$$ \large 
\frac{\partial f}{\partial W} \in \mathbb{R}^{k \times n \times m}$$

**This is a rank-3 tensor.**
This tensor tells us how each output component $f_{i}$ changes with each element $w_{jk}$​ of the input matrix.

In our case the function $f$ corresponds to $a^l$, and the components of $a^l$ are the individual activations of $a_{i}^l$, which correspond to the different $f_{i}$ .

This is an example of matrixes for the first 3 layers of the network: layer 1, layer 2 and layer 3.

<figure>
<img
src="/assets/Backpropagation-media/64b73722444b2241bacbd82e7e9349dce2b896f0.svg"
class="wikilink" alt="./resources/tensor.svg" />
<figcaption aria-hidden="true">./resources/tensor.svg</figcaption>
</figure>

*Figure 2: In our case, the function $f$ corresponds to $a^l$, and the components of $a^l$ are the individual activations $a^l_i$, which correspond to the different $f_i$.*

We understood how to differentiate with respect to vectors an matrixes, now we need to find the namesake of the chainrule but for vectors.
So we need to evaluate something like:

$$\large \frac{\partial C \left(y(x) \right)}{\partial x}$$

Where:
- $C$ is a scalar function
- $\large y:\mathbb{R}^n\to \mathbb{R}^m$ is a vector valued function
- and $\large x \in \mathbb{R}^n$
The composition $C(y(x))$ is still a scalar function --- it maps $\large x∈\mathbb{R}^n$ to a real number via $\large y$.

As we saw before, we are taking the derivative of a scalar function with respect to a vector, and that is just the gradient.

$$\large \frac{\partial C \left(y(x) \right)}{\partial x} =\nabla_{c} C=\begin{pmatrix}
\frac{\partial C}{\partial x_{1}} \\
\frac{\partial C}{\partial x_{2}} \\
\vdots \\
\frac{\partial C}{\partial x_{n}}
\end{pmatrix}$$

But now, we can go a step further and express how $\large C$ depends on $\large x$ **through** $\large y(x)$.

$$\large C=C(y_{1}(x),y_{2}(x),\dots,y_{m}(x))$$

So what we are dealing with is a composition of functions, so we use the chain rule for multivariable calculus we have seen when we were dealing with the computational graphs:

$$\large \frac{\partial C}{\partial x_{j}}=\sum_{i=1}^m \frac{\partial C}{\partial y_{i}} \cdot\frac{\partial y_{i}}{\partial x_{j}}$$

This tells us: to compute how $C$ changes as we vary $\large x_{j}$​, we look at how each $\large y_{i}$​ changes with $x_{j}$​, weighted by how sensitive $\large C$ is to $\large y_{i}$​.

$\large x$ is a vector of $\large n$ components, so if we stack all these pieces of chain rule together we get:

$$\large \nabla_{x} C=\begin{pmatrix}
\sum_{i=1}^m \frac{\partial C}{\partial y_{i}} \cdot\frac{\partial y_{i}}{\partial x_{1}} \\
\sum_{i=1}^m \frac{\partial C}{\partial y_{i}} \cdot\frac{\partial y_{i}}{\partial x_{2}} \\
\vdots \\
\sum_{i=1}^m \frac{\partial C}{\partial y_{i}} \cdot\frac{\partial y_{i}}{\partial x_{n}}
\end{pmatrix}$$

#### Matrix Form: Vector Chain Rule

Rather than writing all those sums explicitly, we can express this **compactly using matrix multiplication**.

Let's define:

- The **Jacobian matrix** $\large \frac{\partial y}{\partial x}\in \mathbb{R}^{m \times n}$ , whose $(i,j)-th$ entry is $\large \frac{\partial y_{i}}{\partial x_{j}}$
- The **gradient of $\large C$ with respect to $\large y$**, $\large \nabla_{y} C =\mathbb{R}^m$, whose $i$ components is $\large \frac{\partial C}{\partial y_{i}}$
Then the full gradient is:

$$
\large \nabla_{c} C = \left(\frac{\partial y}{\partial x}\right)^{\top} \cdot \nabla_{y} C

\tag{5}
$$

That is the Chain rule for vector-valued function.

## Chain Rule with Matrix-Valued Variables

We now want to compute the derivative of a scalar function composed with a function of a matrix:

$$\large \frac{\partial C(Y(W))}{\partial W}$$

------------------------------------------------------------------------

Differentiate a scalar with respect to a matrix:
$$ \large
\frac{\partial C}{\partial W} =
\begin{pmatrix}
\frac{\partial C}{\partial w_{11}} & \frac{\partial C}{\partial w_{12}} & \cdots & \frac{\partial C}{\partial w_{1n}} \\
\frac{\partial C}{\partial w_{21}} & \frac{\partial C}{\partial w_{22}} & \cdots & \frac{\partial C}{\partial w_{2n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial C}{\partial w_{m1}} & \frac{\partial C}{\partial w_{m2}} & \cdots & \frac{\partial C}{\partial w_{mn}} \\
\end{pmatrix}$$

Our case the scalar C depends on a vector $\large y$, and that vector depends on the matrix $\large W$

- $\large C: \mathbb{R}^{m} \to \mathbb{R}$ is a scalar function  
- $\large  y: \mathbb{R}^{k \times n} \to \mathbb{R}^{m}$ is a vector-valued function  
- $\large  W \in \mathbb{R}^{k \times n}$ is a matrix input

------------------------------------------------------------------------

In our case the scalar function $\large C$ Depends on a vector $\large y$ and the vector depends on a matrix $\large W$.

This leads us to the composition

$$\large  C\left(y\left(W\right)\right)$$

With $\large y$ with the form of:

$$\large y=\begin{pmatrix}
y_{1}(W) \\
y_{2}(W) \\
\vdots \\
y_{k}(W)
\end{pmatrix}$$

Now we have to use the matrix definition looking at the components of the matrix $\large W$:

$$\large \left[ \frac{\partial C\left(y(W)\right)}{\partial W}\right]_{jk} = \frac{\partial C\left(y(W)\right)}{\partial w_{jk}}$$

How $\large C$ depends on $\large w_{ij}$? It depends on $\large y$ that depends on $\large W$. So let's apply the chain rule but for scalars.

$$\large \frac{\partial C(y(W))}{\partial w_{jk}} = \sum_{i=1}^m \frac{\partial C}{\partial y_i} \cdot \frac{\partial y_i}{\partial w_{jk}}$$

Now we can put it back into the full matrix:

$$\large \frac{\partial C(y(W))}{\partial W} =
\begin{pmatrix}
\sum_i \frac{\partial C}{\partial y_i} \cdot \frac{\partial y_i}{\partial w_{11}} & \cdots & \sum_i \frac{\partial C}{\partial y_i} \cdot \frac{\partial y_i}{\partial w_{1n}} \\
\sum_i \frac{\partial C}{\partial y_i} \cdot \frac{\partial y_i}{\partial w_{21}} & \cdots & \sum_i \frac{\partial C}{\partial y_i} \cdot \frac{\partial y_i}{\partial w_{2n}} \\
\vdots & \ddots & \vdots \\
\sum_i \frac{\partial C}{\partial y_i} \cdot \frac{\partial y_i}{\partial w_{m1}} & \cdots & \sum_i \frac{\partial C}{\partial y_i} \cdot \frac{\partial y_i}{\partial w_{mn}} \\
\end{pmatrix}$$

Each entry in this matrix is a sum of scalar-by-scalar products. We just applied the scalar chain rule to each component and it's all there in one matrix.

#### Let's simplify the notation.

- $\large \frac{\partial y}{\partial W}$ is a tensor because it's a vector with respect to a matrix.
- $\large \frac{\partial C}{\partial y}$ it's a vector
  So we can write what we wrote as:

  $$\large \frac{\partial C(y(W))}{\partial W} = \sum_{i=1}^k \frac{\partial C}{\partial y_{i}} \cdot \frac{\partial y_{i}}{\partial W}$$

  #### Using Tensor contraction.
  To simplify again our notation to scale it better on our neural network we need to use a tensor contraction.
  A **tensor contraction** is an operation on a tensor that arises from the canonical pairing of a vector space and its dual.
  When we "contract" over an axis, we're summing along that dimension reducing its size.

So the final expression is:

$$
\large \frac{\partial C}{\partial W} = \sum_{i=1}^k \frac{\partial C}{\partial y_{i}} \cdot \frac{\partial y_{i}}{\partial W} \to\large \frac{\partial C}{\partial W} =  \frac{\partial y}{\partial W} \overset{i}{\cdot} \frac{C}{\partial y}

\tag{6}
$$

That's how we simplify the chain rule when the variable is a matrix and the intermediate function is vector-valued.

# Backpropagation algorithm

Our main goal is to compute:


$$
\large \frac{\partial C \left( a^L\left( a^{L-1}\right) \dots\left( a^l\left(w^l \right)\right)\right)}{\partial w^l}

\tag{7}
$$

We are taking the derivative with respect to a matrix, but this scalar depends on a vector, that depends on a vector that depends on a vector and so on until it depends on the matrix $\large w^l$.

We start from the outermost layer, the layer linked with the matrix $\large w^l$ and we compute the tensor contraction:


$$
\large \frac{\partial C}{\partial W} =  \frac{\partial a^l}{\partial w^l} \overset{i}{\cdot} \frac{C}{\partial a^l}

\tag{8}
$$

But the vector $\large a^l$ depends on the vector $\large a^{l+1}$ and so on until the last layer $a^L$, to compute how each layer change each other we use the chain rule for vectors we found earlier:

$$
\large \frac{\partial C}{\partial a^l} = \left(\frac{\partial a^{l+1}}{\partial a^l}\right)^{\top} \cdot \frac{\partial C}{\partial a^{l+1}} 

\tag{9}
$$

And this is true for every layer:


$$
\large \frac{\partial C}{\partial a^{l+1}} = \left(\frac{\partial a^{l+2}}{\partial a^{l+1}}\right)^{\top} \cdot \frac{\partial C}{\partial a^{l+2}} 
$$

And that is the recursive hear of backpropagation until the last layer $\large a^L$.

### Error backpropagation signal

Some sources define the error signal as:

$$ \large
\delta^l := \frac{\partial C}{\partial a^l}$$

This can be used recursively:

$$\large \frac{\partial C}{\partial a^l} = \left(\frac{\partial a^{l+1}}{\partial a^l}\right)^{\top} \cdot \frac{\partial C}{\partial a^{l+1}}  \to \delta^l= \left(\frac{\partial a^{l+1}}{\partial a^l}\right)^{\top} \cdot \delta^{l+1}$$

However, in standard backpropagation, the error is usually defined as:
$$ \large
\delta^l := \frac{\partial C}{\partial z^l}$$

The reason for the last definition is the most straight forward:

If we are for example in the neuron $\large j^{th}$ in the layer one. If we make a little change to the weighed input of the neuron $\large z_{j}^1$ adding the quantity $\large \Delta z_{j}^1$, then the activation input instead of outputting $\large \sigma(z_{j}^1)$ it will output $\large \sigma(z_{j}^1 + \Delta z_{j}^1 )$.
This change propagates through all the network and finally change the cost function of a quantity:

$$\frac{\partial C}{\partial z_{j}^1}\cdot\Delta z_{j}^1$$

If we want to try to minimize the cost function, we can try to chose the quantity $\large \Delta z_{j}^1$ so that makes the cost smaller.
If the quantity $\large \frac{\partial C}{\partial z_{j}^1}$ is close to zero then there is little to change, indeed if the quantity $\large \frac{\partial C}{\partial z_{j}^1}$ is big we can chose the quantity $\large \Delta z_{j}^1$ of the opposite sign to try to minimize the cost.

In this heuristic sense, this quantity $\large \frac{\partial C}{\partial z_{j}^1}$ is the measure of the error of the neuron $j^{th}$ in the layer 1.

$$\large \delta_{j}^1=\frac{\partial C}{\partial z_{j}^1}$$

## Why both definitions can coexist

There is no contradiction between defining the error as $\large \frac{\partial C}{\partial a^l}$ or $\large \frac{\partial C}{\partial z^l}$, because they are directly related through the chain rule.
Since $\large a^l = \sigma(z^l)$, we apply the chain rule:

$$ \large
\frac{\partial C}{\partial z^l} = \frac{\partial C}{\partial a^l} \cdot \frac{\partial a^l}{\partial z^l}$$

Because $\large \sigma$ is applied elementwise, this becomes an elementwise (Hadamard) product:

$$ \large
\frac{\partial C}{\partial z^l} = \frac{\partial C}{\partial a^l} \odot \sigma'(z^l)$$

That is,

$$ \large
\delta^l = \tilde{\delta}^l \odot \sigma'(z^l)$$

where

$$\large
\tilde{\delta}^l := \frac{\partial C}{\partial a^l}$$

So depending on the context, one may define and propagate $\large \tilde{\delta}^l$ (via the recursive expression), and then obtain the standard error $\large \delta^l$ by applying the derivative of the activation. They are just two steps in the same computation, not conflicting definitions.

This is the key equation for backpropagation that links the error of one layer to the layer of the next one.


$$
\Large \delta^l= \left(\frac{\partial a^{l+1}}{\partial a^l}\right)^{\top} \cdot \delta^{l+1}

\tag{11}
$$

Since we are interested in finding:

$$
\large \frac{\partial C}{\partial w^l} \text{ for all layers  }l
$$

Once we know the vector $\large \delta^l$ we can compute the weight gradient using a tensor contraction (using equation 8).

$$
\large 
\frac{\partial C}{\partial w^l} =  \frac{\partial a^l}{\partial w^l} \overset{i}{\cdot} \delta^l 
\tag{12}
$$

Before jumping to the algorithm itself and the implementation though code, I think is due to spend at least a little time explaining how the gradient descend work. How exactly finding all these derivative and stuff how actually is going to make our neural network learn something?

## Gradient descend

**The gradient is a vector that points in the direction of the greatest rate of increase of a scalar field.**

The gradient is calculable only of a scalar-valued differentiable function.

So the gradient is defined only for functions that take a $n-dimention$ vector and gives as output a scalar.

$$\large
f:\mathbb{R}^n\to\mathbb{R}$$

So you can use gradients only with scalar fields.

We can define the gradient of a function $f(r)$ as:

$$\large df = \nabla f dr$$

What is formula is saying is how much the function will change according to the displacement $dr$.
If we are in an 3D space then $dr = (dx,dy,dz)$.

Since $\nabla f$ is a vector that point to the steepest increase of the function.

If the displacement is towards the vector $\nabla f$ then df is going to be bigger, while if the displacement is in the opposite direction the change of the function is going to be little.
If the displacement is orthogonal to the $\nabla f$ vector, there is going to be no change in the the function, this is for the definition of dot product.

The gradient for a scalar function at a point $p$ is defined as:

$$
\large {\displaystyle \nabla f(p)={\begin{bmatrix}{\frac {\partial f}{\partial x_{1}}}(p)\\\vdots \\{\frac {\partial f}{\partial x_{n}}}(p)\end{bmatrix}}}

\tag{13}
$$

This definition works only if the function is differentiable in p, otherwise obviously we cannot do the partial derivates.

By the definition of gradient follows that if we want to minimize our scalar function, that quantifies how far the model's prediction $h(x)$ is from the true output $y$ for a **single data point**, then we have to move in the opposite direction of the gradient from that point $\large p$.

The update rule for the gradient descend is:

$$\large a_{n+1} = a_{n} - \eta\nabla f(a_{n})$$

$\large \eta$ is the learning rate, is represent how long is the step you are going to take towards the function local minimum, $\Large a_{n}$ is the point you are evaluating the gradient, from that point you move to the direction of the fastest decrease of the function.

How can we apply gradient descent to learn in a neural network? The idea is to use gradient descent to find the weights $\large w_{k}$ and biases $\large b_{l}$ which minimize the cost in Equation. To see how this works, let's restate the gradient descent update rule, with the weights and biases replacing the variables $\large v_{j}$. In other words, our "position" now has components $\large w_{k}$ and $\large b_{l}$, and the gradient vector $\nabla C$ has corresponding components $\large \frac{\partial C}{\partial w_{k}}$ and $\large \frac{\partial C}{\partial b}$. Writing out the gradient descent update rule in terms of components, we have the update rule for the bias:
$$ \large
 b_{1} \to b_{1}^{′} = b_{1} - \eta \frac{\partial C}{\partial b_{1}}
 \tag{14}$$

And the update rule for the weights of every single layer:

$$\large w_{k} \to w_{k}^{′} = w_{k}-\eta \frac{\partial C}{\partial w_{k}}
\tag{15}$$

By applying this rule we can "roll down the hill" and hopefully find a minimum of the cost function.
Of course we cannot be sure that the minimum we'll find will be the global minimum of the function, but just a local minimum.
The lesson is that we cannot be sure if there is a better configuration of weights and biases that make our model predict better.

A better implementation of gradient descend is stochastic gradient descend, as show in the chapter of the cost function, we have defined a cost function with multiple sample.
The idea of stochastic gradient descend is to estimate the gradient $\nabla C$ by computing  $\nabla C_{x}$ for a small sample of randomly chosen training inputs. By averaging over this small sample it turns out that we can quickly get a good estimate of the true gradient $\nabla C$, and this helps speed up gradient descent, and thus learning.

We are calling those random input training samples $\large(X_{1},X_{2},\dots,X_{m} )$ we will call them **mini-batch**.
The $m$ must be large enough to have a roughly similar result for the gradient.
So that $\nabla C_{X_{j}}$ and $\nabla C_{X}$ will be almost the same, but the gradient will be much easier to compute.

If this is true so we have:

$$
\Large 

\frac{\sum_{j=1}^m \nabla C_{X_{j}}}{m} \approx \frac{\sum_{X} \nabla C_{X}}{n} =\nabla_{} C
$$

Where the second sum is over **all** the set of training data, so we get the approximation used for stochastic gradient descend:

$$\Large
\frac{\sum_{j=1}^m \nabla C_{X_{j}}}{m} \approx \nabla C$$

Taking this into account the update rule for stochastic descend becomes:

$$
\large
w_k' = w_k - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial w_k}

\tag{16}
$$

$$

\large

b_l' = b_l - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial b_l}
\tag{17}
$$


Let's make a concrete example on how a neural network is trainer using stochastic gradient descend:

### Start of Training:

- Your model begins with **random weights and biases**.
- It doesn't know anything yet --- its predictions are probably **garbage**.
  ### Epoch 1:
- **Shuffle** the full training set (e.g. 100,000 examples).
- Break it into **mini-batches** (e.g. 100,000 examples → 1,562 mini-batches of 64 examples).
  #### For each mini-batch:

1.  **Backpropagation step**: compute the **gradients** $\large \to$ how much each weight and bias should change to reduce the error.
2.  **Update** the weights and biases using the SGD formulas:
    ### Epoch 2:

- **Shuffle** the data again (important to avoid fixed patterns).
- Repeat the mini-batch training process.
- Now, weights and biases are **slightly better**.
- The model should start making **better predictions**.
  ### Repeat for Many Epochs (e.g., 10, 50, 100):
- With each epoch, the model sees **all training examples again** (in new order).
- Parameters are **updated gradually**.
- The **cost function** ( C ) (overall error) should **decrease** over time.
- The model becomes better at:
  - **Generalizing**
  - **Predicting accurately**
  - **Learning** meaningful patterns

## Algorithm and implementation

The algorithm for backpropagation is composed of three phases:
1) Input a sample of training example
2) For each training sample:
1) Feedforward phase
2) Compute output error
3) Backpropagate the error
3) Gradient descend

### 2) phase

We need to find the quantities $\large \frac{\partial C}{\partial w^l}$ and $\large \frac{\partial C}{\partial b^l}$ for all layers $l$.

So for each layer $l$ starting from the last one and moving backwards we have to compute first:

$$\large \delta^L :=\frac{\partial C}{\partial a^L}$$

And then propagate the error backwards:


$$
\large
\delta^l = \left( \frac{\partial a^{l+1}}{\partial a^l} \right)^\top \cdot \delta^{l+1}, \quad \text{for } l = L, L - 1, L - 2, \dots, 1

\tag{11}
$$

And for each layer compute what we are looking for as:

$$
\large \frac{\partial C}{\partial w^l} =  \frac{\partial a^l}{\partial w^l} \overset{i}{\cdot} \delta^l 
\tag{12}
$$

And:

$$ 
\large 
\frac{\partial C}{\partial b^l} = \delta^l
$$

### 3) phase
Once computed these derivatives we have to use gradient descend as:

$$

\large
w_k' = w_k - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial w_k}

\tag{16}
$$

$$

\large
b_l' = b_l - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial b_l}
\tag{17}
$$



This is the big summary of the backpropagation algorithm of his most general form, if we want to implement this algorithm we need to decide which activation function and cost function we are going to use.
We are going to use the sigmoid function:

$$
\large \sigma(x)=\frac{1}{1+e^{-x}},\quad \text{with} \quad \sigma^{′}(x)=\sigma(x)(1-\sigma(x))
$$

Meanwhile as cost function the squared error:

$$ \large
C = \frac{1}{2} \| a^L - y \|^2,
$$

I'm going to use the code of Micheal Nielsen you can find on his book, however you can find the complete code on my github page used with training samples and MINST dataset.

Before write the code we need to understand how the equations seen on phase 2 of the algorithm changes if we use the sigmoid function and the squared error as cost function.

We start computing $\large \delta^L$ the error for the last layer we defined as:

$$\large \delta^L :=\frac{\partial C}{\partial a^L}$$

Now if we take the derivative of the squared error with respect to $\large a^L$ we obtain (check for yourself):

$$\large 
\frac{\partial C}{\partial a^L}=a^L-y$$

the error term at the output layer is defined as
$$ \large
\delta^L = \frac{\partial C}{\partial z^L}.$$

There is no contradiction in defining the output layer error as

$$ \large
\delta^L := \frac{\partial C}{\partial a^L},$$

because this is a valid partial derivative that describes how the cost changes with respect to the activation at layer $\large L$. However, during backpropagation, we ultimately need the derivative with respect to $\large z^L$, since $\large z^L$depends directly on the weights. By applying the chain rule, we refine our expression:

$$ \large
\delta^L = \frac{\partial C}{\partial z^L} = \frac{\partial C}{\partial a^L} \odot \frac{\partial a^L}{\partial z^L}.$$

Since we now that $\large \frac{\partial C}{\partial a^L} = a^L - y$, and we know the form of the derivative of the activation function:

$$ \large
\frac{\partial a^L}{\partial z^L} = \sigma'(z^L) = a^L \odot (1 - a^L),$$

this leads us to the final expression for the error on the last layer.

$$ \large
\delta^L = (a^L - y) \odot a^L \odot (1 - a^L).$$

Now we need to propagate the error back recursively using the rule:

$$
\large
\delta^l = \left( \frac{\partial a^{l+1}}{\partial a^l} \right)^\top \cdot \delta^{l+1}, \quad \text{for } l = L, L - 1, L - 2, \dots, 1

\tag{11}
$$

**What form does this take with the sigmoid activation function?**

$$\large
\begin{align*}
a^{l+1} &= \sigma(z^{l+1}) \\
        &= \sigma(W^{l+1} a^l + b^{l+1}) \\
\\
\frac{\partial a^{l+1}}{\partial a^l} 
        &= \frac{\partial \sigma(z^{l+1})}{\partial a^l} \\
        &= \frac{\partial \sigma(z^{l+1})}{\partial z^{l+1}} \cdot \frac{\partial z^{l+1}}{\partial a^l}
\end{align*}$$

In the last term I've just applied the chain rule.
We know:

$$ 
\large
z^{l+1} = W^{l+1} a^l + b^{l+1}
$$

Since this is a linear function of $\large a^l$, taking it's derivative the second term of the last expression becomes:

$$ 
\large
\frac{\partial z^{l+1}}{\partial a^l} = W^{l+1}
$$

Activations are applied element by element:

$$ 
\large
a^{l+1} = \sigma(z^{l+1})
$$

So the derivative is a **diagonal matrix** with the derivative of $\large \sigma$ applied to each entry:

$$ \large
\frac{\partial \sigma(z^{l+1})}{\partial z^{l+1}} = \mathrm{diag}(\sigma'(z^{l+1}))$$

This matrix has $\large \sigma'(z_i)$ on the diagonal and zeros elsewhere.
So adding all up together, the transpose becomes:

$$ \large
\left( \frac{\partial a^{l+1}}{\partial a^l} \right)^\top = \left(\mathrm{diag}(\sigma'(z^{l+1})) \cdot (W^{l+1})^\top \right)$$

We have not finished yet, we have to multiply this by the error of $\large \delta^{l+1}=\frac{\partial C}{\partial a^{l+1}}$.
But multiplying a diagonal matrix by a vector is just an elementwise product!

$$ 
\large
\sigma'(z^{l+1}) \odot \left( (W^{l+1})^\top \cdot \frac{\partial C}{\partial a^{l+1}} \right)
$$

This gives us the clean recursive formula we love in backpropagation. So:

$$ \large
\delta^l = (W^{l+1})^\top \cdot \delta^{l+1} \odot \sigma'(z^l)$$

That we can write also (remembering for form of the first derivative of the activation function):

$$ \large
\delta^l = (W^{l+1})^\top \cdot \delta^{l+1} \odot a^l \odot (1 - a^l)$$

### Gradient of the weights

Once we have the error vector $\large \delta^l$, we can compute the gradient of the cost with respect to the weights:

$$ \large
\frac{\partial C}{\partial w^l} = \frac{\partial a^l}{\partial w^l} \overset{i}{\cdot} \delta^l$$

Let's examine a single element of this expression:

$$ \large
\frac{\partial a^l_i}{\partial w^l_{jk}} = \frac{\partial \sigma(z^l_i)}{\partial w^l_{jk}} = \sigma'(z^l_i) \cdot \frac{\partial z^l_i}{\partial w^l_{jk}}$$

Since:

$$ \large
z^l_i = \sum_k w^l_{ik} a^{l-1}_k + b^l_i
$$

Then:

$$ 
\large
\frac{\partial z^l_i}{\partial w^l_{jk}} =
\begin{cases}
a^{l-1}_k & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}
$$

So:

$$
 \large
\frac{\partial a^l_j}{\partial w^l_{jk}} = \sigma'(z^l_j) \cdot a^{l-1}_k
$$

Multiplying by the error:

$$ 
\large
\frac{\partial C}{\partial w^l_{jk}} = \delta^l_j \cdot a^{l-1}_k
$$

Switching to matrix form the full gradient is given by:

$$ \large
\frac{\partial C}{\partial w^l} = \delta^l (a^{l-1})^\top$$

This is the outer product between the error vector at layer $\large l$ and the activations from the previous layer.

### What about the bias $\large b^l$?

The bias appears in the pre-activation like this:

$$ 
\large
z^l = W^l a^{l-1} + b^l
$$

So when we differentiate with respect to $\large b^l$, it's even simpler.

We know:

$$ 
\large
a^l = \sigma(z^l) = \sigma(W^l a^{l-1} + b^l)

$$

This means the bias $\large b^l$ directly and linearly affects the pre-activation $\large z^l$.  
That is:

$$ 
\large
\frac{\partial z^l}{\partial b^l} = I
$$

(the identity matrix).

So, using the chain rule:

$$
\large
\frac{\partial C}{\partial b^l} =
\frac{\partial z^l}{\partial b^l} \cdot \frac{\partial C}{\partial z^l}
= I \cdot \delta^l = \delta^l
$$

The gradient of the cost with respect to the bias is simply the error vector:

$$ 
\large
\frac{\partial C}{\partial b^l} = \delta^l
$$

## Conclusion
## Key Takeaway: From General Backpropagation to Implementation

Starting from the most general backpropagation equations, we derived the specific forms used in practice, depending on the activation and cost functions chosen. Here is a concise summary of how the expressions evolve:

------------------------------------------------------------------------

### Output Layer Error

$$\large
\delta^L := \frac{\partial C}{\partial a^L} \quad \to \quad \delta^L = (a^L - y) \odot a^L \odot (1 - a^L)$$

------------------------------------------------------------------------

### Recursive Error for Hidden Layers

$$\large
\delta^l = \left( \frac{\partial a^{l+1}}{\partial a^l} \right)^\top \cdot \delta^{l+1} \quad \to \quad \delta^l = (W^{l+1})^\top \cdot \delta^{l+1} \odot a^l \odot (1 - a^l)$$

------------------------------------------------------------------------

### Gradient of Weights

$$\large
\frac{\partial C}{\partial w^l} = \frac{\partial a^l}{\partial w^l} \overset{i}{\cdot} \delta^l \quad \to \quad \frac{\partial C}{\partial w^l} = \delta^l (a^{l-1})^\top$$

------------------------------------------------------------------------

### Gradient of Biases

$$\large
\frac{\partial C}{\partial b^l} = \delta^l$$

------------------------------------------------------------------------

This summary highlights the transition from theoretical derivatives to practical formulas used in neural network training.

# Implementation

Now we can finally dive into the implementation of these algorithm using the equations found above, you can find the complete code on my github page, indeed below you will find a brief explanation of the key concepts.

### Setup

We are going to setup the data structures used in the class, the methods `np.random,randn(x,y)` creates a structure of the shape indicated inside the parenthesis, created randomly.

``` python
self.num_layers = len(sizes)
self.sizes = sizes
self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
```

The zip function connect two turples, for example:

``` python
names = ["Simon", "Erik"]
scores = [15,22]
res = zip(names, scores)
print(res)

#the result would be
-->[("Simon", 15), ("Erik",22)]
```

For the biases we are creating a matrix of dimentions $\large y \times 1$, we are not assigning it a simple vector because with a matrix will be simpler doing later operation, like calculating the activation of a neuron.

the code `sizes[1:]` is used to ignore the first layer (the input layer of the network), this is because we don't want to assign biases this this layer.

In the `self.weights` part of code we are randomly assigning the weights of the network, the code `sizes[:-1], sizes[1:]` is used to take into account all the layers excluding the first and the last one.
If for example our `sizes` turple is `[1,2,1]` we would have two matrixes of dimentions: $\large 2 \times 1$ and $\large 1 \times 2$:

$$
\large
\begin{pmatrix}
w_{1,1}^1 \\
w_{2,1}^1
\end{pmatrix} \text{ and }

\begin{pmatrix}
w_{1,1}^2 w_{1,2}^2
\end{pmatrix}
$$

(remember the notation for weights of a network I've introduced at the beginning of this article).

This is just the introductory code, that set up in a random way the weights and biases of the network.

Then we have the main function of this algorithm `update_mini_batch`, is the function that update the weights and biases by applying gradient descend using backpropagation.

``` python
def update_mini_batch(self, mini_batch, eta):


    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

  

for x, y in mini_batch:

    delta_nabla_b, delta_nabla_w = self.backprop(x, y)

    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    self.weights = [w-(eta/len(mini_batch))*nw 
                    for w, nw in zip(self.weights, nabla_w)]

    self.biases = [b-(eta/len(mini_batch))*nb

                    for b, nb in zip(self.biases, nabla_b)]
```

The first two line are used to set up a data structure where the gradient for the biases and weights are going to be calculated, `np.zeros(b.shapes)` creates a numpy array of zeros with the shape of b.

The key part of this function is the loop, that for each value of the numpy array `mini_batch`, each item is composed of an input (x) and the expected output (y).

The most important line of code is `delta_nabla_b, delta_nabla_w = self.backprop(x, y)` because it calls the function backpropagation that gives back for each layer $l$:

$$\delta\nabla b^l =\large \frac{\partial C}{\partial b^{l}}$$

And

$$\delta\nabla w^l =\large \frac{\partial C}{\partial w^{l}}$$

``` python
nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
```

Those two lines are used to **accumulate the gradients** (partial derivatives) for each layer in a neural network during **mini-batch stochastic gradient descent (SGD)**.

Then we can update the weights and biases according to the equations we have previously analized:

$$ \large
w_k' = w_k - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial w_k}$$

$$\large
b_l' = b_l - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial b_l}$$

``` python
self.weights = [w-(eta/len(mini_batch))*nw 
    for w, nw in zip(self.weights, nabla_w)]

self.biases = [b-(eta/len(mini_batch))*nb
    for b, nb in zip(self.biases, nabla_b)]
```

Finally we are able to understand how this piece of code really work, and I'm going to explain in detail.

``` python
def backprop(self, x, y):

    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer

    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

# backward pass
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

    return (nabla_b, nabla_w)
```

These first two lines as in the update_mini_batch function are used to set up the shape of the the numpy array that we are going to return at the end of the function.

``` python
nabla_b = [np.zeros(b.shape) for b in self.biases]
nabla_w = [np.zeros(w.shape) for w in self.weights]
```

The next piece of code is the feedforward phase, a key part of the process, we need to know the activation of every single layer if we want to backpropagate the error.
Remember from before how we defined the activation for a layer $\large \ell$.

$$\large a^{\ell}=\sigma\left( W^{\ell}a^{\ell-1} + b^{\ell} \right)$$

The input x stores the activation of the **first layer of the network** while the activation array is going to store all the activations arrays of the network.
Remember also how we defined the middle quantity $\large z$:

$$\large z^{\ell} = W^{\ell}a^{\ell-1} + b^{\ell}$$

``` python
# feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer

    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
```

Remembering that it's easy to understand what this code is actually doing, is just applying the dot product between matrixes and adding the bias, and applying the activation function.

Here comes the key part of the algorithm, how do we write the code for a backward pass?
First of all remember we need the compute the error of the last layer as:

$$ \large
\delta^L = (a^L - y) \odot a^L \odot (1 - a^L)$$

And we compute also the partial derivative for the weights and biases for the last layer.
$$ \large
\frac{\partial C}{\partial w^L} = \delta^L (a^{L-1})^\top$$


$$\large
\frac{\partial C}{\partial b^L} = \delta^L$$

Then we repeat this for all the layers until the first one, we compute the activation of the layer, calculate the error thank to the backpropagation equation:

$$\large
\delta^l = (W^{l+1})^\top \cdot \delta^{l+1} \odot a^l \odot (1 - a^l)$$

And compute again the gradient of weights and biases, we'll stop until we have reached the first layer and return the result.

``` python
# backward pass
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

    return (nabla_b, nabla_w)
```

## References

- [Neural Networks and Deep Learning -- Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/optimization-2/)
- [Deep Learning Book -- Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press)](https://www.deeplearningbook.org/)
- [Medium-Step-By-Step Derivation](https://medium.com/data-science/backpropagation-step-by-step-derivation-99ac8fbdcc28)
