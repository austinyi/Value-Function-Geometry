# Value Function Geometry
## 0. Index
1. [Introduction](#intro)
2. [Background](#pre)
3. [Value Function Linear Approximations](#approx)
4. [Value Function Polytope in Reinforcement Learning](#polytope)
5. [Summary](#sum)

## 1. <a name="intro">Introduction</a>
  
Recently, research on the geometric properties of value functions and attempts to apply these properties to various fields of Reinforcement Learning (RL) have been actively conducted. This post is an introduction to Value Function Geometry including an introduction of the paper "The Value Function Polytope in Reinforcement Learning" by Dadashi et al<sup>[1](#Dadashi:2019)</sup>. 

This post aims to explain the basic concepts of value function linear approximation and representation learning. Also, the paper<sup>[1](#Dadashi:2019)</sup> establishes the geometric shape of value function space in finite state-action Markov decision processes: a general polytope. By reimplementing the paper myself, I was able to check the surprising theorems and geometric properties. I have attached some results that I checked directly in the post.

<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure1.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 1: Mapping between policies and value functions<sup><a href="#Dadashi:2019">1</a></sup></i></font>
</center>
<br>


## 2. <a name="pre">Background</a>

We usually consider an environment with a Markov Decision Process(MDP) in Reinforcement Learning<sup>[2](#Sutton:2018)</sup>. Markov Decision Process $\langle\mathcal{X}, \mathcal{A}, r, P, \gamma\rangle$ is a Markov reward process with decisions, where $\mathcal{X}$ is the state space, $\mathcal{A}$ is the action space, $r$ is the reward function, $P$ is the transition function, and $\gamma \in$ (0,1) is the discount factor. 

A stationary policy  $\pi$ is a mapping from states $x \in \mathcal{X}$ to distributions over actions $a \in \mathcal{A}$. We denote the space of all policies by $\mathcal{P}(\mathcal{A})^\mathcal{X}$. Given a policy $\pi$ and a transition function $P$, we can obtain the state-to-state transition function
$$\mathcal{P^\pi}(x'|x):=\sum_{a\in\mathcal{A}}\pi(a|x)P(x'|x,a)$$

### State-value function

The return $G_t$ is the total discounted reward from time-step $t$ and state $x_t$ by following a policy $\pi$; $x_{t}$ and $a_{t}$ are the state and action at time $t$.
$$G_t=\sum_{k=0}^{\infty} \gamma^k R(x_{t+k}, a_{t+k})$$

The state-value function $V^\pi$ is defined as the expected return $G_t$ starting from a particular state and following policy $\pi$.

$$V^\pi(x) = \mathbb{E}_{P^\pi}[G_t \mid X_t=x]$$

### Action-value function

The action-value function $Q^\pi$ is defined as the expected return $G_t$ starting from a particular state and action, and then following policy $\pi$.

$$Q^\pi(x,a)=\mathbb{E}_{P^\pi}[G_t \mid X_t=x, A_t=a]$$

In particular, the state-value function is an expectation over the action-value functions under a policy $\pi$.

$$V^\pi(x) =\sum_{a \in \mathcal{A}}\pi(a|x)Q^\pi(x,a)$$

In the case, we take an action by following an argmax policy $\pi$,  

$$V^\pi(x) =max_{a \in \mathcal{A}}Q^\pi(x,a).$$

<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure2.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 2: Geometric View of Value Function<sup><a href="#Poupart:2013">3</a></sup></i></font>
</center>
<br>

Figure 2 represents the geometric relationship between the state-value function and the action-value function, when following an argmax policy. Each line corresponds to the action-value function and the bold line corresponds to the state-value function. It can be seen that the maximum action-value selected for each state is the state-value function. In Figure 2, it can also be checked that $\alpha_4$ does not play any useful role in the state-value function.

<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure3.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 3: Value Functions for the mountain car problem using</i></font>
</center>
<br>

Figure 3 is an example of value functions that I obtained for the mountain car problem. Both used DQN (Deep Q-Networks) to train the agent. Figure 3(a) is based on 2000 runs of mountain car without changing the reward, and Figure 3(b) is based on 500 runs of mountain car with adding a heuristic reward. I gave a bonus reward of 15 times the velocity of the next state. By giving a heuristic reward to the agent, the mountain car was able to learn and train faster.


## 3. <a name="approx">Value Function Linear Approximations</a>

A value function for a policy $\pi$ is denoted as $V^\pi:\mathcal{X}\rightarrow \mathbb{R}$, and a d-dimensional representation is a mapping $\phi :\mathcal{X}\rightarrow \mathbb{R}^d$; $\phi(x)=(\phi_1(x), \phi_2(x), \cdots, \phi_d(x))$ is the feature vector for state $x$, where $\phi_1, \phi_2, \cdots, \phi_d:\mathcal{X}\rightarrow\mathbb{R}$ are the feature functions.

For a given representation $\phi$ with a weight vector $\theta=(\theta_1, \theta_2, \cdots, \theta_d)\in\mathbb{R}^d$, the linear approximation for a value function $\hat{V}_{\phi, \theta}:\mathcal{X}\rightarrow\mathbb{R}$ is defined as:

$$\hat{V}_{\phi, \theta}(x):=\phi(x)^T\theta=\sum_{i=1}^{d}\theta_i\cdot\phi_i(x)$$

<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure4.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 4: A deep reinforcement learning architecture viewed as a two-part approximation</i><sup><a href="#Bellemare:2019">4</a></sup></font>
</center>
<br>

### Bellman operator

Assume a finite state space:

$$\mathcal{X}= \left \{x_1, x_2, \cdots,x_n \right\}.$$

Let's start by introducing some vector and matrix notation to make it more convenient. 


$$r_\pi(x):=\sum_{a\in\mathcal{A}}\pi(a|x)\times r(x,a);$$

$$r_\pi:=(r_\pi(x_1), r_\pi(x_2),\cdots,r_\pi(x_n))$$

$$\mathcal{P^\pi}
(x'|x)=\sum_{a\in\mathcal{A}}\pi(a|x)P(x'|x,a)$$

$$\mathcal{P^\pi}:=(\mathcal{P}_\pi(x_j|x_i))_{i\times j}; 1\leq i,j \leq n$$

$$V^\pi(x) = \mathbb{E}_{P^\pi}[G_t \mid X_t=x]$$

$$V^\pi:=(V^\pi(x_1),V^\pi(x_2),\cdots,V^\pi(x_n))$$

It is well known that the state-value function $V^\pi(x)$ satisfies Bellman's equation:

$$V^\pi(x)=r_\pi(x)+\gamma\mathbb{E}_{P_\pi}V^\pi(x').$$


Given a policy $\pi$, we define the Bellman operator $B_\pi$ as:

$$B_\pi V=r_\pi+\gamma\mathcal{P}_\pi V$$

Since $V^\pi(x)$ satisfies Bellman's equation, $V^\pi$ is the fixed point of operator $B_\pi$; $B_\pi V_\pi=V_\pi$. Then,

$$V^\pi=r_\pi+\gamma\mathcal{P}_\pi V^\pi=(I-\gamma\mathcal{P}_\pi)^{-1}r_\pi$$

In the case, we do not know the entire model completely, we cannot calculate $V^\pi$ explicitly by the above equation. However, by starting with any vector $V$ and applying the Bellman operator $B_\pi$ repeatedly, it will finally converge and reach the fixed point $V^\pi$. This can be explained by the Contraction Mapping Theorem.

### Projection operator

Projection operator is a widely used operator in Statistics for Multiple Linear Regression Models and Least Squares Estimation.

For an $n\times k$ matrix $U$ of full column rank, $\mathcal{C}_U$ is the column space of $U$, the linear subspace in $\mathbb{R}^n$ spanned by the columns of the matrix $U$.
$$\mathcal{C}_U=\left \{ \beta_1u_1+\beta_2u_2+\cdots \beta_ku_k:\beta_1,\beta_2,\cdots,\beta_p \in \mathbb{R} \right \}$$

$$=\left \{ U{\beta}:\beta\in \mathbb{R}^k \right \}$$

 We let $\Pi_U$ denote the projection operator for subspace spanned by $U$ such that $$\Pi(Y|\mathcal{C}_U)=\Pi_UY=U\hat{\beta}.$$
 
 Projector operator satifies the normal equations:
$$Y-\Pi_UY=Y-U\hat{\beta}\perp\mathcal{C}_U$$

$$\Leftrightarrow U^T(Y-U\hat{\beta})=0 $$

$$\Leftrightarrow \hat{\beta}=(U^TU)^{-1}U^TY$$

$$\Leftrightarrow\Pi_U=U(U^TU)^{-1}U^T$$

What if $U^TU$ is not invertible? Then, we will have infinite solutions for $\hat{\beta}$ and $\Pi_U$. However, since $U$ is full column rank, it implies that $U^TU$ is positive definite and invertible.

Back to the value function approximation, let's assume a finite state space $\mathcal{X}$ of $n$ states; $\mathcal{X}= \left \{x_1, x_2, \cdots,x_n \right\}$. Also, we write $\Phi\in\mathbb{R}^{n\times d}$ to denote the matrix whose columns are $\left \{ \phi_j(x_1),\phi_j(x_2),\cdots,\phi_j(x_n)\right \},$ $j=1,2,\cdots,d$.

We consider the approximation minimizing the squared error:
$$L(\phi; \pi)=\left \| \hat{V}_{\phi,\theta}-V^\pi \right \|_{2}^{2}=\sum_{x\in \mathcal{X}}(\phi(x)^T\theta-V^\pi(x))^2.$$

$\Pi_\Phi$ performs an orthogonal projection of value function vector $V^\pi$ onto the linear subspace $H=\left \{ \Phi\theta: \theta \in \mathbb{R}^d\right \}$. So, $\Pi_\Phi V^\pi$ is the value function we seek when doing linear function approximation in subspace $H$ defined by minimizing $L(\phi;\pi)$.


## 4. <a name="polytope">Value Function Polytope in Reinforcement Learning</a>


### Definition 1 : Polytope

$P$ is a ***convex polytope*** in $\mathbb{R}^n$ iff there are $k \in \mathbb{N}$ points $x_1, x_2, \cdots, x_k ∈ \mathbb{R}^n$ such that $P = Conv(x_1, \cdots , x_k)$.

We write $Conv(x_1, \cdots , x_k)$ to denote the convex hull of the points $x_1, x_2, \cdots, x_k$.

$P$ is a  (possibly non-convex) ***polytope*** iff it is a finite union of convex polytopes.

### Definition 2 : Polyhedron

$P$ is a ***convex polyhedron*** iff there are $k \in \mathbb{N}$ half-spaces $\hat{H}_1, \hat{H}_2, \cdots, \hat{H}_k$ whose intersection is $P$, that is 
$$P=\bigcap_{i=1}^{k}\hat{H}_k.$$

$P$ is a (possibly non-convex) ***polyhedron*** iff it is a finite union of convex polyhedra.

### The Space of Value Functions

The main contribution of Dadashi et al<sup>[1](#Dadashi:2019)</sup> is the characterization of the space of value functions. The space of value functions is the set of all value functions obtained from all possible policies.

$$f_v:\mathcal{P}(\mathcal{A})^\mathcal{|X|}\rightarrow \mathbb{R}^\mathcal{|X|}$$

$$\pi\mapsto V^\pi=(I-\gamma P^\pi)^{-1}r_\pi$$
 
$$\mathcal{V}=f_v(\mathcal{P}(\mathcal{A})^\mathcal{|X|})=\left \{f_v(\pi)|\pi \in  \mathcal{P}(\mathcal{A})^\mathcal{|X|}) \right \}$$

Dadashi et al<sup>[1](#Dadashi:2019)</sup> show that $\mathcal{V}=f_v(\mathcal{P}(\mathcal{A})^\mathcal{|X|})$ is a polytope. Instead of proving it directly by the definition of (convex) polytope, they use the important proposition that a bounded, convex polyhedron is a convex polytope<sup>[6](#Ziegler:2012)</sup>.

<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure5.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 5: Space of Value Functions for 2-state MDPs </i></font>
</center>
<br>

Figure 5 depicts the value function space $\mathcal{V}$ corresponidng to two different 2-state MDPs. The space is made of 100,000 value functions corresponding to 100,000 randomly sampled policies from $\mathcal{P}(\mathcal{A})^\mathcal{|X|}$. The specific details of MDPs presented in this figure are as follows.

#### Figure 5: Left
$$|\mathcal{A}|=2, \gamma=0.9$$

$$\hat{r}=[0.68, -0.89, 0.90, -0.90]$$

$$\hat{P}= [[0.3, 0.7], [0.9, 0.1], [0.1, 0.9], [0.4, 0.6]]$$

#### Figure 5: Right
$$|\mathcal{A}|=3, \gamma=0.8$$

$$\hat{r}=[-0.1, -1.0, 0.1, 0.4, 1.5, 0.1]$$

$$\hat{P}= [[0.90, 0.10], [0.20, 0.80], [0.70, 0.30],$$

$$[0.05, 0.95], [0.25, 0.75], [0.30, 0.70]]$$

I used the following convention to express $r$, $P$ (also the paper): 
$$r(s_i,a_j)=\hat{r}[i\times|\mathcal{A}|+j]$$

$$P(s_k|s_i,a_j)=\hat{P}[i\times|\mathcal{A}|+j][k]$$

To easily check the shape of the polytope, I checked with the value function spaces from 2-state MDPs, which is convenient to visualize. The policy space $\mathcal{P}(\mathcal{A})^\mathcal{|X|}$ is the Cartesian product of $|X|$ simplices. However, in general, a wide variety of polytope shapes are possible for the value function space $\mathcal{V}$.

In fact, if you look closely at the right of Figure 5, you'll find something a little odd. You may think it's not a perfect polytope because the upper left hand is a little empty. To double-check if the policy was sampled uniformly from the policy space, I visualized the mapping between policy samples and value function samples. 

<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure6.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 6: Mapping between sample policies and sample value functions. The red points are the value functions of deterministic policies </i></font>
</center>
<br>

To easily visualize the policy space and value function space, 2-state 2-action MDP was used for Figure 6. It includes 50,000 pairs of policy and value function samples. The specific details of MDP presented in this figure are as follows.

$$|\mathcal{A}|=2, \gamma=0.9$$

$$\hat{r}=[0.68, -0.89, 0.90, -0.90]$$

$$\hat{P}= [[ 0.3, 0.7], [ 0.9, 0.1], [ 0.1, 0.9], [ 0.4, 0.6]]$$

The left of Figure 6 definitely shows that the policy has been sampled uniformly. Despite the uniformly selected policies, you can still see the sparse empty space in the upper left corner of the value function space. Also, to confirm the existence of the polytope's vertex, I took the value functions of deterministic policies and confirmed that they become the vertex. In conclusion, we can know that the sparse empty space in the figure is simply due to the lack of sample numbers. It can also be seen that the polytope of the value function space does not consist of equal density.

### Definition 3 : Policy Agreement

Two policies $\pi_1,\pi_2$ ***agree*** on states $s_1,\cdots,s_k\in\mathcal{X}$ if $\pi_1(\cdot |s)=\pi_2(\cdot |s)$ for each $s=s_1,\cdots,s_k.$

For a given policy $\pi$, we will denote by $Y_{s_1,\cdots,s_k}^{\pi}\subseteq \mathcal{P(A)^X}$ the set of policies which agree with $\pi$ on $s_1,\cdots,s_k$. Thus, $Y_{\mathcal{X}\setminus \left \{ s \right \}}^{\pi}$, for $s\in\mathcal{X}$, describes the set of policies that agree with $\pi$ on all states except $s$.

### Definition 4 : Policy Determinism

A policy $\pi$ is
 - ***$s$-deterministic*** for $s\in\mathcal{X}$ if $\pi(a|s)\in \left \{0,1 \right \}$.
 - ***semi-deterministic*** if it is $s$-deterministic for at least one $s\in\mathcal{X}$.
 - ***deterministic*** if it is $s$-deterministic for all states $s\in\mathcal{X}$.

We denote by $D_{s,a}$ the set of semi-deterministic policies that take action $a$ on state $s$.


### Value Functions and Policy Agreement

Now, I will introduce one of the most amazing theorem of this paper, line theorem. It characterizes the subsets of value function space that are generated by the policies which agree on all but one state. Surprisingly, it describes a line segment within the value function polytope and this line segment is monotone.

### Theorem 1 : Line Theorem

*Let $s$ be a state and $\pi$, a policy. Then there are two $s$-deterministic policies in $Y_{\mathcal{X}\setminus \left \{ s \right \}}^{\pi}$, denoted $\pi_l,\pi_u$, which bracket the value of all other policies $\pi'\in Y_{\mathcal{X}\setminus \left \{ s \right \}}^{\pi}:$*

$$f_v(\pi_l)\preceq f_v(\pi') \preceq f_v(\pi_u).$$

*Furthermore, the image of $f_v$ restricted to $Y_{\mathcal{X}\setminus \left \{ s \right \}}^{\pi}$ is a line segment, and the following three sets are equivalent:*

 $$f_v(Y_{\mathcal{X}\setminus \left \{ s \right \}}^{\pi})$$
 
 $$\left \{f_v(\alpha\pi_l+(1-\alpha)\pi_u)|\alpha \in [0,1] \right \}$$
 
 $$\left \{\alpha f_v(\pi_l)+(1-\alpha)f_v(\pi_u)|\alpha \in [0,1] \right \}$$


Theorem 1 implies that we can generate $f_v(Y_{\mathcal{X}\setminus \left \{ s \right \}}^{\pi})$, the image of $f_v$ restricted to $Y_{\mathcal{X}\setminus \left \{ s \right \}}^{\pi}$, by drawing the value functions of mixtures of two policies, $\pi_l$ and $\pi_u$. Also, we can even generate it by just taking two value functions, $f_v(\pi_l)$ and $f_v(\pi_u)$, and drawing a line segment between the two points.

<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure7.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 7: Illustration of Theorem 1 Line Theorem. </i></font>
</center>
<br>

Figure 7 actually illustrates the value function space drawn by the interpolated policies between $\pi_l$ and $\pi_u$. The red points describe the value functions of mixtures of policies that agree everywhere but one state. By sampling 200 mixtures of policies that agree but one state and drawing each value function, we could see it forms a line segment.  The specific details of MDPs presented in Figure 7 are as follows.

#### Figure 7: Left
$$|\mathcal{X}|=2, |\mathcal{A}|=2, \gamma=0.9$$

$$\hat{r}=[0.68, -0.89, 0.90, -0.90]$$

$$\hat{P}= [[0.3, 0.7], [0.9, 0.1], [0.1, 0.9], [0.4, 0.6]]$$

#### Figure 7: Right
$$|\mathcal{X}|=2, |\mathcal{A}|=2, \gamma=0.9$$

$$\hat{r}=[-0.45, -0.1, 0.5, 0.5]$$

$$\hat{P}= [[ 0.7, 0.3], [ 0.99, 0.01], [ 0.2, 0.8], [ 0.99, 0.01]]$$

For both figures in Figure 7, the following $\pi_l$ and $\pi_u$ which agree on $s_1$ were used for the interpolation.

$$\hat\pi_l = [[0.2, 0.8], [0, 1]]$$

$$\hat\pi_u = [[0.2, 0.8], [1, 0]]$$

where $\pi(a_j|s_i)=\hat\pi[i][j]$.


<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure8.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 8: Value functions of mixtures of two policies in the general case. </i></font>
</center>
<br>

The red points in Figure 8 are the value functions of mixtures of two policies. You can see that the red points are curved, not straight. It can be seen that the condition, the policies should agree on all but one state, is a necessary condition.


By visualizing the policy space, we can easily recognize that the two policies do not agree on any states. In the case of $|\mathcal{X}|=2$ and $|\mathcal{A}|=2$, if the two policies agree on one state, the interpolation of the two policies shall be horizontal or vertical. The specific details of MDP presented in Figure 8 are same as the MDP presented in Figure 7: Left, and following $\pi_1$ and $\pi_2$ were used for mixtures.

$$\hat\pi_1 = [[0.2, 0.8], [0.8, 0.2]]$$

$$\hat\pi_2 = [[0.6, 0.4], [0.3, 0.7]]$$


***Corollary 1.*** *For any set of states $s_1,\cdots,s_k \in \mathcal{X}$ and a policy $\pi$, $V^\pi$ can be expressed as a convex combination of value functions of $\left \{ s_1,\cdots,s_k  \right \}$-deterministic policies. In particular, $\mathcal{V}$ is included in the convex hull of the value functions of deterministic policies.*

Corollary 1 can be proved by mathematical induction on the number of states $k$. The result of $k=1$ are seen directly from the Line Theorem.

<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure9.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 9: Visual representation of Corollary 1. The red dots are the value functions of deterministic policies.</i></font>
</center>
<br>

As you can see in Figure 9, the space of value functions is included in the convex hull of value functions of deterministic policies. Also, you can check the relationship between the vertices of $\mathcal{V}$ and deterministic policies. However, in Figure 10, we can also observe the fact that the value functions of deterministic policies are not necessarily the vertices of $\mathcal{V}$ and that the vertices of $\mathcal{V}$ are not necessarily the value functions of deterministic policies.

<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure10.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 10: Visual representation of Corollary 1. The value functions of deterministic policies are not identical to the vertices of V. </i></font>
</center>
<br>



### The Boundary of $\mathcal{V}$

Theorem 2 and Corollary 3 show that the boundary of value function space is described by semi-deterministic policies.

Let's start with first defining the affine vector space before we mention Theorem 2. Consider a policy $\pi$ and $k$ states $s_1,\cdots,s_k$. Then,


$$H_{s_1,\cdots,s_k}^{\pi}=V^\pi+Span(C^\pi_{k+1},\cdots, C^{\pi}_{|\mathcal{X}|}), $$

where $C^\pi_{k+1},\cdots, C^{\pi}_{|\mathcal{X}|}$ are the columns of the matrix $(I-\gamma P^\pi)^{-1}$ corresponding to states other than $s_1,\cdots,s_k$.

### Theorem 2.

*Consider the ensemble of policies $Y_{s_1,\cdots,s_k}^{\pi}$ that agree with $\pi$ on states $\mathcal{X}'= \left \{ s_1, \cdots , s_k \right \}$. Suppose $\forall s \notin \mathcal{X}'$, $\forall a \in \mathcal{A}$, $\nexists \pi' \in Y_{s_1,\cdots,s_k}^{\pi} \cap D_{s,a}$  s.t. $f_v(\pi')=f_v(\pi)$, then $f(v_\pi)$ has a relative neighborhood in $\mathcal{V}$ $\cap$ $H_{s_1,\cdots,s_k}^{\pi}$.*


***Corollary 3.*** *Consider a policy $\pi \in \mathcal{P(A)^X}$, the states $\mathcal{X}'=\left \{ s_1, \cdots , s_k \right \}$, and the ensemble $Y_{s_1,\cdots,s_k}^{\pi}$ of policies that agree with $\pi$ on $s_1,\cdots,s_k$. Define $\mathcal{V}^y=f_v( Y_{s_1,\cdots,s_k}^{\pi})$, we have that the relative boundary of $\mathcal{V}^y$ in  $H_{s_1,\cdots,s_k}^{\pi}$ is included in the value functions spanned by policies in $Y_{s_1,\cdots,s_k}^{\pi}$ that are s-deterministic for $s\notin \mathcal{X}':$*

$$\partial \mathcal{V}^y \subset \bigcup_{s \notin \mathcal{X}'} \bigcup_{a \in \mathcal{A}} f_v( Y_{s_1,\cdots,s_k}^{\pi}\cap D_{s,a})$$

*where $\partial$ refers to the relative boundary $\partial_{H_{s_1,\cdots,s_k}^{\pi}}$.*


<br>
<center>
    <img src="https://ml2blogpost.s3.ap-northeast-2.amazonaws.com/imgs_bongsoo/figure11.png" width=400/>
</center>
<center>
    <font size=2pt><i>Figure 11: Visual representation of Corollary 3. The orange points are the value functions of semi-deterministic policies.</i></font>
</center>
<br>

### The Polytope of Value Functions

By combining all the previous results, Dadashi et al<sup>[1](#Dadashi:2019)</sup> proved the value function space $\mathcal{V}$ is a polytope. Theorem 3 is the final result of this paper<sup>[1](#Dadashi:2019)</sup>.

### Theorem 3.

*Consider a policy $\pi \in \mathcal{P(A)^X}$, the states $s_1,\cdots,s_k \in \mathcal{X}$, and the ensemble $Y_{s_1,\cdots,s_k}^{\pi}$ of policies that agree with $\pi$ on $s_1,\cdots,s_k$. Then $f_v(Y_{s_1,\cdots,s_k}^{\pi})$ is a polytope and in particular, $\mathcal{V}=f_v(Y^\pi_{\empty})$ is a polytope.*

Theorem 3 can be proved by induction on $k$, the cardinality of the number of states. If $k=|\mathcal{X}|$, $Y_{s_1,\cdots,s_k}^{\pi}$ is the set of policies which agree with $\pi$ on every state. Since only $\pi$ can agree with $\pi$ on every state, $Y_{s_1,\cdots,s_k}^{\pi}=\left \{ \pi \right \}$. Therefore, $f_v(Y_{s_1,\cdots,s_k}^{\pi})=\left \{ f_v(\pi) \right \}$ is a polytope. Now, assume the theorem is true for $k=l+1$ and prove that it is true for $k=l$. This will prove the Theorem 3.



## 5. <a name="sum">Summary</a>

There have been many developments in the field of value function geometry recently. In the case of finite state-action spaces, Dadashi et al<sup>[1](#Dadashi:2019)</sup> characterized the shape of value function space: a general polytope. It helps our understanding of the dynamics of reinforcement learning algorithms.

Although there is still a problem with generalizing to infinite or continuous state-action spaces, it is certain to be a great study that has created a new direction exploring the field of reinforcement learning. Bellemare et al<sup>[4](#Bellemare:2019)</sup> discovered a connection between representation learning and the polytopal structure of value functions, and a number of follow-up papers are being published<sup>[7](#Dabney:2020)</sup> <sup>[8](#Ghosh:2020)</sup>. Also, exploring the relationship between value function approximation and the geometry of value functions will be another interesting field of study.

Before joining ML2, I did not have a good understanding of reinforcement learning. However, thanks to all of the members of ML2, I was able to gain valuable research experience in reinforcement learning and the field of value function geometry during my internship. I am still deeply interested in developing the research during my internship.


### Reference

<!-- 1 -->
##### <sup><a name="Dadashi:2019">1</a></sup><sub>Dadashi, R., Taiga, A. A., Roux, N.L., Schuurmans, D., and Bellemare, M. G. [The value function polytope in reinforcement learning,](https://arxiv.org/abs/1901.11524) In <i>ICML</i>, 2019.</sub>
<!-- 2 -->
##### <sup><a name="Sutton:2018">2</a></sup><sub>Sutton, R. S. and Barto, A. G. [Reinforcement Learning: An Introduction,](https://mitpress.mit.edu/books/reinforcement-learning-second-edition) MIT Press, 2nd edition, 2018.</sub>
<!-- 3 -->
##### <sup><a name="Poupart:2013">3</a></sup><sub>Poupart, P. and Boutilier, C. [Value-Directed Belief State Approximation for POMDPs,](https://arxiv.org/pdf/1301.3887.pdf) <i>arXiv preprint arXiv:1301.3887</i>, 2013.</sub>
<!-- 4 -->
##### <sup><a name="Bellemare:2019">4</a></sup><sub>Bellemare, M. G., Dabney, W., Dadashi, R., Taiga, A. A., Castro, P. S., Roux, N. L., Schuurmans, D., Lattimore, T., and Lyle, C. [A geometric perspective on optimal representations for reinforcement learning,](https://arxiv.org/pdf/1901.11530.pdf) In <i>Neural Information Processing Systems (NeurIPS)</i>, 2019.</sub>
<!-- 5-->
##### <sup><a name="Rao:2020">5</a></sup><sub>Rao, A. [Value Function Geometry and Gradient TD,](http://web.stanford.edu/class/cme241/lecture_slides/ValueFunctionGeometry.pdf) Stanford CME 241 Lecture Slides, 2020.</sub>
<!-- 6-->
##### <sup><a name="Ziegler:2019">6</a></sup><sub>Ziegler, G. M.  [Lectures on Polytopes,](https://link.springer.com/book/10.1007/978-1-4613-8431-1) volume 152. Springer Science & Business Media, 2012.</sub>
<!-- 7 -->
##### <sup><a name="Dabney:2020">7</a></sup><sub>Dabney, W., Barreto, A., Rowland, M., Dadashi, R., Quan, J., Bellemare, M. G. and Silver, D. [The Value-Improvement Path Towards Better Representations for Reinforcement Learning,](https://arxiv.org/pdf/2006.02243.pdf) <i>arXiv preprint arXiv:2006.02243</i>, 2020.</sub>
<!-- 8 -->
##### <sup><a name="Ghosh:2020">8</a></sup><sub>Ghosh, D. and Bellemare, M. G. [Representations for Stable Off-Policy Reinforcement Learning,](https://arxiv.org/pdf/2007.05520.pdf) <i>arXiv preprint arXiv:2007.05520</i>, 2020.</sub>


