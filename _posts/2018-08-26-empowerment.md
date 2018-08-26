---
layout: post
published: true
title: Empowerment
subtitle: Empowerment as intrinsic reward
date: '2018-08-26'
---
## Explaining Empowerment.

Reinforcement Learning is hard. Despite the various successes of Deep Reinforcement Learning, a limitation of the standard reinforcement learning approach is that an agent is only able to learn using external rewards obtained from it's environment. 
Truly autonomous agents are required to function in environments with no rewards or very sparse reward signals.
Intrinsically motivated reinforcement learning is a variant that aims to tackle the problem of no reward reinforcement learning by equipping the agent with intrinsic rewards such as Curiosity ([Curiosity is all you need](https://navneet-nmk.github.io/2018-08-10-first-post/)) and Empowerment (There could be many other formulations).

All these definitions have in common the way they allow an agent to reason about the value of information in action-state sequences it observes. 
Empowerment is one such measure that which is defined as the channel capacity (Information theory reference ) between actions and states that maximizes the influence of an agent on it's near future.

Information theoretically, empowerment is defined as the mutual information between the action and the subsequent state acheived by that action. Policies that take into account the empowerment of a state consistently drives the agent to states with high potential.

This all sounds fine and dandy. Then why don't we see much of empowerment in the current literature. Because it is seriously hard to calculate the empowerment. Really hard.

It requires integrating over all actions and states which is okay for small environments and discrete action spaces but enter the continuous realm and all hell breaks loose.

Most works circumvent these issues by using discrete actions spaces.

### What are the solutions ?
**Variational Empowerment**:
There are varying methods to calculate these which are kind of similar and just vary in the way how the lower bound to the mutual information is calculated.

### What is mutual information ?
Mutual information is a quantity that is used for measuring the relationship between random variables. 
It is defined as the decrease in uncertanity of a random variable x given another random variable z.

### Why use mutual information and not correlation?
Mutual information, unlike correlation, is able to capture non-linear statistical dependencies between random variables.

### KL Divergence Representation
What is the KL Divergence ? - The KL Divergence measures how one probability distribution p diverges from a second expected probability distribution q.

![KL Divergence](http://latex.codecogs.com/gif.latex?D%5Cleft%28P%20%5Cmiddle%5C%7C%20Q%5Cright%29%20%3D%20E_%7Bp%7D%5B%5Clog%5Cfrac%7BdP%7D%7BdQ%7D%5D%20%3D%20%5Cint%20P%5Clog%5Cfrac%7BP%7D%7BQ%7DdP)

MI is equivalent to the KL divergence between the joint and the product of the marginals.
Larger the divergence, stronger the interdependence between the random variables.
Note that MI can also be represented in the form of the shannon entropies.

![MI-KL Divergence Representation](http://latex.codecogs.com/gif.latex?I%28X%3BZ%29%20%3D%20D%5Cleft%28P_%7Bxz%7D%20%5Cmiddle%5C%7C%20P_%7Bx%7D%5Cotimes%20P_%7Bz%7D%5Cright%29%20%3D%20H%28X%29%20-%20H%28X%7CZ%29)

### What is empowerment ? Mathematically.
Empowerment is defined as the mutual information between the control input and the subsequent state.

![Empowerment](http://latex.codecogs.com/gif.latex?%5Cvarepsilon%20%28s%29%20%3D%20max_%7Bw%7DI%28S%5E%7B%27%7D%2C%20a%7CS%29)

where w is the source distribution policy (Not to be confused with the empowerment(or intrinsic reward) maximizing policy).

### Empowerment KL Divergence Representation
Empowerment can be represented in the KL divergence because MI has a KL divergence representation. 

![Empowerment-KL Divergence](http://latex.codecogs.com/gif.latex?KL%20%28p%28s%5E%7B%27%7D%2C%20a%7Cs%29%7C%7Cp%28s%5E%7B%27%7D%7Cs%29w%28a%7Cs%29%20%29)

The marginal transition **p(s'\|s)** is the problem here. For continuous action spaces especially. Since we need to integrate over all actions to get this probability distribution. (Note, later in the post we will look at empowerment for discrete action spaces).

Two Ways to circumvent this intractable distribution:

1.Approximate **p(s'\|s)** using variational approximation. (Non-trivial)
2.Replace **p(s'\|s)** with the planning distribution (inverse dynamics distribution), **p(a\|s', s)** and approximate this (still intractable) distribution. (Much easier than 1).

### How do we use the planning distribution? 
Avoid the integral over all actions by switching to the planning distribution.

![Mutual Information-Modified](http://latex.codecogs.com/gif.latex?I%28s%27%2C%20a%7Cs%29%20%3D%20%5Cint%20%5Cint%20p%28s%27%2C%20a%7Cs%29%5Cln%20%5Cfrac%7Bp%28s%27%7Cs%29p%28a%7Cs%27%2C%20s%29%7D%7Bp%28s%27%7Cs%29w%28a%7Cs%29%7D)

Despite being intractable, the planning distribution can be approximated by another distribution **q(a\|s', s)**. 
So finally we have the lower bound to the mutual information that we required.

![Lower Bound - Mutual Information](http://latex.codecogs.com/gif.latex?I%28s%27%2C%20a%7Cs%29%20%5Cgeq%20%5Cint%20%5Cint%20p%28s%27%2C%20a%7Cs%29%5Cln%20%5Cfrac%7Bq%28a%7Cs%27%2C%20s%29%7D%7Bw%28a%7Cs%29%7D%20ds%27%20du)

Since, I is constant with respect to q, maximizing the lower bound with respect to the planning distribution approximation (q) and the source distribution w, we can maximze the mutual information (or empowerment).

One caveat:
This lower bound is dependent on how well the distribution q is able to approximate the true planning distribution. Better the approximation, tighter the lower bound.

### How do we calculate the lower bound ?
We can sample from the system dynamics (assuming it is available) and the source distribution and estimate the gradients of the lower bound of Mutual Information using Monte Carlo Sampling and the reparameterization trick. ([Simple explanation of the reparameterization trick](https://medium.com/@llionj/the-reparameterization-trick-4ff30fe92954))

### Exploiting the reward (or the Empowerment)
The estimation of the empowerment enables us to efficiently learn the source distribution, w.
The main goal of this training is not to lean a policy but to learn to estimate empowerment of a particular state.
Empowerment can be perceived as the unsupervised value of a state, it can be used to train agents that proceed towards empowering states.
The reward in the reinforcement learning setting is then the negative empowerment value. 


### Enough of maths. Practical Approach.
Have 3 networks-
1. Forward Dynamics Model that takes in the current state and the action and predicts the next state.
2. The Policy Network, pi, that takes in the current state and predicts the action.
3. The Source network, w, that takes in the current state and predicts the action (this is used for the calculation of the empowerment of a state).
4. The Planning Network, q, that takes in the current and the next state and predicts the action (this is similar to the inverse dynamics model in [Curiosity is all you need](https://navneet-nmk.github.io/2018-08-10-first-post/))

**The Training Loop**:
1. Sample the environment using the Policy Network.
2. Train the forward dynamics model.
3. Sample one step action from the source distribution and the next state from the forward      dynamics model.
4. Calculate the mutual information (the Empowerment).
5. Calculate the reward function. (A function of the Empowerment)
6. Gradient ascent on w and q.
7. Gradient ascent on pi.

Source Distribution

	class source_distribution(nn.Module):

    def __init__(self,
                 state_space,
                 action_space,
                 conv_kernel_size,
                 conv_layers,
                 hidden, input_channels, height, width):
        super(source_distribution, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.input_channels = input_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers
        self.height = height
        self.width = width

        # Source Architecture
        # Given a state, this network predicts the action

        self.conv1 = nn.Conv2d(in_channels=self.input_channels,
                               out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers,
                               out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers,
                               out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2,
                               out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size, stride=2)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Hidden Layers
        self.hidden_1 = nn.Linear(in_features=self.height//16*self.width//16*self.conv_layers*2,
                                  out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.action_space)

        # Output activation function
        self.output_activ = nn.Softmax()

    def forward(self, current_state):
        x = self.conv1(current_state)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.lrelu(x)

        x = x.view((-1, self.height//16*self.width//16*self.conv_layers*2))
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.output(x)
        output = self.output_activ(x)

        return output
        
       
Inverse Dynamics Distribution

	class inverse_dynamics_distribution(nn.Module):
    
    def __init__(self, state_space,
                 action_space,
                 height, width,
                 conv_kernel_size,
                 conv_layers, hidden):
        super(inverse_dynamics_distribution, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.height = height
        self.width = width
        self.conv_kernel_size = conv_kernel_size
        self.hidden = hidden
        self.conv_layers = conv_layers

        # Inverse Dynamics Architecture

        # Given the current state and the next state, this network predicts the action

        self.conv1 = nn.Conv2d(in_channels=self.input_channels*2,
                               out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers,
                               out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers,
                               out_channels=self.conv_layers * 2,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers * 2,
                               out_channels=self.conv_layers * 2,
                               kernel_size=self.conv_kernel_size, stride=2)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Hidden Layers
        self.hidden_1 = nn.Linear(in_features=self.height // 16 * self.width // 16 * 	self.conv_layers * 2,
                                  out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.action_space)

        # Output activation function
        self.output_activ = nn.Softmax()

    def forward(self, current_state, next_state):
        state = torch.cat([current_state, next_state], dim=-1)
        x = self.conv1(state)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.lrelu(x)

        x = x.view((-1, self.height // 16 * self.width // 16 * self.conv_layers * 2))
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.output(x)
        output = self.output_activ(x)

        return output
        
        
Forward Dynamics Distribution



### Empowerment for discrete action spaces. (Current Research)

For the past few months, I have been dabbling with intrinsic rewards for the atari games. Learning these games and reaching super human performance is non trivial, especially for sparse reward games such as Montezuma's revenge, where exploration is of prime importance.




[Pytorch-RL](https://github.com/navneet-nmk/pytorch-rl)