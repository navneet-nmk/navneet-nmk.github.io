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

## What are the solutions ?
Variational Empowerment
There are varying methods to calculate these which are kind of similar and just vary in the way how the lower bound to the mutual information is calculated.

## What is mutual information ?
Mutual information is a quantity that is used for measuring the relationship between random variables. 
It is defined as the decrease in uncertanity of a random variable x given another random variable z.

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

The marginal transition _p(s'|s)_ is the problem here. For continuous action spaces especially. Since we need to integrate over all actions to get this probability distribution. (Note, later in the post we will look at empowerment for discrete action spaces).

2 Ways to circumvent this intractable distribution:
1. Approximate p(s'|s) using variational approximation. (Non-trivial)
2. Replace p(s'|s) with the planning distribution (inverse dynamics distribution), p(a|s', s) and approximate this (still intractable) distribution. (Much easier than 1).










