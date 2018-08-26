---
layout: post
published: false
title: Empowerment
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

