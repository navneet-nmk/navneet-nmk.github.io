---
layout: post
published: true
title: Exploration via Hierarchical Meta Reinforcement Learning
---
Today we will discuss about a new research project/idea that I have been working on for the past couple of months. 

This post will be quite long so ...

**** TLDR: Combination of Stochasticity, task agnostic exploration and task based exploration measures greatly helps in sparse reward tasks.

Prologue

From the very beginning of my studies in machine learning, and specifically, reinforcement learning, I have been quite intrigued in the behaviour of agents in sparse reward tasks. 
Humans are extermely capable at exploring their environments and that too in an efficient manner. Having read about various exploration strategies such as curiosity, information theoretic approaches and count based exploration, the overarching theme that comes out is that most of these measures explore the environment quite well, agreed, albeit in a truly inefficient manner. 

I like to think about exploration in two ways - task agnostic and task based and I find this is analogous to the types of Reinforcement Learning - Model based and Model free. 
Bear with me- 
Task agnostic measures such as curiosity continuosly explore the environment but do so without leveraging any prior information. They keep on exploring the environment in a somewhat trial and error fashion to find the all elusive reward(Sounds something like Model Free RL, right?).

Task based exploration, on the other hand, explore in the new task using the prior information the agent gained exploring in previous tasks. No crazy, expensive trial and error(Model based RL bells ringing).

But you know what, both the above measures are not perfect. Task agnostic measures are quite inefficent but can grab that reward that lies in that remote area of the environment. 
While task based measures are pretty efficient but they may ignore some really 'hidden' rewards.

Moreover, humans don't really rely on just on of these ways to explore their environment. Often, they make use of a combination of both these measures to explore. And this is the motivation for the work that I am going to explain. 

But first some discussion of the types of exploration explained above namely, Empowerment and MAESN.

## Empowerment 

Empowerment is basically the control an agent has over it's future. 
More formally, it is defined as the channel capacity between actions and states that maximizes the influence of an agent on it's near future or the mutual information between the action and the subsequent state acheived by that action.

For the work we will discuss about, we need to focus on a specific paper (Beautiful Paper by the way) that makes use of empowerment to enable a policy to learn various diverse skills, Diversity is all you need.


### Diversity is all you need.

This paper closely follows prior work, Varitional Intrinsic Control, to learn complex, diverse skills that may or may not solve a given task. However unlike the VIC paper, they make a couple of important changes such as the use of a maximum entropy policy and a fixed skills distribution that greatly helps the policy to acquire useful , diverse skills. 

The following are the primary components of their algorithm:

1. Skills should dictate the state an agent visits.
2. States should be used to distinguish between the various skills (and not states and actions).
3. Those skills are learned that act as randomly as possible (Use a maximum entropy policy).

An overview of the algorithm presented in the paper: 

![DIAYN]({{site.baseurl}}/img/model-1.png)

They were actually able to learn pretty useful and diverse skills and were able to achieve pretty decent results on various continuous control tasks.

![DIAYN_rewards]({{site.baseurl}}/img/many_rewards.png)


So far so good. Now we will take a look at task based exploration method, mainly MAESN or Model Agnostic Exploration with Structured Noise (Yeah, that's a mouthful).

## MAESN

MAESN is based on gradient based meta learning. SO we will first learn about Meta Learning and specifically, Model Agnostic Meta Learning (MAML).

### MAML
