# LSM

**Context:** Research internship during my first year of master\
**Year:** 2024-2025

## Description

Building a Liquid State Machine with Short-Term Plasticity from scratch based on Maass(2002) and the work of Ricaro de Azambuja *ricardo.azambuja@gmail.com*.

## Notes

at first, I only had one current channel for the lif neurons and was doing the current decay into the synapse's model as in : http://www.scholarpedia.org/article/Short-term_synaptic_plasticity

but here, 2 current channels (inh. and exc.) from the differential equation from :
    
    - R. de Azambuja, F. B. Klein, S. V. Adams, M. F. Stoelen, A. Cangelosi
        Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller.

and doing the current decay among the LIF model for each current channel


