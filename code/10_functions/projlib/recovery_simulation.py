#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for recovering latent states from GMMHMM
Dr. Drew E. Winters
Created on 9/16/2025
"""

import numpy as np
import statistics


## Function for decoding by trial latent state from C_hat ____________
def decode_state_from_c_hat(ct, lengths):
    """ Takes the predicted conditions (c) or states 
        over the lengths specified to get the most probable value 
        for each latent cognitive state 
    """
    decoded = []
    start = 0
    for l in lengths:
        segment_probs = ct[start:start + l]
        mode = statistics.mode(segment_probs)
        decoded.append(mode)
        start += l
    return np.array(decoded)


## Function for decoding by trial latent state from posterior ________
def decode_state_from_posterior(post_probs, lengths):
    decoded = []
    start = 0
    for l in lengths:
        segment_probs = post_probs[start:start+l]
        summed_probs = segment_probs.sum(axis=0)
        decoded.append(np.argmax(summed_probs))
        start += l
    return np.array(decoded)