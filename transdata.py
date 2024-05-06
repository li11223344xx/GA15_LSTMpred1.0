import streamlit as st
import pandas as pd

import GA15v
import GA20v
import PCA15v
import PCA20v

def get_descriptor_module(descriptor_type):
    if descriptor_type == 'GA15v':
        return GA15v.GA15v
    elif descriptor_type == 'GA20v':
        return GA20v.GA20v
    elif descriptor_type == 'PCA15v':
        return PCA15v.PCA15v
    elif descriptor_type == 'PCA20v':
        return PCA20v.PCA20v

def process_sequence(seq, func):
    # Process a single sequence with given function
    if len(seq) == 6:  # Ensure sequence length is 6
        return func(seq)
    else:
        return f"Error: Sequence length {len(seq)} is not 6."

def process_file(contents, descriptor_type):
    # Process multiple sequences from file content
    results = []
    func = get_descriptor_module(descriptor_type)
    sequences = contents.strip().split()
    for seq in sequences:
        result = process_sequence(seq, func)
        results.append((seq, result))
    return results

