import numpy as np
from w2v_utils import *

words,word_to_vec_map = read_glove_vecs('data/glove.6b.50d.txt')

#COISINE SIMILARITY

def cosine_similarity(u, v):
    """
    Compute the dot product between u and v
    Compute the L2 norm of u
    Compute the L2 norm of v
    Compute the cosine_similarity defined by fomula
    Return cosine_similarity
    """

    distance = 0.0
    dot = np.dot(u,v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity

# WORD ANALOGY TASK

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    *Convert words to lower case
    *Get the word embedding v_a, v_b and v_c
    *Initialize max_cosine_sim to large negative number
    *Initialize best_word with None, it will help keep track of the word to output
    *Loop over the whole word vector set
        *to avoid best_word being one of the input words, pass on them
        *Compute cosine similarity between the vector(e_b-e_a) and the vector((w's vector representation) - e_c)
        *If the cosine_sim is more than max_cosine_sim seen so far,
            *then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word
    Returns: best_word
    """

    word_a = word_a.lower()
    word_b = word_b.lower()
    word_c = word_c.lower()

    e_a = word_to_vec_map["word_a"]
    e_b = word_to_vec_map["word_b"]
    e_c = word_to_vec_map["word_c"]

    words = word_to_vec_map.keys()
    max_cosine_sim = -100

    best_word = None

    for w in words:
        if w in[word_a, word_b, word_c]:
            continue

        cosine_sim = cosine_similarity(e_b-e_a, word_to_vec_map[w] - e_c)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word

#DEBIASING WORD VECTORS

def neutralize(word, g, word_to_vec_map):
    """
    * Select word vector representation of "word". Use word_to_vec_map.
    * Compute e_biascomponent using formula give above.
    * neutralize e by substracting e_biascomponent from it
    * e_debiased should be equal to its orthogonal projection.
    Returns: e_debiased
    """
     e = word_to_vec_map[word]
     e_biascomponent = np.dot(e, g) / np.linalg.norm(g)**2 * 2
     e_debiased = e - e_biascomponent

     return e_debiased

def equalize(pair, bias_axis, word_to_vec_map):
    """
    * Select word vector representation of "word". Use word_to_vec_map.
    * Compute the mean of e_w1 and e_w2
    * Compute the projections of mu over the bias axis and the orthogonal axis
    * Compute e_w1b and e_w2B
    * Adjust the Bias part of e_w1B and e_w2B
    * Debias by equalizing e1 and e2 to the sum of their corrected projections
    Returns e1,e2
    """

    w1,w2 = pair
    e_w1,e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    mu = (e_w1 + e_w2) / 2
    mu_B = np.dot(mu, bias_axis) / np.linalg.norm(bias_axis)**2 *bias_axis
    mu_orth = mu - mu_B

    e_w1B = np.dot(e_w1, bias_axis) / np.linalg.norm(bias_axis)**2 *bias_axis
    e_w2B = np.dot(e_w2, bias_axis) / np.linalg.norm(bias_axis)**2 *bias_axis

    corrected_e_w1B = np.linalg.norm(1 - np.linalg.norm(mu_orth)**2)**(1/2) *(e_w1B - mu_B )/ np.linalg.norm((e_w1 - mu_orth) - mu_B)
    corrected_e_w2B =  np.linalg.norm(1 - np.linalg.norm(mu_orth)**2)**(1/2) *(e_w2B - mu_B) / np.linalg.norm((e_w2 - mu_orth) - mu_B)

    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1,e2
