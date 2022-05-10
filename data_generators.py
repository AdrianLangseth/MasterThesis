import globals
import numpy as np

"""
Helper-file to make data for testing. The data here are only nonsense, but have the required properties 
in terms of dimensionality, data-types, max- and min-values etc

They are in two "versions": If we do not need class-label, the thing will just push out "x-data" as the return.
This is then a numpy array. If a class labels is requested, we get a dict.   
"""


def document_data_generator(also_generate_class: bool = False):
    """
    Simple data generator that just gives back random stuff.
    Each row is a document. A document has the following structure:
    -- Some integers that represent the words in the title
    -- Some integers that represent the words in the body
    -- One integer that represents the category
    -- One integer that represents the sub-category
    :param also_generate_class: Also generate class label
    :return: A numpy array
    """

    txt = np.random.randint(low=0, high=globals.model_params['vocabulary_size'],
                            size=(
                                globals.model_params['max_no_documents_in_user_profile'],
                                globals.model_params['title_size'] + globals.model_params['body_size']
                            ))
    categories = np.random.randint(low=0, high=globals.model_params['no_categories'],
                                   size=(globals.model_params['max_no_documents_in_user_profile'], 1))
    sub_categories = np.random.randint(low=0, high=globals.model_params['no_sub_categories'],
                                       size=(globals.model_params['max_no_documents_in_user_profile'], 1))

    ret = np.concatenate((txt, categories, sub_categories), axis=1)

    if also_generate_class:
        """
        Need some silly class variable. Here I'll check if the user has interacted with a document
        containing the word id 0. If so, it is class 1 otherwise it is class 0
        """
        minimums = np.min(ret, axis=-1)
        y_vals = 1 * (minimums == 0)

        ret = {'x': ret, 'y': y_vals}

    # Return
    return ret


'''
def user_doc_data_generator(no_users, no_docs, words_per_doc, vocab_size, also_generate_class=False):
    
    ret = np.random.randint(low=0, high=vocab_size, size=(no_users, no_docs, words_per_doc))

    if also_generate_class:
        """
        Need some silly class variable. Here I'll check if the user has interacted with a document
        containing the word id 0. If so, it is class 1 otherwise it is class 0
        """
        minimums = np.min(np.min(ret, axis=-1), axis=-1)
        y_vals = 1 * (minimums == 0)

        ret = {'x': ret, 'y': y_vals}

    # Return
    return ret
'''


def user_doc_data_generator(no_users: int, also_generate_class=False):
    """
    Generate data for a number of users. Do so by simply going one user at the time and generate data using
    document_data_generator for *that* user.
    :param no_users: No users to generate data for
    :param also_generate_class: Should we hae a class-thingy here, too?
    :return: Either a numpy array or a dict of arrays
    """
    ret = None
    for u in range(no_users):
        user_data = document_data_generator(also_generate_class=False)
        if ret is None:
            # Haven't generated the resulting data structure yet. I do it like this to get the first user
            # data telling me about shapes
            ret_shapes = [no_users] + list(user_data.shape)
            ret = np.zeros(shape=ret_shapes, dtype=user_data.dtype)
        ret[u] = user_data

    if also_generate_class:
        """
        Need some silly class variable. Here I'll check if the user has interacted with a document
        containing the word id 0. If so, it is class 1 otherwise it is class 0
        """
        minimums = np.min(np.min(ret, axis=-1), axis=-1)
        y_vals = 1 * (minimums == 0)

        ret = {'x': ret, 'y': y_vals}

    # Return
    return ret


def generate_naml_test_data(no_users):
    cand = user_doc_data_generator(no_users)[:, 0, :]
    hist = user_doc_data_generator(no_users, False)
    y = 1*(np.min(np.min(hist, axis=-1), axis=-1) < 4) * 1 * (np.min(cand, axis=-1) < 4)

    # todo: find better test_criteria
    return (cand, hist), y

