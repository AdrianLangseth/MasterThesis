"""
This is the entry-point to start running.
Basically only starts w & b, then kicks off whatever test we want to do
"""

import wandb
import globals
from user_encoder import __test_user_encoder_learning as test_learning
from data_generator import main_load

if __name__ == "__main__":
    # test_document_embedder()
    # test_forward_pass(no_users=4)
    wandb.init(
        project="NAML",
        # name="Stuff",  # Should be something unique and informative... Defaults to strange texts that make no sense.
        # that is perfectly OK, since w & b will log all (hyper-)parameters anyway, and therefore connects name to data
        entity="adrianlangseth",
        notes="Just testing",
        job_type="train",
        config=globals.model_params,  # Adding all settings to have them logged on the w & b GUI
    )

    """
    Here is the main call of the project. It assumes a learning system that is **parameter-free** in that everything
    is defined in the dictionaries in globals.py. 
    
    It assumes a readily available data loader. In the present version of the code it is a simple numpy table 
    kept in memory that is pushed through, but basically anything Keras can do will work. 
    The current call kicks off the user - embedding, which assumes a data tensor of shape
    (no_users, no_docs,  total-size-of-doc)
    
    Parameterization from globals:
    no_users:set by the call to the data generator. Never used in the model definition, so no need to have in globals
    no_docs: This is the max number of documents used to build a user profile. 
                globals.data_params['max_no_documents_in_user_profile']
    total-size-of-doc: Total raw-data-representation of each document. Size is defined as 
                globals.data_params['title_size'] + globals.data_params['body_size'] + 2,
                where the + 2 is one item for "category", and one for "subcategory"   
    
    All data is assumed to be integers. The textual input  (the first globals.data_params['title_size'] + 
    globals.data_params['body_size']) elements are word IDs, so assumed to be in 
    [0, globals.data_params['vocabulary_size] - 1].
    The category is in  [0, globals.data_params['no_categories] - 1].
    The sub-category is in [0, globals.data_params['no_sub_categories] - 1].
    """
    test_learning().save("./saved_model")

    wandb.finish()
