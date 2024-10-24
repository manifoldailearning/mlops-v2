# Test the model_training function in the train.py file

from src.train import model_training


def test_model_training():
    '''
    Test the model_training function
    '''
    study = model_training()
    
    assert study.best_trial is not None
