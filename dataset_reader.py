from glob import glob


class Dataset(object):
    easy_dataset = glob('./data/real_and_fake_face/training_fake/easy*')
    mid_dataset = glob('./data/real_and_fake_face/training_fake/mid*')
    hard_dataset = glob('./data/real_and_fake_face/training_fake/hard*')
    real_dataset = glob('./data/real_and_fake_face/training_real/*')

    

    

