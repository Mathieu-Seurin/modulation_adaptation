from data_provider.tokenizer import CLEVRTokenizer
import json

import os
import h5py

import numpy as np

def create_h5(mode="train"):

    clever_location = '/media/datas1/dataset/clevr/CLEVR_v1.0/'

    all_games_location = os.path.join(clever_location, 'questions/CLEVR_{}_questions.json'.format(mode))

    all_games_raw = json.load(open(all_games_location))
    print("Raw games loaded")


    all_games_dict = {'image_filenames':[],
                      'questions':[],
                      'questions_raw': [],
                      'answers':[],
                      'answers_raw':[],
                      'orig_idxs':[],
                      'question_types':[]
                      }

    dict_location = os.path.join(clever_location,"dict.json")
    tokenizer = CLEVRTokenizer(dict_location)

    # Compute max len
    max_question_len = 0
    for num_game, game in enumerate(all_games_raw['questions']):

        question_raw = game['question']
        len_q = len(tokenizer.encode_question(question_raw))

        if len_q > max_question_len:
            max_question_len = len_q


    print("Max length is ",max_question_len)

    for num_game, game in enumerate(all_games_raw['questions']):

        question_raw = game['question']
        encoded_q = tokenizer.encode_question_padd(question_raw, max_len=max_question_len)
        img_idx = game['image_filename']

        orig_id = game['question_index']
        question_type = game['question_family_index']

        answer_raw = game['answer']
        answer = tokenizer.encode_answer(answer_raw)

        all_games_dict['image_filenames'].append(img_idx)
        all_games_dict['questions'].append(encoded_q)
        all_games_dict['questions_raw'].append(question_raw)
        all_games_dict['answers'].append(answer)
        all_games_dict['answers_raw'].append(answer_raw)
        all_games_dict['orig_idxs'].append(orig_id)
        all_games_dict['question_types'].append(question_type)


    assert len(all_games_dict['image_filenames']) == len(all_games_dict['questions'])
    assert len(all_games_dict['questions_raw']) == len(all_games_dict['questions'])
    assert len(all_games_dict['answers']) == len(all_games_dict['questions'])
    assert len(all_games_dict['answers_raw']) == len(all_games_dict['questions'])
    assert len(all_games_dict['orig_idxs']) == len(all_games_dict['questions'])
    assert len(all_games_dict['question_types']) == len(all_games_dict['questions'])

    print("Games parsed")



    h5_location = os.path.join("/home/sequel/mseurin/", "{}_questions.h5".format(mode))

    with h5py.File(h5_location, "w") as f:

        img_filename = f.create_dataset("image_filenames", data=np.array(all_games_dict['image_filenames'], dtype=np.string_))
        questions = f.create_dataset("questions", data=np.array(all_games_dict['questions']))
        questions_raw = f.create_dataset("questions_raw", data=np.array(all_games_dict['questions_raw'], dtype=np.string_))
        answers = f.create_dataset("answers", data= np.array(all_games_dict['answers']))
        answers_raw = f.create_dataset("answers_raw", data=np.array(all_games_dict['answers_raw'], dtype=np.string_))
        orig_idxs = f.create_dataset("orig_idxs", data=np.array(all_games_dict['orig_idxs']))
        question_types = f.create_dataset("question_types", data=np.array(all_games_dict['question_types']))


    print("This is it for ", mode)




if __name__ == "__main__":

    create_h5("train")
    create_h5("val")
