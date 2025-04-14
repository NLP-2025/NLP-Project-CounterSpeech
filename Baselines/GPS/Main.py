import argparse
import os
import torch
from tqdm import tqdm
from language_quality import extract_good_candidates_by_LQ
from utils import read_candidates, initialize_train_test_dataset, to_method_object, convert_to_contexts_responses


dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import pandas as pd

def main(args):
    print('Start Main...')

    # Step 1: Load test set from CSV (your custom hateful inputs)
    test_df = pd.read_csv('./IntentConanv2/test.csv')
    test_x_text = test_df['hatespeech'].astype(str).tolist()

    # Step 2: Load generated candidate responses (from Module 1)
    candidates = candidates = read_candidates('./data/' + args.dataset + '_candidates.txt')

    # Step 3: Load training dataset to train retrieval model
    train_x_text, train_y_text, _, _ = initialize_train_test_dataset(args.dataset)
    contexts_train, responses_train = convert_to_contexts_responses(train_x_text, train_y_text)

    # Step 4: Filter candidate responses by Language Quality
    print('[Info] Filtering candidates using Language Quality Score...')
    candidates = extract_good_candidates_by_LQ(candidates, LQ_thres=0.52, num_of_generation=30000)

    # Step 5: Use TF-IDF to narrow candidate pool per test input
    print('[Info] Getting top-k relevant candidates via TF-IDF...')
    tfidf = to_method_object('TF_IDF')
    tfidf.train(contexts_train, responses_train)
    good_candidates_index = tfidf.sort_responses(test_x_text, candidates, min(args.kpq, len(candidates)))
    good_candidates = [[candidates[y] for y in x] for x in good_candidates_index]

    # Step 6: Use high-quality similarity model to select best candidate per context
    final_method_name = 'CONVERT_SIM'  # or any from your METHODS list
    print(f'[Info] Selecting best response using {final_method_name}...')
    method = to_method_object(final_method_name)
    method.train(contexts_train, responses_train)

    output_responses = []
    for i, test_input in enumerate(tqdm(test_x_text)):
        best_idx = method.rank_responses([test_input], good_candidates[i])
        output_responses.append(good_candidates[i][best_idx.item()])

    # Step 7: Save to file
    test_df['generated_counterspeech'] = output_responses
    output_path = './final_responses.csv'
    test_df.to_csv(output_path, index=False)
    print(f'\n[Done] Saved generated responses to {output_path}')
    print(f'After filtering by LQ, {len(candidates)} candidate responses remain.\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Main.py', description='choose dataset from reddit, gab, conan')
    parser.add_argument('--kpq', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='reddit', choices=['reddit', 'gab', 'conan','intentconanv2'])
    args = parser.parse_args()
    main(args)

