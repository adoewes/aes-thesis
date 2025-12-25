import numpy as np
import os
import re
import joblib
import xgboost
import shap
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def predict_score_mixed_sbert(sbert_feature, ease_feature, lang_error):    

    #model_name = "model_ease6_punc_spell_unique_similarity_sbert_asap_778"
    #model_name = "model_ease6_punc_spell_unique_similarity_sbertandbow_langerror_asap6_780_normalized"
    #model_name = "model_asap1_counterfactual"
    model_name = "model_asap1_counterfactual.json"

    #model_name = re.sub("asap", domain, model_name)
    #=================================================
    #print("model name : ", model_name)

    script_dir = os.path.dirname(__file__)

    model_filename = os.path.join(
        script_dir, '../sbert/', model_name)

    #model = joblib.load(model_filename)
    model = xgboost.Booster()
    model.load_model(model_filename)
    #model = xgboost.load_model(model_filename)

    #====================================================================
    features = get_ease_selection_sbert_normalized(ease_feature, sbert_feature, lang_error)

    #write_features_to_file(features)

    #print("features shape: ", features.shape)   
    
    features_dmatrix = xgboost.DMatrix(features.reshape(1, -1), feature_names=get_feature_names_extended())

    score = model.predict(features_dmatrix)[0]

    #print("Real score: ", score)
    
    score_float = np.round(score,3)

    score_int = int(np.round(score))

    #print(score)

    #feedback = xgb_score_explanation(features, features_dmatrix, model)

    #return(score, feedback)
    return (score_float, score_int)

def get_ease_selection_sbert(x_ease416, x_sbert, lang_error):

    x_12 = x_ease416[:, :12]
    x_spell_error = x_ease416[:, -2].reshape(-1,1)
    x_unique_words = x_ease416[:, -1].reshape(-1,1)

    # columns 2,3,4 --> comma counts, Apostrophe counts, other punctuation counts
    # we sum all of the punctuation counts
    x_punc_counts = np.sum(x_12[:, [2,3,4]], axis=1)
    x_punc_counts = x_punc_counts.reshape(-1, 1)

    x_6 = np.delete(x_12, [2,3,4,7,9,11], axis=1)

    sbert_similarity = get_similarity_score(x_sbert)

    #answer_bow = x_ease416[:, 12:-2]

    #similarity_bow = get_similarity_bow(answer_bow)

    lang_error = np.array(lang_error).reshape(-1,1)

    ease_selection_sbert = np.concatenate((x_6, x_punc_counts, x_spell_error, x_unique_words, sbert_similarity, lang_error, x_sbert), axis=1)

    print("Feature size : ", len(ease_selection_sbert))

    # This one is the old feature vectors of 778 dimensions, without similarity_bow and lang_error
    #ease_selection_sbert = np.concatenate((x_6, x_punc_counts, x_spell_error, x_unique_words, sbert_similarity, x_sbert), axis=1)
    print("Mixed feature shape: ", ease_selection_sbert.shape)
    print("Final feature: ", ease_selection_sbert)

    return ease_selection_sbert


def get_ease_selection_sbert_normalized(x_ease416, x_sbert, lang_error):
    x_12 = x_ease416[:, :12]
    x_spell_error = x_ease416[:, -2].reshape(-1,1)
    x_unique_words = x_ease416[:, -1].reshape(-1,1)

    # columns 2,3,4 --> comma counts, Apostrophe counts, other punctuation counts
    # we sum all of the punctuation counts
    x_punc_counts = np.sum(x_12[:, [2,3,4]], axis=1)
    x_punc_counts = x_punc_counts.reshape(-1, 1)

    x_6 = np.delete(x_12, [2,3,4,7,9,11], axis=1)

    #print(x_6)

    sbert_similarity = get_similarity_score(x_sbert)

    #answer_bow = x_ease416[:, 12:-2]

    #similarity_bow = get_similarity_bow(answer_bow)

    lang_error = np.array(lang_error).reshape(-1,1)

    #ease_selection_sbert = np.concatenate((x_6, x_punc_counts, x_spell_error, x_unique_words, sbert_similarity, similarity_bow, lang_error, x_sbert), axis=1)
    ease_selection_sbert = np.concatenate((x_6, x_punc_counts, x_spell_error, x_unique_words, sbert_similarity, lang_error, x_sbert), axis=1)

    # This one is the old feature vectors of 778 dimensions, without similarity_bow and lang_error
    #ease_selection_sbert = np.concatenate((x_6, x_punc_counts, x_spell_error, x_unique_words, sbert_similarity, x_sbert), axis=1)
    
    # Normalize spelling error / number of word
    spell_error = ease_selection_sbert[:,7]
    word_count = ease_selection_sbert[:,1]
    spell_error_normalized = spell_error / word_count
    
    ease_selection_sbert[:,7] = spell_error_normalized

    # Normalize language error / answer length
    lang_error = ease_selection_sbert[:,11]
    answer_length = ease_selection_sbert[:,0]
    lang_error_normalized = lang_error / answer_length
    
    ease_selection_sbert[:,11] = lang_error_normalized


    #print("Mixed feature shape (normalized): ", ease_selection_sbert.shape)
    #`print("Final feature: ", ease_selection_sbert)

    return ease_selection_sbert


def get_similarity_bow(answer_bow):

    script_dir = os.path.dirname(__file__)

    prompt_bow_filename = os.path.join(
        script_dir, '../sbert/', 'prompt_bow_asap1')

    prompt_bow = joblib.load(prompt_bow_filename)

    similarity_bow = cosine_similarity(answer_bow.reshape(1,-1), prompt_bow.reshape(1,-1))

    return similarity_bow


def get_similarity_score(x_sbert):

    script_dir = os.path.dirname(__file__)

    prompt_filename = os.path.join(
        script_dir, '../sbert/', 'sbert_prompt_asap1')

    prompt = joblib.load(prompt_filename)

    similarity_score = cosine_similarity(x_sbert, prompt)

    return similarity_score


def get_feature_names():
    ease_feats = ['Answer Length', 'Word Counts', 'Average Word Length', 'Good n-gram', 'Prompt Overlap', 
              'Prompt Overlap (synonyms)', 'Punctuation Counts', 'Spelling Error', 'Unique Words', 'Prompt Similarity SBert']

    sbert_feats = []
    sbert_dim = 768

    for i in range(0, sbert_dim):
        fname = "sbert_" + str(i) 
        sbert_feats.append(fname)

    feature_names = ease_feats + sbert_feats

    #print("len feature names: ", len(feature_names))

    return feature_names


def get_feature_names_extended():
    ease_feats = ['Answer Length', 'Word Counts', 'Average Word Length', 'Good n-gram', 'Prompt Overlap', 
              'Prompt Overlap (synonyms)', 'Punctuation Counts', 'Spelling Error', 'Unique Words', 'Prompt Similarity SBert']        

    sbert_feats = []
    sbert_dim = 768

    for i in range(0, sbert_dim):
        fname = "sbert_" + str(i) 
        sbert_feats.append(fname)
    
    #prompt_similarity_bow = ["Prompt Similarity BOW"]
    lang_error = ["Language Error"]
    
    #feature_names = ease_feats + prompt_similarity_bow + lang_error + sbert_feats 
    feature_names = ease_feats + lang_error + sbert_feats

    #print("len feature names: ", len(feature_names))
    
    return feature_names


def xgb_score_explanation(answer, dmatrix, model):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(answer)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    shap.force_plot(explainer.expected_value, shap_values, answer, text_rotation=15,
                    feature_names=get_feature_names_extended(), matplotlib=True, show=False)

    plt.savefig("aestool/static/graphs/cc.png", bbox_inches='tight')

    plt.clf()

    shap.decision_plot(explainer.expected_value, shap_values, answer, feature_names=get_feature_names_extended(), show=False, highlight=0)

    plt.savefig("aestool/static/graphs/dd.png", bbox_inches='tight')

    a = get_feature_names_extended()
    b = shap_values.reshape(780,)

    feedback = get_feedback_messages(a,b)    

    return feedback


def get_feedback_messages(feature_names, shap_values):

    # For 778 version, replace [:12] with [:10]
    
    aa = feature_names[:12]

    bb = (shap_values.tolist())[:12]

    features_contribution = dict(zip(aa,bb))

    print(features_contribution)
    feedback_names = ["Answer Length", "Relevance to the prompt", "Grammar & Mechanics", "Word Choice"]
    feedback_values = []

    length_features = features_contribution['Answer Length'] + features_contribution['Word Counts']
    feedback_values.append(length_features)

    relevance_features = features_contribution['Prompt Overlap'] + features_contribution['Prompt Overlap (synonyms)']
    feedback_values.append(relevance_features)

    grammar_features = features_contribution['Good n-gram'] + features_contribution['Punctuation Counts']
    feedback_values.append(grammar_features)

    difficult_word_features = features_contribution['Average Word Length'] + features_contribution['Unique Words']
    feedback_values.append(difficult_word_features)
    
    #language_features = grammar_features + difficult_word_features
    #feedback_values.append(language_features)

    cc_pos = [(i,j) for (i,j) in zip(feedback_names, feedback_values) if j >= 0]
    cc_pos = sorted(cc_pos, key=lambda x:x[1], reverse=True)

    cc_neg = [(i,j) for (i,j) in zip(feedback_names, feedback_values) if j < 0]
    cc_neg = sorted(cc_neg, key=lambda x:x[1])

    pos_feedback = [i for (i,j) in cc_pos]
    neg_feedback = [i for (i,j) in cc_neg]

    print("CONTRIBUTION : ", list(zip(feedback_names,feedback_values)))

    feedback = (pos_feedback, neg_feedback)

    return feedback


def write_features_to_file(features):
    script_dir = os.path.dirname(__file__)

    filename = os.path.join(
        script_dir, '../sbert/', 'debug_features.txt')

    #a = "Coba deh"
    with open(filename, 'a') as file:
        np.savetxt(file, features.reshape(1,-1))