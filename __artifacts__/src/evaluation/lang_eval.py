from nltk.translate import bleu_score


def get_bleu_score(truth, transcription, *, bleu_level=4, multi_blue=False):
    if multi_blue:
        weights = [[1 / bleu_level] * b_l for b_l in range(1, bleu_level + 1)]
    else:
        weights = [1 / bleu_level] * bleu_level
    return bleu_score.sentence_bleu([truth], transcription, weights)
