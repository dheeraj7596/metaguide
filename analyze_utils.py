from sklearn.metrics import classification_report


def count_none(y):
    counts = 0
    for i in y:
        if i is None:
            counts += 1
    return counts


def get_same(y_phrase_list, y_metadata_list, ytrue_list):
    y_pred = []
    y_true_sub = []
    for i, y1 in enumerate(y_phrase_list):
        y2 = y_metadata_list[i]
        if y1 is None or y2 is None:
            continue
        if y1 == y2:
            y_pred.append(y1)
            y_true_sub.append(ytrue_list[i])
    return y_pred, y_true_sub


def get_different(y_pseudo_list, y_phrase_list, y_metadata_list, ytrue_list):
    y_pred_phrase = []
    y_pred_metadata = []
    y_true_sub = []
    y_pseudo_sub = []

    for i, y1 in enumerate(y_phrase_list):
        y2 = y_metadata_list[i]
        if y1 is None or y2 is None:
            continue
        if y1 != y2:
            y_pred_phrase.append(y1)
            y_pred_metadata.append(y2)
            y_true_sub.append(ytrue_list[i])
            y_pseudo_sub.append(y_pseudo_list[i])

    return y_pred_phrase, y_pred_metadata, y_pseudo_sub, y_true_sub


def get_None_mismatch(y1_list, y2_list, ytrue_list):
    y_pred = []
    y_true_sub = []
    for i, y1 in enumerate(y1_list):
        y2 = y2_list[i]
        if y1 is None and y2 is not None:
            y_pred.append(y2)
            y_true_sub.append(ytrue_list[i])
    return y_pred, y_true_sub


def count_common(y1_list, y2_list):
    count = 0
    for i, y1 in enumerate(y1_list):
        y2 = y2_list[i]
        if y1 == y2:
            count += 1
    return count


def analyze(y_pseudo, y_phrase, y_metadata, y_true):
    print("****************** ANALYSIS ******************")
    if len(y_metadata) == 0 or len(y_phrase) == 0:
        return

    print("Number of NONE in y_phrase: ", count_none(y_phrase))
    print("Number of NONE in y_metadata: ", count_none(y_metadata))
    print("Case-1: l_phrase = l_metadata and l_phrase ≠ NONE and l_metadata ≠ NONE")
    y_pred, y_true_sub = get_same(y_phrase, y_metadata, y_true)
    if len(y_pred) > 0:
        print(classification_report(y_pred, y_true_sub))

    print("Case-2: l_phrase ≠ l_metadata and l_phrase ≠ NONE and l_metadata ≠ NONE")
    y_pred_phrase, y_pred_metadata, y_pseudo_sub, y_true_sub = get_different(y_pseudo, y_phrase, y_metadata, y_true)
    if len(y_pred_phrase) > 0 and len(y_pred_metadata) > 0:
        print("CLASSIFICATION REPORT FOR PHRASE-Labels")
        print(classification_report(y_pred_phrase, y_true_sub))
        print("CLASSIFICATION REPORT FOR METADATA-Labels")
        print(classification_report(y_pred_metadata, y_true_sub))
        print("Number of labels matches in pseudo labels and phrase labels: ",
              count_common(y_pseudo_sub, y_pred_phrase))
        print("Number of labels matches in pseudo labels and metadata labels: ",
              count_common(y_pseudo_sub, y_pred_metadata))

    print("Case-3: l_phrase = NONE and l_metadata ≠ NONE")
    y_pred, y_true_sub = get_None_mismatch(y_phrase, y_metadata, y_true)
    if len(y_pred) > 0:
        print(classification_report(y_pred, y_true_sub))

    print("Case-4: l_metadata = NONE and l_phrase ≠ NONE")
    y_pred, y_true_sub = get_None_mismatch(y_metadata, y_phrase, y_true)
    if len(y_pred) > 0:
        print(classification_report(y_pred, y_true_sub))
