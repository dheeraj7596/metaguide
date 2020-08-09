from torchtext.data import Dataset
import torchtext.data as data


class TrainValFullDataset(Dataset):
    def __init__(self, lines, labels, text_field, label_field):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for i in range(len(lines)):
            temp_list = [lines[i], labels[i]]
            ex = data.Example.fromlist(temp_list, fields)
            examples.append(ex)
        super().__init__(examples, fields)

    @classmethod
    def splits(cls, text_field, label_field, train_sents, train_labels, validation_sents, val_labels, full_sents,
               full_labels, train_subtrees=False):
        """Create dataset objects for splits of the SST dataset.
        """
        train_data = cls(train_sents, train_labels, text_field, label_field)
        val_data = cls(validation_sents, val_labels, text_field, label_field)
        full_data = cls(full_sents, full_labels, text_field, label_field)
        return tuple(d for d in (train_data, val_data, full_data) if d is not None)
