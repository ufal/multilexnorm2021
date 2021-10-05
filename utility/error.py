class Error:
    def __init__(self, gold_dataset):
        super().__init__()
        self.gold_dataset = gold_dataset

    def __call__(self, predictions, output_directory=None):
        inputs, gold_outputs = self.gold_dataset.inputs, self.gold_dataset.outputs
        lai, accuracy, error, precision, recall, f1, false_negatives, false_positives = self.calculate_error(inputs, gold_outputs, predictions)

        if output_directory:
            self.log(output_directory, inputs, gold_outputs, predictions, false_positives, false_negatives)

        return lai, accuracy, error, precision, recall, f1

    def log(self, output_directory, inputs, gold_outputs, predictions, false_positives, false_negatives):
        with open(f"{output_directory}/aligned_outputs.txt", "w") as f:
            for i in range(len(inputs)):
                lens = [max(len(w0), len(w1), len(w2)) for w0, w1, w2 in zip(inputs[i], gold_outputs[i], predictions[i])]
                form = ' '.join([f"{{: >{l}}}" for l in lens])

                f.write(f"INPT: {form.format(*inputs[i])}\n")
                f.write(f"GOLD: {form.format(*gold_outputs[i])}\n")
                f.write(f"PRED: {form.format(*predictions[i])}\n\n")

            print(flush=True)

        for name in ["false_positives", "false_negatives"]:
            with open(f"{output_directory}/{name}.txt", "w") as f:
                indices = locals()[name]
                f.write(f"{len(indices)}\n\n")

                for sent_index, word_index in indices:
                    lens = [max(len(w0), len(w1), len(w2)) for w0, w1, w2 in zip(inputs[sent_index], gold_outputs[sent_index], predictions[sent_index])]
                    lens = [f"{{: >{l}}}" for l in lens]
                    lens = lens[:word_index] + [" >>", lens[word_index], "<< "] + lens[word_index+1:]
                    form = ' '.join(lens)

                    f.write(f"INPT: {form.format(*inputs[sent_index])}\n")
                    f.write(f"GOLD: {form.format(*gold_outputs[sent_index])}\n")
                    f.write(f"PRED: {form.format(*predictions[sent_index])}\n\n")


    # based on multilexnorm/scripts/normEval.py + added precision and recall
    def calculate_error(self, raw, gold, pred):
        cor = 0
        changed = 0
        total = 0

        false_negatives, false_positives = [], []
        true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0

        if len(gold) != len(pred):
            print('Error: gold normalization contains a different numer of sentences(' + str(len(gold)) + ') compared to system output(' + str(len(pred)) + ')')

        for sent_index, (sentRaw, sentGold, sentPred) in enumerate(zip(raw, gold, pred)):
            if len(sentGold) != len(sentPred):
                print('Error: a sentence has a different length in you output, check the order of the sentences')
            for word_index, (wordRaw, wordGold, wordPred) in enumerate(zip(sentRaw, sentGold, sentPred)):
                if wordRaw != wordGold:
                    changed += 1
                if wordGold == wordPred:
                    cor += 1

                if wordRaw == wordGold and wordGold == wordPred:
                    true_negative += 1
                if wordRaw == wordGold and wordGold != wordPred:
                    false_positive += 1
                    false_positives.append((sent_index, word_index))
                if wordRaw != wordGold and wordGold != wordPred:
                    false_negative += 1
                    false_negatives.append((sent_index, word_index))
                if wordRaw != wordGold and wordGold == wordPred:
                    true_positive += 1

                total += 1

        accuracy = cor / total if total > 0 else 0.0
        lai = (total - changed) / total if total > 0 else 0.0
        err = (accuracy - lai) / (1-lai) if lai < 1.0 else float("-inf")

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0.0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0.0 else 0.0
        f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

        return lai, accuracy, err, precision, recall, f1, false_positives, false_negatives
