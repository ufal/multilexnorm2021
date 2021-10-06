import os
import os.path
from utility.postprocessor import NonePostprocessor, AlnumPostprocessor, AspellPostprocessor


class OutputAssembler:
    def __init__(self, directory, args, dataset):
        self.directory = directory
        self.dataset = dataset
        self.postprocessing = {
            "none": NonePostprocessor,
            "alnum": AlnumPostprocessor,
            "aspell": AspellPostprocessor
        }[args.postprocessing.type](args.postprocessing.bias)

        self.cache = {}

    def step(self, output_dict):
        output_dict = (output_dict["predictions"], output_dict["scores"], output_dict["sentence_ids"], output_dict["word_ids"])
        for word_preds, scores, sent_id, word_id in zip(*output_dict):
            word_preds = [w.replace('\n', '').replace('\t', ' ') for w in word_preds]
            pairs = list(zip(word_preds, scores))

            self.cache.setdefault(sent_id, {})[word_id] = pairs

    def flush(self):
        predictions = self.assemble(self.cache)
        inputs = self.dataset.inputs

        raw_path = f"{self.directory}/raw_outputs.txt"
        postprocessed_path = f"{self.directory}/outputs.txt"

        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        with open(raw_path, "w") as f:
            for i, input_sentence in enumerate(inputs):
                for j, input_word in enumerate(input_sentence):
                    try:
                        prediction_string = '\t'.join([f"{w}\t{s}" for w, s in predictions[i][j]])
                    except:
                        print(i, j, len(predictions[i]))
                        for k, p in enumerate(predictions[i]):
                            print(k, p)
                        print(flush=True)
                        exit()
                    line = f"{input_word}\t{prediction_string}"
                    f.write(f"{line}\n")
                f.write("\n")

        self.postprocessing.process_file(raw_path, postprocessed_path)

    def assemble(self, prediction_dict):
        prediction_list = []
        for sent_id, raw_sentence in enumerate(self.dataset.inputs):
            prediction_list.append(
                [prediction_dict.get(sent_id, {}).get(word_id, [(raw_word, 0.0)]) for word_id, raw_word in enumerate(raw_sentence)]
            )

        return prediction_list
