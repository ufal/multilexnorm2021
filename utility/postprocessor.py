class AbstractPostprocessor:
    def __init__(self, bias=1.0):
        self.bias = bias

    def __call__(self, raw, predictions):
        pass

    def process_file(self, input_path, output_path):
        with open(input_path, "r") as f:
            sentences = f.read().split("\n\n")[:-1]
            sentences = [s.split('\n') for s in sentences]

        with open(output_path, "w") as f:
            for sentence in sentences:
                for word in sentence:
                    raw, *predictions = word.split('\t')
                    predictions = [(word, float(score)) for word, score in zip(predictions[::2], predictions[1::2])]
                    prediction = self(raw, predictions)
                    f.write(f"{raw}\t{prediction}\n")
                f.write("\n")

    def rebalance(self, raw, predictions):
        predictions = [(w, s) if w != raw else (w, s*self.bias) for w, s in predictions]
        predictions = sorted(predictions, key=lambda item: item[1], reverse=True)
        return predictions


class NonePostprocessor(AbstractPostprocessor):
    def __call__(self, raw, predictions):
        predictions = self.rebalance(raw, predictions)
        return predictions[0][0]


class AlnumPostprocessor(AbstractPostprocessor):
    def __call__(self, raw, predictions):
        if raw.isdigit() and len(raw) > 1:
            return raw
        if not raw.replace("'", "").isalnum():
            return raw
        predictions = self.rebalance(raw, predictions)
        return predictions[0][0]
