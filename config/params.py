import yaml


class Params:
    def load(self, argv):
        args = self.convert_to_args(argv)
        if "config" not in args:
#            args["config"] = "config/sr_small_wiki.yaml"
#            args["config"] = "config/large_byt5.yaml"
            args["config"] = "config/hr_small_finetune.yaml"
        self.load_saved_config(args["config"])
        self.load_state_dict(args)
        self.update_state_dict(argv)
        return self

    def convert_to_args(self, argv):
        args = {}
        argv = zip(argv[::2], argv[1::2])
        for arg1, arg2 in argv:
            d = args
            keys = arg1[2:].split(".")
            for k in keys[:-1]:
                if k not in d: d[k] = {}
                d = d[k]
            d[keys[-1]] = arg2

        return args

    def update_state_dict(self, argv):
        argv = zip(argv[::2], argv[1::2])
        for arg1, arg2 in argv:
            d = self.__state_dict__
            keys = arg1[2:].split(".")
            for k in keys[:-1]:
                if k not in d: d[k] = {}
                d = d[k]

            if keys[-1] in d:
                d[keys[-1]] = type(d[keys[-1]])(arg2)
            else:
                d[keys[-1]] = arg2

    def load_state_dict(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                if hasattr(self, k):
                    v = getattr(self, k).load_state_dict(v)
                else:
                    v = Params().load_state_dict(v)
            elif hasattr(self, k):
                v = type(getattr(self, k))(v)

            setattr(self, k, v)

        return self

    def state_dict(self):
        return self.__state_dict__

    def load_saved_config(self, config):
        with open(config, "r", encoding="utf-8") as f:
            self.__state_dict__ = yaml.full_load(f)
        self.load_state_dict(self.__state_dict__)

    def save(self, json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            yaml.dump(self.state_dict(), f, indent=4)

    def __str__(self):
        return yaml.dump(self.state_dict(), indent=4)
