class DataFeeder():
    def __init__(self, args) -> None:
        self.args = args
        self.batch_size = args.batch_size
        self.env = args.env

    def sample(self, policy) -> dict:
        raise NotImplementedError
    
    def last_log(self) -> dict:
        raise NotImplementedError

    