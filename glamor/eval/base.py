
class BaseEval:

    def eval(self, model):
        raise NotImplementedError

    @property
    def prefix(self):
        return self._prefix
