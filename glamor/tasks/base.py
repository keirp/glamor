
class BaseTaskDist:

    @property
    def name(self):
        return self._name

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class NoneTaskDist(BaseTaskDist):

    def __iter__(self):
        return self

    def __next__(self):
        return None
