import abc


class OrigDataModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def read_rel_triple_files(self):
        pass

    @abc.abstractmethod
    def read_attr_triple_files(self):
        pass
