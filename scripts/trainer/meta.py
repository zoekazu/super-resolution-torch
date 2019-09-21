
import abc


class TrainerMeta(abc.ABC):

    @abc.abstractmethod
    def train(self):
        pass
