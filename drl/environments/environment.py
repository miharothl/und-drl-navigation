import abc


class Environment(abc.ABC):

    @abc.abstractmethod
    def step(self, action):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def get_action_space(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass