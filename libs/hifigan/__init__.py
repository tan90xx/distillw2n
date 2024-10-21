from .models import Generator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

from .model import hifigan

__all__ = ['hifigan']