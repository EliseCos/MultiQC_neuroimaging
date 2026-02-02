from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

@dataclass
class HtmlContent:
    content: str
    metadata: dict

class Section(metaclass=ABCMeta):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def anchor(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def description(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError()
    
    @abstractmethod
    def filter_metrics(self, metrics):
        raise NotImplementedError()
    
    @abstractmethod
    def get_bundles(self):
        raise NotImplementedError()
    
    @abstractmethod
    def filter_bundles(self, bundles):
        raise NotImplementedError()
    
    @abstractmethod
    def build_html(self) -> HtmlContent:
        raise NotImplementedError()
